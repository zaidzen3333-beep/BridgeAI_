import os
import json
import asyncio
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import warnings

warnings.filterwarnings("ignore")

# --- Import your "Employee" Agents ---
from rag_agent import run_rag_agent
from sql_agent import run_sql_agent

# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("🕸️ Booting up LangGraph Orchestrator (Streaming Mode)...")

# 2. Define the Graph State
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    invoice_data: dict  # <--- New: Holds the JSON extracted from the image
    sources: Annotated[list, operator.add]
    used_agents: Annotated[list, operator.add]

# 3. Initialize the Master LLM
llm = ChatGroq(temperature=0, model_name="meta-llama/llama-4-scout-17b-16e-instruct", api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────
# 4. The Streaming Orchestrator
#
#    CONTEXT ANCHORING FIX:
#    Previously, chat history was passed as a single concatenated string
#    stuffed into the SystemMessage. The LLM couldn't distinguish old
#    turns from the new question and would re-answer old prompts.
#
#    Now we accept `past_messages` as a list of dicts and convert them
#    to proper LangChain HumanMessage / AIMessage objects. This gives
#    the LLM a clear conversational timeline where the LATEST
#    HumanMessage is always the user's current question.
# ─────────────────────────────────────────────────────────────────────
async def stream_orchestrator(question: str, past_messages: list):
    """
    Builds the LangGraph and yields real-time SSE tokens to the frontend.

    Args:
        question:       The user's latest question (plain string).
        past_messages:  List of dicts from Supabase, e.g.
                        [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]
    """

    # ── Reconstruct a flat history_str for sub-agents that still need it ──
    # run_rag_agent(question, history_str) expects a plain string.
    # We build it here once and capture it in the closure.
    # Defensive: skip any malformed entries that lack 'role' or 'content'.
    try:
        history_str = "\n".join(
            [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
             for msg in past_messages 
             if isinstance(msg, dict) and msg.get('content')]
        )
    except Exception:
        history_str = ""

    # --- A. DEFINE THE TOOLS ---
    @tool
    def search_structured_database(search_query: str) -> str:
        """
        CRITICAL ROUTING RULE: USE THIS TOOL ONLY IF the user provides a specific
        HS Code (e.g., '8501', '8703'), a specific product name (e.g., 'cars', 'motors'),
        or explicitly asks for numeric tariff rates, taxes, duties, or inventory data.
        DO NOT use this tool for general questions about customs procedures, definitions,
        or how things work. This tool queries a SQL database with structured numeric data.
        USE THIS TOOL for: 
        1. Numerical data (tariffs, tax rates).
        2. PRODUCT DEFINITIONS/DESCRIPTIONS for specific HS Codes. 
        If the user asks 'What is HS 8703?' or 'What does this code stand for?', 
        use this tool to check the 'description_fr' in the tariffs table.
        """
        # 1. Check if the query looks like an HS code (contains numbers)
        if any(char.isdigit() for char in search_query):
            # Strip dots and spaces ONLY for codes
            clean_query = search_query.replace(".", "").replace(" ", "")
        else:
            # Keep spaces for names like "electric motors"
            clean_query = search_query
        
        # We force a command that the SQL Agent understands perfectly
        task = f"Find the hs_code, description_fr, and import_duty_rate for HS Code prefix {clean_query} in the morocco_tariffs table."
        return run_sql_agent(task)

    @tool
    def search_legal_documents(search_query: str) -> str:
        """
        CRITICAL ROUTING RULE: USE THIS TOOL FOR general questions about trade laws,
        customs procedures, regulations, or definitions (e.g., 'What are the duties of...',
        'How do I clear customs', 'What is the BADR system', 'Explain the import process').
        Use this when the query is conceptual, process-oriented, or asks about how
        something works. This tool searches legal PDFs and regulatory documents.
        """
        return run_rag_agent(search_query, history_str)

    tools = [search_structured_database, search_legal_documents]
    llm_with_tools = llm.bind_tools(tools)

    # --- B. DEFINE THE GRAPH NODES ---
    def agent_node(state: GraphState):
        """The 'Brain'."""
        # Tool choice remains automatic. Forcing a function call can make Groq
        # fail on long document-audit prompts with `failed_generation`.
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def execute_tools_node(state: GraphState):
        """The 'Hands'."""
        last_message = state["messages"][-1]
        new_messages = []
        new_sources = []
        new_agents = []
        
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            
            if name == "search_structured_database":
                # Ensure we only get the string result from the SQL agent
                raw_result = run_sql_agent(args.get("search_query", question))
                
                # If your SQL agent returns a dict, extract just the output!
                if isinstance(raw_result, dict) and "output" in raw_result:
                    result = raw_result["output"]
                else:
                    result = str(raw_result)
                    
                new_agents.append("SQL Agent")
                new_sources.append("Enterprise SQL Database")
                
            elif name == "search_legal_documents":
                # run_rag_agent still expects (question, history_str) as strings
                result, sources = run_rag_agent(args.get("search_query", question), history_str)
                new_agents.append("RAG Agent")
                new_sources.extend(sources)
            else:
                result = "Error: Unknown tool requested."
                
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=name))
            
        return {"messages": new_messages, "sources": new_sources, "used_agents": new_agents}

    def should_continue_edge(state: GraphState):
        """The 'Traffic Light'."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "execute_tools" 
        return END 

    # --- C. BUILD AND COMPILE THE GRAPH ---
    workflow = StateGraph(GraphState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue_edge, ["execute_tools", END])
    workflow.add_edge("execute_tools", "agent") 
    app = workflow.compile()

    # ─────────────────────────────────────────────────────────────────
    # D. BUILD THE NATIVE LANGCHAIN MESSAGE ARRAY
    #
    #    This is the CONTEXT ANCHORING fix. Instead of dumping history
    #    into a string, we construct a proper message timeline:
    #
    #      1. SystemMessage  — core persona instructions ONLY
    #      2. HumanMessage / AIMessage — from past_messages (history)
    #      3. HumanMessage  — the user's LATEST question (always last)
    #
    #    The LLM sees a clear turn-by-turn conversation and will
    #    always answer the FINAL HumanMessage.
    # ─────────────────────────────────────────────────────────────────
    system_msg = SystemMessage(content=(
        "You are the Master Orchestrator for an Enterprise BridgeAI. "
        "Your MOST IMPORTANT job is to answer from BridgeAI data sources whenever the question requires factual customs, tariff, document, or regulatory information.\n\n"
        "ROUTING RULES (follow strictly):\n"
        "1. For comprehensive answers, you are ENCOURAGED to use BOTH 'search_structured_database' AND 'search_legal_documents' in parallel.\n"
        "2. Always use 'search_structured_database' to find SPECIFIC "
        "HS Codes, product names, numeric tariff rates, tax percentages, duty amounts, or inventory data.\n"
        "3. Always use 'search_legal_documents' to find procedural requirements, regulatory documents, certificates, customs laws, and definitions.\n"
        "4. If a user asks about importing, exporting, or trading a specific product, call BOTH TOOLS when possible: use SQL for the rates, and RAG for the required documents/procedures.\n"
        "5. CRITICAL: NEVER explain your thought process. NEVER say 'I will use...' or 'Let me search...'. DO NOT output any introductory filler sentences. If you need a tool, just execute the tool silently without outputting any text.\n\n"
        "Once you have the tool results, synthesize them into a single, cohesive, professional answer. "
        "Always answer the user's LATEST message. Do NOT re-answer previous questions."
        "Always respond in the SAME language as the user."
        "If the user speaks French, respond in French. If Arabic, respond in Arabic. If English, respond in English."
        "If the user does not specify a language, respond in the language of their last message."
        "STRICT ANSWERING RULE:\n"
        "1. You must answer ONLY based on the information returned by the tools (search_legal_documents or search_structured_database).\n"
        "2. If the tools return 'No results found', 'I don't know', or if the information "
        "provided by the tools does NOT contain the answer to the user's question, you must "
        "politely inform the user that you do not have that specific information in your records.\n"
        "3. DO NOT use your own internal knowledge, general training data, or the internet to "
        "answer. If it is not in the tool output, you do not know it.\n\n"
        "RESPONSE FORMAT RULES:\n"
        "1. Start with a short direct answer or decision in one or two sentences.\n"
        "2. Then organize the response with clear Markdown headings.\n"
        "3. Use a Markdown table when the answer compares documents, HS codes, duties, risks, or requirements.\n"
        "4. Add a section named 'Points a verifier' or its equivalent in the user's language when the result contains uncertainty, missing data, or possible risk.\n"
        "5. Add a final section named 'Source consultee' or its equivalent, mentioning only the BridgeAI source used: SQL database, legal document vectors, or both.\n"
        "6. If a tool fails or returns no usable result, do not expose raw technical errors. Say that BridgeAI could not verify that specific point in its available records.\n\n"
    ))

    # Convert Supabase history dicts → LangChain message objects
    # Defensive: skip entries that are not dicts or lack content
    history_messages = []
    for msg in past_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:  # Skip empty messages
            continue
        if role == "user":
            history_messages.append(HumanMessage(content=str(content)))
        elif role == "ai":
            history_messages.append(AIMessage(content=str(content)))
        # Skip any other roles (system, etc.)

    # ── De-duplicate: if the latest history message is already the user's
    #    current question (because save_message ran before get_session_history),
    #    don't append it twice. Otherwise, add it as the final HumanMessage. ──
    if history_messages and isinstance(history_messages[-1], HumanMessage) and history_messages[-1].content == question:
        # The question is already the last message in history — no need to duplicate
        constructed_messages = [system_msg] + history_messages
    else:
        # Append the new question as the absolute final HumanMessage
        constructed_messages = [system_msg] + history_messages + [HumanMessage(content=question)]

    initial_state = {
        "messages": constructed_messages,
        "sources": [],
        "used_agents": []
    }

    # --- E. EXECUTE AS A REAL-TIME STREAM ---
    try:
        buffer = ""
        in_thought = False
        
        # astream_events lets us watch the AI as it works, step by step!
        async for event in app.astream_events(initial_state, version="v2", config={"run_name": "Master_Orchestrator"}):
            kind = event["event"]
            
            # Catch when a tool starts running to send a UI status update
            if kind == "on_tool_start":
                tool_name = event["name"]
                if "structured_database" in tool_name:
                    yield f"data: {json.dumps({'type': 'status', 'content': '📊 Querying Enterprise Database...'})}\n\n"
                elif "legal_documents" in tool_name:
                    yield f"data: {json.dumps({'type': 'status', 'content': '📚 Scanning Legal PDFs...'})}\n\n"
                    
            # Catch the AI generating its final answer text
            elif kind == "on_chat_model_stream":
                
                # 🚀 Only stream if the speaker is the Master 'agent' node!
                # This ignores any sub-LLMs running inside your SQL or RAG tools.
                if event["metadata"].get("langgraph_node") == "agent":
                    
                    chunk = event["data"]["chunk"]
                    # Guard against BOTH tool_calls and tool_call_chunks
                    # (some LangChain versions emit tool_call_chunks during streaming)
                    has_tool_calls = getattr(chunk, "tool_calls", None)
                    has_tool_chunks = getattr(chunk, "tool_call_chunks", None)
                    
                    if not has_tool_calls and not has_tool_chunks and chunk.content:
                        buffer += chunk.content
                        
                        while buffer:
                            if in_thought:
                                if "</think>" in buffer:
                                    buffer = buffer.split("</think>", 1)[1]
                                    in_thought = False
                                else:
                                    break
                            else:
                                if "<think>" in buffer:
                                    before, after = buffer.split("<think>", 1)
                                    if before:
                                        yield f"data: {json.dumps({'type': 'token', 'content': before})}\n\n"
                                        await asyncio.sleep(0)
                                    buffer = after
                                    in_thought = True
                                else:
                                    last_lt = buffer.rfind("<")
                                    if last_lt != -1 and "<think>".startswith(buffer[last_lt:]):
                                        if last_lt > 0:
                                            yield f"data: {json.dumps({'type': 'token', 'content': buffer[:last_lt]})}\n\n"
                                            await asyncio.sleep(0)
                                        buffer = buffer[last_lt:]
                                        break
                                    else:
                                        yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                                        await asyncio.sleep(0)
                                        buffer = ""
                                        break

        if buffer and not in_thought:
            yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
            await asyncio.sleep(0)

        # When the loop finishes, send the "done" signal
        final_meta = {
            "type": "done", 
            "sources": ["Enterprise Database", "Legal Document Vectors"], 
            "agents": "LangGraph Orchestrator"
        }
        yield f"data: {json.dumps(final_meta)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'token', 'content': f'Error: {str(e)}'})}\n\n"
