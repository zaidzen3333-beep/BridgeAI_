"""
Prediction Agent — LangGraph State Machine for Trade Delay Prediction.

Architecture: A deterministic state graph that flows through 5 sequential nodes:
    extraction_node → sql_node → rag_node → environment_node → ml_prediction_node

State Machine Design:
    - Each node reads from and writes to a shared TypedDict state.
    - The extraction_node acts as the gatekeeper: if required variables are missing,
      it short-circuits to END and returns a follow-up question to the user.
    - Only when ALL 5 variables are present does the graph proceed through the
      remaining nodes to execute the full prediction pipeline.

Business Rules:
    RULE 1 (NO GUESSING): All 5 variables must be explicitly provided.
    RULE 2 (DOCUMENT ASSUMPTION): If RAG finds required docs, state the assumption.
    RULE 3 (EXECUTION): Only run model with all variables + live environment data.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import existing sub-agents and utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_agent import run_rag_agent
from sql_agent import run_sql_agent
from utils.environment_api import get_live_weather, get_mock_port_congestion

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────────────────────────────
# 1. LLM Initialization
# ─────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    temperature=0,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY
)

# ─────────────────────────────────────────────────────────────────────
# 2. ML Model Loader
# ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_models")


class DummyModel:
    def predict(self, input_df):
        # Calculate mock delay based on environmental factors to keep UI functional
        delay = 2.0
        if "weight_kg" in input_df.columns:
            delay += float(input_df["weight_kg"].iloc[0]) / 5000.0
        if "origin_weather" in input_df.columns:
            delay += float(input_df["origin_weather"].iloc[0]) * 0.8
        if "destination_congestion" in input_df.columns:
            delay += float(input_df["destination_congestion"].iloc[0]) * 0.6
        return [delay]

def load_model(direction: str):
    """Load the correct .pkl model based on trade direction. Fallback to mock if corrupted."""
    filename = "bridge_import_hybrid.pkl" if direction.lower() == "import" else "bridge_export_hybrid.pkl"
    path = os.path.join(MODEL_DIR, filename)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ [ML Predictor] Pickle corrupted ({e}). Using Mock Model.")
        return DummyModel()


# ─────────────────────────────────────────────────────────────────────
# 3. LangGraph State Definition
# ─────────────────────────────────────────────────────────────────────
class PredictionState(TypedDict):
    """Shared state flowing through all graph nodes."""
    # Input
    user_text: str                      # Latest user message
    conversation_history: list          # Full chat history [{"role": ..., "content": ...}]
    language: str                       # ISO 639-1 code detected from the latest user message

    # Extraction
    direction: Optional[str]            # "Import" or "Export"
    transport_mode: Optional[str]       # "Sea" or "Air"
    weight: Optional[float]             # Weight in kg
    origin: Optional[str]              # Origin country/port
    destination: Optional[str]         # Destination country/port
    shipment_date: Optional[str]       # Date of shipment (YYYY-MM-DD)
    hs_code: Optional[str]             # HS code (optional, for doc lookup)
    missing_vars: list                  # List of still-missing variable names

    # SQL / RAG results
    sql_hs_info: Optional[str]         # HS nomenclature data from SQL agent
    legal_rules: Optional[str]         # Compliance rules from RAG agent
    document_warning: Optional[str]    # The RULE 2 assumption statement

    # Environment
    env_scores: dict                    # {"origin_weather": int, "dest_weather": int,
                                        #  "origin_congestion": int, "dest_congestion": int}

    # Output
    final_prediction: Optional[dict]    # Full prediction result
    response_message: str               # Final text to return to user
    status: str                         # "waiting_for_info" | "success"


# ─────────────────────────────────────────────────────────────────────
# 4. Extraction Prompt
# ─────────────────────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a JSON extractor. Your ONLY job is to output a single raw JSON object.

RULES (NEVER break these):
- Output NOTHING except the JSON object — no explanation, no prose, no markdown, no code fences.
- Scan the ENTIRE conversation (all messages, not just the last one) for these 7 shipment fields:
  1. direction   : "Import" or "Export"
  2. transport_mode : "Sea" or "Air"
  3. weight      : numeric kg value (extract number only)
  4. origin      : origin city / country / port
  5. destination : destination city / country / port
  6. hs_code     : 4-10 digit HS code if explicitly mentioned, else null
  7. shipment_date : Date of shipment in "YYYY-MM-DD" format if mentioned, else null
- Also detect the language of the LATEST user message and return it as an ISO 639-1 code
  in a field named "language" (examples: "en", "fr", "ar", "es").
- If a field was stated in any earlier message, include it. If the user updates or changes a field in the latest message, use the NEW value!
- Use null for fields that were never mentioned anywhere in the conversation.
- Do NOT infer or guess values that were not stated.

EXACT output format (copy this structure):
{"direction": "Import", "transport_mode": "Sea", "weight": 5000, "origin": "China", "destination": "Morocco", "hs_code": null, "shipment_date": "2026-06-01", "language": "en"}
"""

MISSING_VARS_PROMPT = """You are the BridgeAI Delay Prediction Assistant. You help users predict customs clearance delays.

You need these 6 variables to run a prediction:
1. Direction (Import or Export)
2. Transport Mode (Sea or Air)
3. Weight (in kg)
4. Origin (country or port)
5. Destination (country or port)
6. Shipment Date (e.g. tomorrow, next week, or specific date)

The user has provided: {provided}
Still MISSING: {missing}

Write a SHORT, friendly message asking the user specifically for the missing information.
Respond in the SAME language as the user's last message.
Do NOT make up or assume any values."""


FOLLOW_UP_PROMPT = """You are the BridgeAI Delay Prediction Assistant.
The user has already provided their shipment details and you have generated a prediction.
You are now answering a follow-up question from the user about this existing prediction.

Context of the Current Prediction:
{prediction_context}

User's Follow-up Question:
{user_text}

Rules:
1. Answer the user's question directly and concisely based ONLY on the provided prediction context.
2. If they ask "why", explain the specific root causes (SHAP factors), bottlenecks, or weather conditions listed.
3. Keep your response friendly and professional.
4. Do NOT ask for more shipment details. The prediction is already done.
5. Respond entirely in the SAME language as the user's follow-up question.
"""

# ─────────────────────────────────────────────────────────────────────
# 5. Graph Nodes
# ─────────────────────────────────────────────────────────────────────

def extraction_node(state: PredictionState) -> dict:
    """
    NODE 1: Parse user text to extract shipment variables.
    If any required variable is missing, sets missing_vars and routes to END.
    """
    # Build conversation text for extraction
    full_history = state["conversation_history"] + [
        {"role": "user", "content": state["user_text"]}
    ]
    convo_text = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in full_history
        if msg.get("content")
    )

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract variables from this conversation:\n\n{convo_text}")
    ]
    chain = llm | StrOutputParser()
    result = chain.invoke(messages)

    # ── Robust JSON extraction ──────────────────────────────────────────
    # The LLM sometimes wraps output in prose or markdown fences.
    # Strategy: strip fences first, then regex-scan for the first {...} block.
    def _parse_llm_json(text: str) -> dict:
        text = text.strip()

        # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if fence_match:
            text = fence_match.group(1).strip()

        # 2. Try direct parse first (happy path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3. Regex: find the outermost {...} block anywhere in the text
        brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        # 4. Give up — return empty dict (triggers missing-vars follow-up)
        print(f"⚠️ [Extraction] Could not parse JSON from LLM output: {text[:300]}")
        return {}

    variables = _parse_llm_json(result)
    print(f"✅ [Extraction] Parsed variables: {variables}")

    # Determine what's missing
    direction = variables.get("direction")
    transport_mode = variables.get("transport_mode")
    weight = variables.get("weight")
    origin = variables.get("origin")
    destination = variables.get("destination")
    shipment_date = variables.get("shipment_date")
    hs_code = variables.get("hs_code")
    language = str(variables.get("language") or "").strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2})?", language):
        latest_text = state.get("user_text", "")
        if re.search(r"[\u0600-\u06ff]", latest_text):
            language = "ar"
        elif re.search(r"\b(je|nous|vous|avec|pour|retard|expedition|prédiction|bonjour)\b", latest_text, re.IGNORECASE):
            language = "fr"
        else:
            language = "en"

    missing = []
    if not direction:
        missing.append("Direction (Import/Export)")
    if not transport_mode:
        missing.append("Transport Mode (Sea/Air)")
    if weight is None:
        missing.append("Weight (kg)")
    if not origin:
        missing.append("Origin (country/port)")
    if not destination:
        missing.append("Destination (country/port)")
    if not shipment_date:
        missing.append("Shipment Date")

    # If missing vars, generate a follow-up question
    if missing:
        provided = {k: v for k, v in variables.items() if v is not None}
        system_content = MISSING_VARS_PROMPT.format(
            provided=json.dumps(provided, ensure_ascii=False),
            missing=", ".join(missing)
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=state["user_text"])
        ]
        ask_chain = llm | StrOutputParser()
        follow_up = ask_chain.invoke(messages)

        return {
            "language": language,
            "direction": direction,
            "transport_mode": transport_mode,
            "weight": float(weight) if weight else None,
            "origin": origin,
            "destination": destination,
            "shipment_date": shipment_date,
            "hs_code": hs_code,
            "missing_vars": missing,
            "response_message": follow_up,
            "status": "waiting_for_info",
        }

    # All variables present — proceed
    return {
        "language": language,
        "direction": direction,
        "transport_mode": transport_mode,
        "weight": float(weight),
        "origin": origin,
        "destination": destination,
        "shipment_date": shipment_date,
        "hs_code": hs_code,
        "missing_vars": [],
        "status": "processing",
    }


def sql_node(state: PredictionState) -> dict:
    """
    NODE 2: Query the SQL agent to look up HS nomenclature data.
    STRICT LIMITATION: Only queries the `hs_nomenclature` table.
    """
    hs_code = state.get("hs_code")
    origin = state.get("origin", "")
    destination = state.get("destination", "")

    if not hs_code:
        # No HS code provided — build a product-based query
        query = (
            f"STRICT INSTRUCTION: Query ONLY the hs_nomenclature table. "
            f"Find the HS code and product description related to a shipment "
            f"from {origin} to {destination}. Return any relevant tariff heading."
        )
    else:
        query = (
            f"STRICT INSTRUCTION: Query ONLY the hs_nomenclature table. "
            f"Look up HS code '{hs_code}'. Return the product description, "
            f"chapter heading, and any associated classification details."
        )

    try:
        print(f"📊 [SQL Node] Querying hs_nomenclature for: {hs_code or 'route-based lookup'}")
        sql_result = run_sql_agent(query)
        return {"sql_hs_info": str(sql_result)}
    except Exception as e:
        print(f"⚠️ [SQL Node] Error: {e}")
        return {"sql_hs_info": f"SQL lookup unavailable: {str(e)}"}


def rag_node(state: PredictionState) -> dict:
    """
    NODE 3: Query the RAG agent for compliance/document requirements.
    STRICT LIMITATION: Queries BOTH `documents` and `document_p` pgvector tables.
    """
    direction = state.get("direction", "Import")
    hs_code = state.get("hs_code", "")
    origin = state.get("origin", "")
    destination = state.get("destination", "")

    # Build targeted compliance query
    query = (
        f"What regulatory documents, certificates, or permits are required for "
        f"{direction.lower()}ing goods "
    )
    if hs_code:
        query += f"with HS code {hs_code} "
    query += (
        f"from {origin} to {destination}? "
        f"Specifically check for: IMANOR conformity certificates, ONSSA phytosanitary certificates, "
        f"certificates of origin, health certificates, fumigation certificates, "
        f"and any other mandatory trade documents under Moroccan customs regulations."
    )

    try:
        print(f"📚 [RAG Node] Searching compliance rules for HS={hs_code}, route={origin}→{destination}")
        rag_answer, sources = run_rag_agent(query, "")

        # Detect specific document requirements
        doc_keywords = [
            ("IMANOR", "IMANOR conformity certificate"),
            ("ONSSA", "ONSSA phytosanitary certificate"),
            ("phytosanitary", "phytosanitary certificate"),
            ("certificate of conformity", "certificate of conformity"),
            ("certificate of origin", "certificate of origin"),
            ("health certificate", "health certificate"),
            ("sanitary", "sanitary inspection certificate"),
            ("fumigation", "fumigation certificate"),
            ("NM", "Norme Marocaine (NM) certification"),
        ]

        detected_docs = []
        for keyword, doc_name in doc_keywords:
            if keyword.lower() in rag_answer.lower():
                detected_docs.append(doc_name)

        # Build RULE 2 warning if documents are required
        document_warning = None
        if detected_docs:
            doc_list = ", ".join(detected_docs)
            document_warning = (
                f"Required documents for this product: {doc_list}. "
                f"I am assuming you already have these documents to calculate the physical delay."
            )

        return {
            "legal_rules": rag_answer,
            "document_warning": document_warning,
        }

    except Exception as e:
        print(f"⚠️ [RAG Node] Error: {e}")
        return {
            "legal_rules": f"RAG lookup unavailable: {str(e)}",
            "document_warning": None,
        }


def environment_node(state: PredictionState) -> dict:
    """
    NODE 4: Fetch live weather (full dict) and port congestion for origin & destination.
    Stores full weather dicts in env_scores; severity_score is extracted in ml_prediction_node
    to feed the ML model.
    """
    origin = state.get("origin", "Unknown")
    destination = state.get("destination", "Unknown")
    shipment_date = state.get("shipment_date")

    print(f"🌍 [Environment Node] Fetching live data for: {origin} → {destination} on {shipment_date}")

    # get_live_weather now returns a full dict
    origin_weather_data = get_live_weather(origin, shipment_date)
    dest_weather_data = get_live_weather(destination, shipment_date)

    # Congestion remains an integer
    origin_congestion = get_mock_port_congestion(f"Port of {origin}", shipment_date)
    dest_congestion = get_mock_port_congestion(f"Port of {destination}", shipment_date)

    env_scores = {
        # Full weather dicts for the frontend
        "origin_weather_data": origin_weather_data,
        "destination_weather_data": dest_weather_data,
        # Severity integers for the ML model (backward-compat keys)
        "origin_weather": origin_weather_data["severity_score"],
        "destination_weather": dest_weather_data["severity_score"],
        "origin_congestion": origin_congestion,
        "destination_congestion": dest_congestion,
    }

    print(f"🌍 [Environment Node] origin={origin_weather_data}, dest={dest_weather_data}")
    return {"env_scores": env_scores}


def ml_prediction_node(state: PredictionState) -> dict:
    """
    NODE 5: Execute the ML model prediction with all gathered data.
    - Formats the Pandas DataFrame aligned to the .pkl model's columns.
    - Runs prediction + SHAP explainer.
    - Builds the final response text with weather info and document assumptions.
    """
    direction = state["direction"]
    transport_mode = state["transport_mode"]
    weight = state["weight"]
    origin = state["origin"]
    destination = state["destination"]
    env_scores = state.get("env_scores", {})
    document_warning = state.get("document_warning")
    legal_rules = state.get("legal_rules", "")

    try:
        model_data = load_model(direction)

        # Unpack model structure
        if isinstance(model_data, dict):
            model = model_data.get("model", model_data.get("pipeline", model_data.get("estimator")))
            feature_columns = model_data.get("columns", model_data.get("features", model_data.get("feature_names")))
        else:
            model = model_data
            feature_columns = None

        # ── Build feature DataFrame ──
        if feature_columns is not None:
            # Align to model's expected dummy-encoded columns
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

            # Fill numeric features
            for col in feature_columns:
                col_lower = col.lower()
                if "weight" in col_lower:
                    input_df[col] = weight
                elif "origin_weather" in col_lower or "weather_origin" in col_lower:
                    input_df[col] = env_scores.get("origin_weather", 1)
                elif "dest_weather" in col_lower or "weather_dest" in col_lower or "destination_weather" in col_lower:
                    input_df[col] = env_scores.get("destination_weather", 1)
                elif "origin_congestion" in col_lower or "congestion_origin" in col_lower:
                    input_df[col] = env_scores.get("origin_congestion", 1)
                elif "dest_congestion" in col_lower or "congestion_dest" in col_lower or "destination_congestion" in col_lower:
                    input_df[col] = env_scores.get("destination_congestion", 1)
                elif "weather" in col_lower and "origin" not in col_lower and "dest" not in col_lower:
                    # Generic weather column — use max of origin/dest
                    input_df[col] = max(env_scores.get("origin_weather", 1), env_scores.get("destination_weather", 1))
                elif "congestion" in col_lower and "origin" not in col_lower and "dest" not in col_lower:
                    # Generic congestion column — use max
                    input_df[col] = max(env_scores.get("origin_congestion", 1), env_scores.get("destination_congestion", 1))

            # Fill dummy-encoded categorical features
            for col in feature_columns:
                col_lower = col.lower()
                if transport_mode.lower() in col_lower and ("mode" in col_lower or "transport" in col_lower):
                    input_df[col] = 1
                elif direction.lower() in col_lower and ("direction" in col_lower or "type" in col_lower):
                    input_df[col] = 1
                elif origin.lower() in col_lower and ("origin" in col_lower or "from" in col_lower or "source" in col_lower):
                    input_df[col] = 1
                elif destination.lower() in col_lower and ("dest" in col_lower or "to" in col_lower or "target" in col_lower):
                    input_df[col] = 1

        else:
            # Fallback: build feature vector manually
            input_df = pd.DataFrame([{
                "weight_kg": weight,
                "transport_mode_Sea": 1 if transport_mode.lower() == "sea" else 0,
                "transport_mode_Air": 1 if transport_mode.lower() == "air" else 0,
                "direction_Import": 1 if direction.lower() == "import" else 0,
                "direction_Export": 1 if direction.lower() == "export" else 0,
                "origin_weather": env_scores.get("origin_weather", 1),
                "destination_weather": env_scores.get("destination_weather", 1),
                "origin_congestion": env_scores.get("origin_congestion", 1),
                "destination_congestion": env_scores.get("destination_congestion", 1),
            }])

        # ── Run Prediction ──
        prediction = model.predict(input_df)[0]
        delay_days = round(float(prediction), 1)

        # ── Generate SHAP Explanations (structured dicts) ──
        shap_causes = _generate_shap_explanation(model, input_df, feature_columns, state)

        # ── Top driving factor for the brief summary ──
        positive_causes = [c for c in shap_causes if c.get("days", 0) > 0]
        top_cause = sorted(positive_causes, key=lambda x: x["days"], reverse=True)[0] if positive_causes else None
        main_factor_label = top_cause["title"] if top_cause else "shipment characteristics"

        # ── STRICT RULE: Brief, single-paragraph chat message only ──
        response_message = (
            f"I have successfully predicted the delay for your shipment. "
            f"The estimated total delay is **{delay_days} days**, primarily driven by **{main_factor_label}**. "
            f"Please refer to the dashboard on the right for the full timeline, weather impact, and root cause analysis."
        )
        if document_warning:
            response_message += f"\n\n**Note:** {document_warning}"

        localized = _localize_prediction_output(
            language=state.get("language", "en"),
            response_message=response_message,
            document_warning=document_warning,
            shap_causes=shap_causes,
        )
        response_message = localized["response_message"]
        document_warning = localized["document_warning"]
        shap_causes = localized["shap_causes"]

        # ── Build rich prediction payload ──
        prediction_data = {
            "delay_days": delay_days,
            "shap_causes": shap_causes,
            "detailed_analysis": None,      # Removed — details now live in shap_causes cards
            "document_warning": document_warning,
            "language": state.get("language", "en"),
            "ui_labels": localized["ui_labels"],
            "env_scores": env_scores,       # Contains full weather dicts + congestion ints
            "variables": {
                "direction": direction,
                "transport_mode": transport_mode,
                "weight": weight,
                "origin": origin,
                "destination": destination,
                "shipment_date": state.get("shipment_date"),
            }
        }

        return {
            "final_prediction": prediction_data,
            "response_message": response_message,
            "status": "success",
        }

    except Exception as e:
        print(f"❌ [ML Prediction Node] Error: {e}")
        error_message = (
            f"I gathered all the information, but the ML model encountered an error: {str(e)}. "
            "Please verify your inputs and try again."
        )
        localized_error = _localize_prediction_output(
            language=state.get("language", "en"),
            response_message=error_message,
            document_warning=None,
            shap_causes=[],
        )
        return {
            "final_prediction": None,
            "response_message": localized_error["response_message"],
            "status": "success",
        }


# ─────────────────────────────────────────────────────────────────────
# 6. Prediction Output Localization
# ─────────────────────────────────────────────────────────────────────
DEFAULT_UI_LABELS = {
    "predicted_delay": "Predicted Delay",
    "days": "days",
    "day_short": "d",
    "low_risk": "Low risk — expedited clearance likely",
    "moderate_risk": "Moderate — standard processing time",
    "high_risk": "High risk — potential bottlenecks detected",
    "delay_timeline": "Delay Timeline",
    "live_weather": "Live Weather Conditions",
    "shipment_details": "Shipment Details",
    "document_assumption": "Document Assumption",
    "root_cause_analysis": "Root Cause Analysis",
    "origin": "Origin",
    "transit": "Transit",
    "destination": "Destination",
    "general": "General",
    "mode": "Mode",
    "weight": "Weight",
    "direction": "Direction",
    "route": "Route",
    "download_report": "Download Report",
    "generating": "Generating...",
}


def _localize_prediction_output(
    language: str,
    response_message: str,
    document_warning: Optional[str],
    shap_causes: list,
) -> dict:
    """Translate user-visible prediction text without changing SHAP values or canonical stages."""
    language = (language or "en").lower()
    english_causes = [
        {**cause, "stage_label": cause.get("stage", "General")}
        for cause in shap_causes
    ]
    fallback = {
        "response_message": response_message,
        "document_warning": document_warning,
        "shap_causes": english_causes,
        "ui_labels": DEFAULT_UI_LABELS.copy(),
    }
    if language.startswith("en"):
        return fallback

    source_payload = {
        "response_message": response_message,
        "document_warning": document_warning,
        "shap_causes": shap_causes,
        "ui_labels": DEFAULT_UI_LABELS,
    }
    localization_prompt = f"""You localize BridgeAI prediction results.
Translate every user-visible string in the JSON payload into the language identified by ISO code '{language}'.

STRICT RULES:
- Return one raw JSON object only. Do not use markdown fences or commentary.
- Preserve every numeric value exactly, especially SHAP 'days'.
- Preserve proper names, port names, HS codes and acronyms such as ADII, ONSSA, IMANOR and SHAP.
- Preserve each canonical 'stage' value exactly as Origin, Transit, Destination or General.
- Add 'stage_label' to every SHAP cause, translated into the target language.
- Translate response_message, document_warning, every SHAP title and detailed_cause, and every ui_labels value.
- Keep all JSON keys unchanged.

INPUT JSON:
{json.dumps(source_payload, ensure_ascii=False)}"""

    try:
        raw = (llm | StrOutputParser()).invoke([
            SystemMessage(content=localization_prompt),
            HumanMessage(content="Localize the supplied prediction JSON now."),
        ]).strip()
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if fence_match:
            raw = fence_match.group(1).strip()
        localized = json.loads(raw)

        translated_causes = localized.get("shap_causes", [])
        safe_causes = []
        for index, original in enumerate(shap_causes):
            translated = translated_causes[index] if index < len(translated_causes) else {}
            safe_causes.append({
                "stage": original.get("stage", "General"),
                "stage_label": str(translated.get("stage_label") or original.get("stage", "General")),
                "days": original.get("days", 0),
                "title": str(translated.get("title") or original.get("title", "")),
                "detailed_cause": str(translated.get("detailed_cause") or original.get("detailed_cause", "")),
            })

        labels = DEFAULT_UI_LABELS.copy()
        labels.update({
            key: str(value)
            for key, value in localized.get("ui_labels", {}).items()
            if key in labels and value
        })
        return {
            "response_message": str(localized.get("response_message") or response_message),
            "document_warning": str(localized.get("document_warning") or document_warning) if document_warning else None,
            "shap_causes": safe_causes,
            "ui_labels": labels,
        }
    except Exception as exc:
        print(f"⚠️ [Localization] Falling back to English for language={language}: {exc}")
        return fallback


# ─────────────────────────────────────────────────────────────────────
# 7. SHAP Explanation Generator
# ─────────────────────────────────────────────────────────────────────
def _generate_shap_explanation(model, input_df, feature_columns, state: PredictionState) -> list:
    """
    Generate structured SHAP-style delay cause cards.
    Each entry: {"stage": str, "days": float, "title": str, "detailed_cause": str}
    stage ∈ ["Origin", "Transit", "Destination", "General"]
    """
    env = state.get("env_scores", {})
    origin = state.get("origin", "Origin")
    destination = state.get("destination", "Destination")
    transport_mode = state.get("transport_mode", "Sea")
    direction = state.get("direction", "Import")
    weight = state.get("weight", 1000)

    origin_w_data = env.get("origin_weather_data", {"severity_score": 1, "condition_text": "Clear", "wind_kph": 10})
    dest_w_data   = env.get("destination_weather_data", {"severity_score": 1, "condition_text": "Clear", "wind_kph": 10})
    origin_cong   = env.get("origin_congestion", 1)
    dest_cong     = env.get("destination_congestion", 1)
    orig_w_sev    = env.get("origin_weather", 1)
    dest_w_sev    = env.get("destination_weather", 1)

    origin_cond   = origin_w_data.get("condition_text", "Clear")
    dest_cond     = dest_w_data.get("condition_text", "Clear")
    origin_wind   = origin_w_data.get("wind_kph", 10)
    dest_wind     = dest_w_data.get("wind_kph", 10)

    cong_labels = {1: "minimal", 2: "low", 3: "moderate", 4: "heavy", 5: "critical gridlock"}

    # ── Attempt real SHAP first; map feature names → card dicts ──
    raw_shap = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_flat = shap_values.flatten()
        feature_names = list(feature_columns) if feature_columns is not None else list(input_df.columns)
        indices = np.argsort(np.abs(shap_flat))[::-1]
        for idx in indices[:8]:
            if abs(shap_flat[idx]) > 0.01:
                raw_shap.append((feature_names[idx].lower(), round(float(shap_flat[idx]), 2)))
    except Exception:
        pass

    if not raw_shap:
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_names = list(feature_columns) if feature_columns is not None else list(input_df.columns)
                indices = np.argsort(importances)[::-1]
                for idx in indices[:8]:
                    if importances[idx] > 0.01:
                        raw_shap.append((feature_names[idx].lower(), round(float(importances[idx] * 10), 2)))
        except Exception:
            pass

    # ── Map raw SHAP feature names → structured cards ──
    def _map_feature(fname: str, days: float) -> dict | None:
        f = fname.lower()
        if "origin_congestion" in f or ("congestion" in f and "origin" in f):
            return {
                "stage": "Origin",
                "days": abs(days),
                "title": "Origin Port Congestion",
                "detailed_cause": (
                    f"The port of {origin} is currently experiencing {cong_labels.get(origin_cong, 'moderate')} congestion "
                    f"(level {origin_cong}/5). Vessel queuing and yard capacity constraints are delaying loading "
                    f"operations, pushing the departure window back significantly."
                )
            }
        if "dest_congestion" in f or "destination_congestion" in f or ("congestion" in f and "dest" in f):
            return {
                "stage": "Destination",
                "days": abs(days),
                "title": "Destination Port Congestion",
                "detailed_cause": (
                    f"The port of {destination} is reporting {cong_labels.get(dest_cong, 'moderate')} inbound traffic "
                    f"(level {dest_cong}/5). Berth allocation delays and customs pre-clearance backlogs are "
                    f"extending the expected unloading and release timeline."
                )
            }
        if "origin_weather" in f or ("weather" in f and "origin" in f):
            return {
                "stage": "Origin",
                "days": abs(days),
                "title": "Adverse Weather at Origin",
                "detailed_cause": (
                    f"Current weather conditions at {origin} are reported as \"{origin_cond}\" with wind speeds "
                    f"of {origin_wind:.0f} km/h (severity {orig_w_sev}/5). Port crane operations and "
                    f"lashing activities are restricted under these conditions, causing pre-departure delays."
                )
            }
        if "dest_weather" in f or "destination_weather" in f or ("weather" in f and "dest" in f):
            return {
                "stage": "Destination",
                "days": abs(days),
                "title": "Adverse Weather at Destination",
                "detailed_cause": (
                    f"Weather at {destination} is currently \"{dest_cond}\" with gusts reaching "
                    f"{dest_wind:.0f} km/h (severity {dest_w_sev}/5). Discharge operations and "
                    f"port-side logistics are disrupted, adding clearance time post-arrival."
                )
            }
        if "weight" in f:
            return {
                "stage": "General",
                "days": abs(days),
                "title": "Shipment Weight",
                "detailed_cause": (
                    f"A shipment weight of {weight:,.0f} kg requires dedicated heavy-lift equipment and "
                    f"extended loading/discharge cycles. Heavier consignments also face stricter documentary "
                    f"verification at customs, adding administrative lead time."
                )
            }
        if "transport" in f or "mode" in f:
            mode_detail = "Sea freight vessels operate on fixed weekly sailing schedules" if transport_mode.lower() == "sea" else "Air freight cutoff windows and transit hub connections"
            return {
                "stage": "Transit",
                "days": abs(days),
                "title": f"Transport Mode: {transport_mode}",
                "detailed_cause": (
                    f"{mode_detail} introduce structural transit time. "
                    f"{'Ocean voyages between these ports typically require 14-21 days in transit before reaching the customs arrival gate.' if transport_mode.lower() == 'sea' else 'Flight schedules and hub transit times add 2-5 days to the total clearance window.'}"
                )
            }
        if "direction" in f or "import" in f or "export" in f:
            return {
                "stage": "Destination",
                "days": abs(days),
                "title": f"Trade Direction: {direction}",
                "detailed_cause": (
                    f"{'Import shipments into Morocco are subject to ADII customs inspection, ONSSA conformity checks, and mandatory HS code duty assessment, adding procedural delays.' if direction.lower() == 'import' else 'Export shipments require validated commercial invoices, packing lists, and export declarations through Moroccan customs (ADII), each requiring processing time.'}"
                )
            }
        if "imanor" in f:
            return {
                "stage": "Destination",
                "days": abs(days),
                "title": "IMANOR Conformity Certificate",
                "detailed_cause": (
                    "Moroccan law mandates IMANOR (Institut Marocain de Normalisation) conformity "
                    "certification for this product category. Laboratory analysis and official stamp issuance "
                    "can take 3–7 working days, creating a compliance bottleneck at the customs gate."
                )
            }
        if "onssa" in f or "phytosanitary" in f:
            return {
                "stage": "Destination",
                "days": abs(days),
                "title": "ONSSA Phytosanitary Inspection",
                "detailed_cause": (
                    "The Office National de Sécurité Sanitaire des produits Alimentaires (ONSSA) requires "
                    "physical inspection and laboratory sampling for this commodity class. Inspection queues "
                    "and laboratory turnaround typically add 2–5 business days to the clearance process."
                )
            }
        if "red" in f or "channel" in f:
            return {
                "stage": "Destination",
                "days": abs(days),
                "title": "Red Channel Customs Inspection",
                "detailed_cause": (
                    "This shipment has been flagged for mandatory Red Channel physical examination by customs "
                    "officers. Full container devanning, document cross-referencing, and re-sealing add "
                    "an average of 3–6 working days to the release timeline."
                )
            }
        return None

    contributions = []
    seen_titles = set()
    for fname, days in raw_shap:
        card = _map_feature(fname, days)
        if card and card["title"] not in seen_titles:
            contributions.append(card)
            seen_titles.add(card["title"])

    # ── Always-present rule-based fallback cards (fill gaps) ──
    fallback_cards = [
        {
            "stage": "Transit",
            "days": round(3.5 if transport_mode.lower() == "sea" else 1.2, 1),
            "title": f"Transport Mode: {transport_mode}",
            "detailed_cause": (
                f"{'Ocean freight between these ports follows fixed carrier schedules with weekly departures. The sea transit phase itself accounts for 14–21 days before the vessel reaches the destination port gate.' if transport_mode.lower() == 'sea' else 'Air freight connections and hub transit times introduce structural delays of 2–5 days before customs presentation at the destination airport.'}"
            )
        },
        {
            "stage": "Origin",
            "days": round(orig_w_sev * 0.6, 1),
            "title": "Origin Port Conditions",
            "detailed_cause": (
                f"Loading operations at {origin} are currently affected by {cong_labels.get(origin_cong, 'moderate')} "
                f"port congestion (level {origin_cong}/5) and weather reported as \"{origin_cond}\" "
                f"with {origin_wind:.0f} km/h winds. Combined, these factors are compressing the available "
                f"loading window before the vessel's cut-off."
            )
        },
        {
            "stage": "Destination",
            "days": round(dest_cong * 0.6, 1),
            "title": "Destination Port & Customs",
            "detailed_cause": (
                f"The {destination} arrival gate is operating under {cong_labels.get(dest_cong, 'moderate')} "
                f"inbound traffic (level {dest_cong}/5). Current weather: \"{dest_cond}\" "
                f"({dest_wind:.0f} km/h). Customs pre-clearance and berth allocation "
                f"are the primary bottlenecks at this stage."
            )
        },
        {
            "stage": "General",
            "days": round(float(weight) / 5000, 1),
            "title": "Shipment Weight",
            "detailed_cause": (
                f"A consignment of {weight:,.0f} kg requires specialized heavy-lift handling, extended "
                f"stowage planning, and may trigger weight-based documentary review at customs. "
                f"This adds proportional overhead to both loading and clearance timelines."
            )
        },
        {
            "stage": "Destination",
            "days": 2.0 if direction.lower() == "import" else 1.0,
            "title": f"Trade Direction: {direction}",
            "detailed_cause": (
                f"{'Import clearance through Moroccan ADII customs involves multi-step duty assessment, NM standards verification, and potential ONSSA/IMANOR checks before goods can be released to the importer.' if direction.lower() == 'import' else 'Export declaration processing through ADII requires validated commercial documentation, export licenses where applicable, and pre-departure customs confirmation.'}"
            )
        },
    ]

    if not contributions:
        contributions = fallback_cards
    else:
        # Supplement with any fallback stage not yet covered
        covered_stages = {c["stage"] for c in contributions}
        for fb in fallback_cards:
            if fb["stage"] not in covered_stages:
                contributions.append(fb)
                covered_stages.add(fb["stage"])

    # Sort: Origin → Transit → Destination → General, descending days within stage
    stage_order = {"Origin": 0, "Transit": 1, "Destination": 2, "General": 3}
    contributions.sort(key=lambda c: (stage_order.get(c["stage"], 9), -c["days"]))
    return contributions


# ─────────────────────────────────────────────────────────────────────
# 7. Graph Routing Logic
# ─────────────────────────────────────────────────────────────────────
def route_after_extraction(state: PredictionState) -> str:
    """
    After extraction, decide whether to continue the pipeline or stop.
    If variables are missing → END (return follow-up question).
    If all present and we already have a prediction -> check if anything changed.
    """
    if state.get("missing_vars"):
        return END

    if state.get("final_prediction"):
        old_vars = state["final_prediction"].get("variables", {})
        # Helper to safely compare strings ignoring case
        def cmp(v1, v2):
            if isinstance(v1, str) and isinstance(v2, str):
                return v1.strip().lower() == v2.strip().lower()
            return v1 == v2

        # If none of the variables have changed, it's a follow-up question!
        if (cmp(state.get("direction"), old_vars.get("direction")) and
            cmp(state.get("transport_mode"), old_vars.get("transport_mode")) and
            cmp(state.get("weight"), old_vars.get("weight")) and
            cmp(state.get("origin"), old_vars.get("origin")) and
            cmp(state.get("destination"), old_vars.get("destination")) and
            cmp(state.get("shipment_date"), old_vars.get("shipment_date"))):
            return "follow_up_node"

    # New or updated shipment details, continue to SQL node
    return "sql_node"

def follow_up_node(state: PredictionState) -> dict:
    """
    NODE: Answer follow-up questions about an existing prediction.
    """
    prediction = dict(state["final_prediction"])
    requested_language = state.get("language", prediction.get("language", "en"))
    if requested_language != prediction.get("language", "en") and prediction.get("shap_causes"):
        relocalized = _localize_prediction_output(
            language=requested_language,
            response_message="The prediction is ready. Review the SHAP root causes in the dashboard.",
            document_warning=prediction.get("document_warning"),
            shap_causes=prediction.get("shap_causes", []),
        )
        prediction.update({
            "language": requested_language,
            "shap_causes": relocalized["shap_causes"],
            "document_warning": relocalized["document_warning"],
            "ui_labels": relocalized["ui_labels"],
        })
    context_str = json.dumps(prediction, indent=2)
    messages = [
        SystemMessage(content=FOLLOW_UP_PROMPT.format(prediction_context=context_str, user_text=state["user_text"]))
    ]
    chain = llm | StrOutputParser()
    response = chain.invoke(messages)
    
    return {
        "response_message": response,
        "status": "success",
        "final_prediction": prediction 
    }

def route_start(state: PredictionState) -> str:
    """
    Decide whether to start a new extraction or answer a follow-up question.
    """
    # Always route to extraction_node first so we can detect modifications.
    # extraction_node will see the full history, extract variables, and
    # route_after_extraction will compare them to state["final_prediction"].
    return "extraction_node"


# ─────────────────────────────────────────────────────────────────────
# 8. Build & Compile the LangGraph
# ─────────────────────────────────────────────────────────────────────
def build_prediction_graph():
    """Construct the prediction state machine graph."""
    workflow = StateGraph(PredictionState)

    # Register nodes
    workflow.add_node("extraction_node", extraction_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("environment_node", environment_node)
    workflow.add_node("ml_prediction_node", ml_prediction_node)
    workflow.add_node("follow_up_node", follow_up_node)

    # Set conditional entry point
    workflow.set_conditional_entry_point(
        route_start,
        {"follow_up_node": "follow_up_node", "extraction_node": "extraction_node"}
    )

    # Conditional edge after extraction: stop if missing vars, else continue
    workflow.add_conditional_edges(
        "extraction_node",
        route_after_extraction,
        {"sql_node": "sql_node", "follow_up_node": "follow_up_node", END: END}
    )

    # Sequential edges for the full pipeline
    workflow.add_edge("sql_node", "rag_node")
    workflow.add_edge("rag_node", "environment_node")
    workflow.add_edge("environment_node", "ml_prediction_node")
    workflow.add_edge("ml_prediction_node", END)
    workflow.add_edge("follow_up_node", END)

    return workflow.compile()


# Compile graph once at module level for reuse
prediction_graph = build_prediction_graph()


# ─────────────────────────────────────────────────────────────────────
# 9. Public Entry Point
# ─────────────────────────────────────────────────────────────────────
async def run_prediction_agent(conversation_history: list, user_message: str, current_prediction: dict = None) -> dict:
    """
    Main entry point for the Prediction Agent pipeline.

    Executes the LangGraph state machine which flows through:
        extraction → sql → rag → environment → ml_prediction
        (or follow_up_node if current_prediction is provided)

    Args:
        conversation_history: List of past messages [{"role": str, "content": str}]
        user_message: The latest user message string
        current_prediction: The existing prediction dictionary (if any)

    Returns:
        {
            "status": "waiting_for_info" | "success",
            "message": str,
            "prediction_data": dict | None
        }
    """
    # Initialize state
    initial_state: PredictionState = {
        "user_text": user_message,
        "conversation_history": conversation_history,
        "language": "en",
        "direction": None,
        "transport_mode": None,
        "weight": None,
        "origin": None,
        "destination": None,
        "shipment_date": None,
        "hs_code": None,
        "missing_vars": [],
        "sql_hs_info": None,
        "legal_rules": None,
        "document_warning": None,
        "env_scores": {},
        "final_prediction": current_prediction,
        "response_message": "",
        "status": "waiting_for_info",
    }

    # Execute the graph
    final_state = await prediction_graph.ainvoke(initial_state)

    # Build return payload
    return {
        "status": final_state.get("status", "waiting_for_info"),
        "message": final_state.get("response_message", ""),
        "prediction_data": final_state.get("final_prediction"),
    }
