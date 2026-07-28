# BridgeAi: Intelligent Multi-Agent Customs & Logistics Optimization Platform

[![React](https://img.shields.io/badge/Frontend-React_19-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](#)
[![LangGraph](https://img.shields.io/badge/AI_Orchestration-LangGraph-orange.svg)](#)
[![Supabase](https://img.shields.io/badge/Database-Supabase-green.svg)](#)

## Overview
**BridgeAi** is an intelligent multi-agent platform designed to automate and streamline customs clearance procedures and logistics operations between Morocco and the European Union. 

The manual preparation and verification of customs files create critical bottlenecks at borders, which are incompatible with modern Just-in-Time (JIT) logistics. As European regulatory frameworks (such as CBAM, REACH, and TARIC nomenclature) grow increasingly complex, economic operators struggle to maintain compliance. BridgeAi addresses these compounding challenges by transforming regulatory compliance from a bureaucratic burden into a competitive advantage through transparent, explainable artificial intelligence.

## 🌟 Core Features & Pillars
The solution is structured around four foundational pillars:

*   **Intelligent Regulatory Assistance:** Utilizes a Retrieval-Augmented Generation (RAG) Agent (via LangChain and pgvector) alongside a SQL Agent to provide natural-language assistance for customs regulations.
*   **Predictive Customs Processing (Arena Model):** Employs a hybrid Champion-Challenger architecture combining Random Forest and XGBoost to predict clearance delays[cite: 3]. It features SHAP (SHapley Additive exPlanations) to provide transparent, variable-by-variable explainability for every prediction[cite: 3].
*   **Automated Document Analysis:** Integrates LlamaParse to extract structured data from complex, multi-column customs PDFs and trade documentation[cite: 3].
*   **Multi-Criteria Route Optimization:** Features a logistics decision support module built on a MultiDiGraph structure, allowing for route optimization based on time, cost, $CO_{2}$ emissions, and carrier reliability[cite: 3].

## 🏗️ System Architecture
BridgeAi adopts a modular, multi-layered architecture[cite: 3]:

1.  **Presentation Layer:** Built with React 19, Vite, and TypeScript, providing interfaces for role-based navigation (Standard User vs. Administrator) and live streaming of AI responses (SSE)[cite: 3].
2.  **API & Controllers:** Powered by FastAPI to expose secure endpoints for chat, audio transcription (via Whisper), document analysis, and route optimization[cite: 3].
3.  **AI Orchestration Layer:** LangGraph orchestrates the workflow as a persistent state graph, dynamically routing user requests to the appropriate AI agents (RAG, SQL, or Vision)[cite: 3].
4.  **Data & Persistence Layer:** Utilizes PostgreSQL/Supabase for relational data and Row-Level Security (RLS)[cite: 3]. PGVector manages document embeddings (using bge-m3 and bge-large-en-v1.5), while Supabase Storage handles PDF files[cite: 3].

##  Installation & Setup
##  Backend Setup (FastAPI):
  cd backend
  python -m venv venv

  source venv/bin/activate  # On Windows: venv\Scripts\activate
  
  pip install -r requirements.txt
  
  uvicorn main:app --reload
  
## Frontend Setup (React/Vite): 
 cd frontend
 
 npm install
 
 npm run dev

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zaidzen3333-beep/BridgeAI_.git
   cd BridgeAi_
