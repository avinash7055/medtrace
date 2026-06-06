# MedTrace: Self-Evolving Medical AI Agent 🧬⚡

**MedTrace** is the world's first medical AI agent that uses **Arize Phoenix** (self-hosted) as an *active self-improvement tool*, rather than just a passive monitoring dashboard. Built for the **Google Cloud Rapid Agent Hackathon (Arize track)** and fully migrated to run **locally via Ollama**, it autonomously identifies its own weaknesses, experiments with prompt mutations, and promotes winning instructions—with **zero human intervention**.

## 🌟 What Makes MedTrace Unique?

Most agents use tracing purely for observability. **MedTrace closes the loop:**

1. **Active Trace Querying:** The LangGraph agent queries its own low-scoring production traces from Phoenix.
2. **Autonomous Root Cause Diagnosis:** Local Ollama (`gemma2:2b`) analyzes failure patterns and identifies *why* answers were poor.
3. **Self-Improving Prompts:** 3 prompt mutations are generated, evaluated in an A/B experiment, and the winner is automatically promoted—no human required.
4. **Golden Dataset Curation:** High-scoring traces (≥8.0) are autonomously saved as a Golden Dataset for future reference.
5. **Clean Observability:** Evaluation calls use `suppress_tracing()` to keep Phoenix traces noise-free — only real agent spans are recorded.

## 🏗️ Architecture

```
User Query
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ LangGraph Agent (main_agent.py)                        │
│                                                        │
│ 1. query_understanding: Extract intent/entities        │
│ 2. rag_retrieval: ChromaDB vector search               │
│ 3. llm_reasoning: Generate response (active prompt)    │
│ 4. answer_generation: Format with citations            │
│ 5. self_evaluation: LLM-as-Judge (5 rubrics, 0-10)    │
│ 6. evolution_trigger: IF avg_score < 6.5 ───┐         │
└────────────────────────────────────────────┼───────────┘
                                             │
    ┌────────────────────────────────────────▼──────────────────────┐
    │ Evolution Engine (evolution_engine.py)                        │
    │ Step 1: Fetch failure traces from Phoenix                     │
    │ Step 2: Diagnose root cause with Ollama                       │
    │ Step 3: Generate 3 prompt mutations                           │
    │ Step 4: Create A/B experiment, evaluate on failure cases      │
    │ Step 5: Promote winning prompt → active prompt                │
    │ Step 6: Curate Golden Dataset from high-scoring traces        │
    └────────────────────────────────────────▲──────────────────────┘
                                             │
┌────────────────────────────────────────────┴──────────────────────┐
│ Arize Phoenix (Self-Hosted via Docker)                            │
│  • Traces  • OTEL gRPC (port 4317)  • UI (port 6006)             │
└───────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | [LangGraph](https://python.langchain.com/docs/langgraph) |
| **Local LLM** | Ollama `gemma2:2b` |
| **Local Embeddings** | Ollama `nomic-embed-text` |
| **Observability** | [Arize Phoenix](https://arize.com/docs/phoenix) — self-hosted via Docker |
| **Tracing SDK** | `phoenix.otel` + `openinference-instrumentation-langchain` |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Frontend** | React + Vite + Recharts + Lucide Icons |
| **Deployment** | Docker Compose (Phoenix + Backend + Frontend) |

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- **Ollama** running locally with `gemma2:2b` and `nomic-embed-text` models pulled:
  ```bash
  ollama pull gemma2:2b
  ollama pull nomic-embed-text
  ```

> **No cloud credentials or API keys needed** — both Ollama and Phoenix run completely locally.

### 1. Environment Configuration

```bash
cd medtrace
```

Edit `backend/.env` to configure Ollama and Phoenix endpoints:

```env
# Ollama (Local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Phoenix (self-hosted — no API key required)
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
PHOENIX_PROJECT_NAME=medtrace-agent
```

### 2. Run with Docker Compose (Recommended)

```bash
docker compose up --build
```

Once running:

| Service | URL |
|---|---|
| 🖥️ Frontend Dashboard | http://localhost:5173 |
| ⚙️ Backend API | http://localhost:8000 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |
| 🔭 Phoenix UI | http://localhost:6006 |

### 3. Manual Setup (Local Development)

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

# Populate the knowledge base
python sample_data.py

# Start Phoenix separately (Docker)
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

# Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### 4. Test the Agent

```bash
cd backend
python test_agent.py
```

## 🔭 Phoenix Integration Details

MedTrace uses Phoenix as an **active tool**, not just a passive dashboard:

| Phoenix Feature | How MedTrace Uses It |
|---|---|
| **OTEL Tracing** | Every LangGraph node emits a span with eval scores attached |
| **LangChain Auto-Instrumentation** | `LangChainInstrumentor` captures all LLM calls automatically |
| **`suppress_tracing()`** | Evaluation LLM calls are excluded from traces (no noise) |
| **Failure Traces** | Evolution Engine queries low-scoring spans to trigger improvement |
| **A/B Experiments** | 3 prompt mutations are evaluated on the same failure cases |
| **Prompt Hub** | Winning prompt version is promoted and immediately deployed |
| **Golden Dataset** | Traces scoring ≥ 8.0 are auto-curated for future use |

Phoenix UI → `http://localhost:6006` — view all traces, scores, and span attributes live.

## 🎨 Dashboard Tour

The React frontend features a dark glassmorphism design:

1. **Chat Interface (Left):** Ask medical questions. Responses include score badges (accuracy, safety, clarity), citations, processing time, and the active prompt version.
2. **Metrics Panel (Top Right):** Live counts of queries, average score, evolution cycles, and golden examples curated.
3. **Score Evolution Chart (Middle Right):** Graph showing how scores improve after each autonomous prompt mutation cycle.
4. **Phoenix Trace Viewer (Bottom Left):** Live feed of the latest traces with their evaluation scores.
5. **Golden Dataset (Bottom Right):** Curated table of top-scoring (≥8.0) answers selected by the agent itself.

## 📜 License
MIT License
