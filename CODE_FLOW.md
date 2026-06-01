# MedTrace — End-to-End Code Flow 🧬⚡

> **MedTrace** is a Self-Evolving Medical AI Agent that autonomously identifies its weaknesses, generates prompt mutations, runs A/B experiments, and promotes winning prompts — with zero human intervention.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Application Startup Flow](#4-application-startup-flow)
5. [User Query Pipeline (6-Node LangGraph)](#5-user-query-pipeline-6-node-langgraph)
6. [Self-Evaluation (LLM-as-Judge)](#6-self-evaluation-llm-as-judge)
7. [Evolution Engine — Self-Improvement Loop](#7-evolution-engine--self-improvement-loop)
8. [Phoenix MCP Client](#8-phoenix-mcp-client)
9. [Frontend Dashboard](#9-frontend-dashboard)
10. [API Endpoints](#10-api-endpoints)
11. [Docker & Deployment](#11-docker--deployment)
12. [Data Flow Summary](#12-data-flow-summary)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                         │
│  ChatInterface │ MetricsPanel │ EvolutionGraph │ TraceViewer │ Golden  │
│                         localhost:5173                                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ HTTP (Vite proxy → /api)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI + LangGraph)                       │
│                         localhost:8000                                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              LangGraph Agent Pipeline (6 Nodes)                  │   │
│  │                                                                  │   │
│  │  query_understanding → rag_retrieval → gemini_reasoning          │   │
│  │      → answer_generation → self_evaluation → evolution_trigger   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │   ChromaDB        │  │  Evolution Engine │  │  Phoenix MCP Client │  │
│  │   (Vector Store)  │  │  (Self-Improve)   │  │  (Traces + Expts)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ gRPC OTLP (port 4317)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   ARIZE PHOENIX (Docker Container)                      │
│              Traces • Experiments • Prompt Hub • UI                      │
│                     localhost:6006 (UI)                                  │
│                     localhost:4317 (gRPC)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer              | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| **LLM**            | Gemini 2.0 Flash (`google-generativeai` SDK)            |
| **Embeddings**     | Gemini Embedding 001 (`models/gemini-embedding-001`)    |
| **Agent Framework**| LangGraph (stateful node-based pipeline)                |
| **Vector Store**   | ChromaDB (local persistent)                             |
| **Backend API**    | FastAPI + Uvicorn                                       |
| **Observability**  | Arize Phoenix (self-hosted Docker) + OpenTelemetry      |
| **Tracing SDK**    | `phoenix.otel` + `openinference-instrumentation`        |
| **Frontend**       | React + Vite + Recharts + Lucide Icons                  |
| **Deployment**     | Docker Compose (Phoenix container)                      |

---

## 3. Project Structure

```
medtrace/
├── docker-compose.yml          # Phoenix Docker container
├── Dockerfile                  # Backend Docker image
│
├── backend/
│   ├── main.py                 # FastAPI entry point + lifespan setup
│   ├── config.py               # Settings, Phoenix tracing, Gemini clients
│   ├── requirements.txt        # Python dependencies
│   ├── sample_data.py          # Populate knowledge base with 50+ Q&A
│   ├── fetch_real_data.py      # Fetch real medical data
│   │
│   ├── agent/                  # LangGraph Agent
│   │   ├── main_agent.py       # Graph builder + run_medtrace_agent()
│   │   ├── state.py            # MedTraceState TypedDict definition
│   │   ├── nodes.py            # 6 node implementations
│   │   ├── prompts.py          # Versioned system prompts (v1, v2, v3...)
│   │   └── tools.py            # ChromaDB retrieval tools
│   │
│   ├── api/                    # REST API
│   │   ├── routes.py           # FastAPI route handlers
│   │   └── models.py           # Pydantic request/response models
│   │
│   ├── evolution/              # Self-Improvement Engine
│   │   ├── evolution_engine.py # Core 8-step evolution cycle
│   │   ├── evaluator.py        # LLM-as-Judge (5 rubrics, 0-10)
│   │   └── phoenix_mcp_client.py # Phoenix API + in-memory fallback
│   │
│   └── knowledge_base/         # Medical Knowledge Base
│       ├── loader.py           # JSON → Documents → ChromaDB indexer
│       ├── vectorstore.py      # ChromaDB singleton
│       └── data/               # Medical Q&A JSON files
│           ├── medical_qa.json
│           └── medical_qa_extra.json
│
└── frontend/
    ├── vite.config.js          # Vite + proxy (/api → localhost:8000)
    ├── package.json
    └── src/
        ├── main.jsx            # React entry point
        ├── App.jsx             # Dashboard layout + data polling
        ├── index.css           # Dark glassmorphism styles
        └── components/
            ├── ChatInterface.jsx   # Chat UI + API calls
            ├── MetricsPanel.jsx    # Live stats cards
            ├── EvolutionGraph.jsx  # Score evolution chart
            ├── TraceViewer.jsx     # Recent traces feed
            └── GoldenDataset.jsx   # Curated best Q&A table
```

---

## 4. Application Startup Flow

**File:** `backend/main.py`

```
Server Start (uvicorn main:app)
    │
    ▼
lifespan() context manager executes
    │
    ├── 1. setup_phoenix_tracing()     ← config.py
    │       │
    │       ├── register() from phoenix.otel
    │       │     → Connects to Phoenix gRPC (localhost:4317)
    │       │     → Project name: "medtrace-agent"
    │       │     → auto_instrument=True (captures LangChain spans)
    │       │
    │       └── GoogleGenAIInstrumentor().instrument()
    │             → Instruments direct Gemini SDK calls
    │
    ├── 2. initialize_knowledge_base()  ← knowledge_base/loader.py
    │       │
    │       ├── load_json_qa_files()
    │       │     → Reads all *.json from knowledge_base/data/
    │       │     → Returns list of {question, answer, topic, source}
    │       │
    │       ├── qa_pairs_to_documents()
    │       │     → Converts to LangChain Document objects
    │       │     → Format: "Question: ...\n\nAnswer: ..."
    │       │
    │       ├── Check existing ChromaDB docs (resume support)
    │       │     → Skip already-indexed documents
    │       │
    │       └── vs.add_documents(batch)
    │             → Embeds via GeminiBatchEmbeddings (config.py)
    │             → Batch size: 90, with 60s sleep for rate limits
    │
    ├── 3. CORS Middleware added
    │       → Allows localhost:5173 (frontend)
    │
    └── 4. Router mounted at /api prefix
            → All routes from api/routes.py
```

### Key Singletons Initialized:
- **Settings** (`config.py`) — loaded from `.env` file via Pydantic
- **Gemini Client** — `get_gemini_client()` singleton
- **LangChain LLM** — `get_langchain_llm()` (ChatGoogleGenerativeAI, temp=0.2)
- **Embedding Model** — `GeminiBatchEmbeddings` class (batches of 100)
- **ChromaDB** — `get_vectorstore()` singleton (persistent at `./chroma_db`)
- **Phoenix Client** — `get_phoenix_client()` singleton

---

## 5. User Query Pipeline (6-Node LangGraph)

**Files:** `agent/main_agent.py`, `agent/nodes.py`, `agent/state.py`

### State Object (Passed Through All Nodes)

```python
class MedTraceState(TypedDict):
    # Input
    query: str
    session_id: str

    # Node 1 output
    query_intent: str            # e.g. "drug_interaction"
    query_entities: List[str]    # e.g. ["aspirin", "warfarin"]

    # Node 2 output
    retrieved_docs: List[Dict]   # [{content, source, topic, score}]
    retrieval_count: int

    # Node 3-4 output
    reasoning: str
    answer: str
    citations: List[str]

    # Node 5 output
    eval_scores: EvalScores      # 5 rubrics, 0-10 each
    avg_score: float
    eval_feedback: str

    # Node 6 output
    evolution_triggered: bool
    evolution_reason: str

    # Metadata
    trace_id: str
    prompt_version: str
    processing_time_ms: float
    error: Optional[str]
```

### Pipeline Flow

```
User Query: "What are the side effects of metformin?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 1: QUERY UNDERSTANDING                                │
│  File: nodes.py → query_understanding()                     │
│                                                             │
│  • Sends query to Gemini with parsing prompt                │
│  • Gemini returns JSON:                                     │
│    {                                                        │
│      "intent": "dosage_info",                               │
│      "entities": ["metformin"],                             │
│      "complexity": "moderate"                               │
│    }                                                        │
│  • Sets OTEL span attributes for Phoenix tracing            │
│                                                             │
│  Output → query_intent, query_entities                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 2: RAG RETRIEVAL                                      │
│  File: nodes.py → rag_retrieval()                           │
│  Tool: tools.py → retrieve_medical_context()                │
│                                                             │
│  • Enriches query with extracted entities:                   │
│    "What are side effects of metformin? metformin"           │
│  • Calls ChromaDB similarity_search_with_relevance_scores() │
│  • Returns top-5 documents with scores                      │
│  • Each doc: {content, source, topic, confidence, score}    │
│                                                             │
│  Output → retrieved_docs, retrieval_count                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 3: GEMINI REASONING                                   │
│  File: nodes.py → gemini_reasoning()                        │
│  Prompts: prompts.py → get_active_prompt()                  │
│                                                             │
│  • Fetches ACTIVE prompt version (v1 initially,             │
│    evolves to v2, v3... after evolution cycles)              │
│  • Formats retrieved docs into context string:              │
│    "[Document 1] Source: X | Topic: Y\n<content>\n---\n..." │
│  • Sends {context + query} to Gemini                        │
│  • Gemini generates structured medical response             │
│                                                             │
│  Output → reasoning, prompt_version                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 4: ANSWER GENERATION                                  │
│  File: nodes.py → answer_generation()                       │
│                                                             │
│  • Takes raw reasoning as the answer                        │
│  • Extracts citations from retrieved docs (score > 0.4)     │
│  • Deduplicates citation sources                            │
│                                                             │
│  Output → answer, citations                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 5: SELF-EVALUATION (LLM-as-Judge)                     │
│  File: nodes.py → self_evaluation()                         │
│  Evaluator: evolution/evaluator.py → evaluate_answer()      │
│                                                             │
│  • Uses suppress_tracing() to keep Phoenix clean            │
│  • Gemini scores its OWN answer on 5 rubrics (0-10):        │
│    ┌──────────────────┬───────────────────────────────────┐  │
│    │ medical_accuracy │ Is info medically correct?        │  │
│    │ completeness     │ All aspects addressed?            │  │
│    │ safety           │ Proper warnings included?         │  │
│    │ clarity          │ Clear and understandable?         │  │
│    │ citation_quality │ Sources properly referenced?      │  │
│    └──────────────────┴───────────────────────────────────┘  │
│  • Calculates avg_score                                     │
│  • Attaches scores as OTEL span attributes                  │
│                                                             │
│  Output → eval_scores, avg_score, eval_feedback             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 6: EVOLUTION TRIGGER                                  │
│  File: nodes.py → evolution_trigger()                       │
│                                                             │
│  • IF avg_score < 6.5 (threshold from config):              │
│    ├── Check global lock (_evolution_running)                │
│    ├── If lock free → acquire lock                          │
│    ├── asyncio.create_task(run_evolution_cycle())            │
│    │     → Runs in BACKGROUND (doesn't block response!)     │
│    └── Return evolution_triggered = True                    │
│                                                             │
│  • IF avg_score >= 6.5:                                     │
│    └── Return evolution_triggered = False (skip)            │
│                                                             │
│  Output → evolution_triggered, evolution_reason             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
                          END
```

### After Pipeline Completes

Back in `api/routes.py`:

```python
# 1. Record trace in Phoenix client (in-memory store)
phoenix = get_phoenix_client()
phoenix.record_trace(state)

# 2. Return AskResponse to frontend
return AskResponse(
    query, answer, eval_scores, avg_score,
    prompt_version, trace_id, citations,
    evolution_triggered, processing_time_ms, error
)
```

---

## 6. Self-Evaluation (LLM-as-Judge)

**File:** `evolution/evaluator.py`

```
evaluate_answer(query, answer)
    │
    ├── Build evaluation prompt (EVALUATION_PROMPT from prompts.py)
    │     → Contains detailed 0-10 rubric for each dimension
    │     → Includes scoring guidelines (e.g., 0-3 = dangerous errors)
    │
    ├── with suppress_tracing():    ← Keeps Phoenix traces noise-free
    │     response = await llm.ainvoke(eval_prompt)
    │
    ├── _parse_scores(response)
    │     → Try JSON parse first
    │     → Fallback: regex extraction of "key": value patterns
    │     → Clamp all scores to [0.0, 10.0]
    │
    └── Return (scores_dict, feedback_string)
```

**Why suppress_tracing()?** Without it, every evaluation LLM call would create its own trace in Phoenix, polluting the real agent traces. This keeps Phoenix clean — only actual user-query traces appear.

---

## 7. Evolution Engine — Self-Improvement Loop

**File:** `evolution/evolution_engine.py`

This is the **heart of MedTrace** — the autonomous self-improvement system.

### When Does It Trigger?

```
Agent pipeline completes
    → avg_score < 6.5?
        → YES → evolution_trigger node fires
            → Is another evolution already running? (global lock)
                → NO → Start run_evolution_cycle() in background
                → YES → Skip (prevent stacking)
```

### 8-Step Evolution Cycle

```
run_evolution_cycle()
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: FETCH FAILURE TRACES                               │
│                                                             │
│  phoenix.get_failure_traces(score_threshold=6.5, limit=20)  │
│  → Tries Phoenix REST API first                             │
│  → Falls back to in-memory trace store                      │
│  → Need at least 2 failures to proceed                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: DIAGNOSE ROOT CAUSE                                │
│                                                             │
│  diagnose_failures(failure_traces)                          │
│  → Samples up to 10 failure traces                          │
│  → Sends to Gemini: "Analyze these failures, identify the   │
│    PRIMARY root cause"                                      │
│  → Root cause categories:                                   │
│    • Missing safety warnings                                │
│    • Incomplete drug interaction coverage                    │
│    • Poor citation quality                                  │
│    • Insufficient clinical detail                           │
│    • Unclear structure/formatting                           │
│    • Lacking dosage adjustment guidance                     │
│    • Missing emergency escalation cues                     │
│  → Returns: "Answers consistently miss drug interaction     │
│    severity grading and contraindication warnings"          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: GENERATE 3 PROMPT MUTATIONS                        │
│                                                             │
│  generate_prompt_mutations(current_prompt, root_cause, 3)   │
│  → Sends current prompt + root cause to Gemini              │
│  → "Generate 3 improved variants, each taking a DIFFERENT   │
│    approach to solving the problem"                         │
│  → Returns 3 PromptMutation objects:                        │
│    {name, prompt, rationale, evaluated_score}               │
│  → Has @retry(3 attempts) with exponential backoff          │
│  → Fallback: append root cause as enhancement to current    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: CREATE PHOENIX A/B EXPERIMENT                      │
│                                                             │
│  phoenix.create_experiment(mutations, test_cases)           │
│  → Creates experiment in in-memory store                    │
│  → Also registers with Phoenix REST API if available        │
│  → Returns experiment_id                                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: EVALUATE MUTATIONS ON FAILURE CASES                │
│                                                             │
│  evaluate_mutations_on_failures(mutations, failure_traces)  │
│  → For each mutation (3 total):                             │
│    → For each failure case (up to 5):                       │
│      → Retrieve fresh context from ChromaDB                 │
│      → Generate answer using mutation's prompt              │
│      → Evaluate answer with LLM-as-Judge                    │
│      → Collect score                                        │
│    → mutation.evaluated_score = average across cases         │
│  → Total LLM calls: ~30 (3 mutations × 5 cases × 2 calls) │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: SELECT WINNER                                      │
│                                                             │
│  winner = max(mutations, key=evaluated_score)               │
│  → Only promote if winner.score > old_avg_score             │
│  → If no improvement → return status="no_improvement"      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 7: PROMOTE TO PHOENIX PROMPT HUB                      │
│                                                             │
│  new_version = "v4" (auto-incremented)                      │
│  update_active_prompt(new_version, winner.prompt)            │
│  → Updates CURRENT_PROMPT_VERSION in prompts.py (in-memory) │
│  → Updates SYSTEM_PROMPTS dict with new prompt              │
│  phoenix.update_prompt_hub("medtrace_system_prompt", ...)   │
│  → Stores in in-memory prompt hub                           │
│  → Registers with Phoenix API if available                  │
│                                                             │
│  ★ From now on, ALL new queries use the improved prompt! ★  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 8: CURATE GOLDEN DATASET                              │
│                                                             │
│  → Fetch last 100 traces                                    │
│  → Filter: avg_score >= 8.0                                 │
│  → phoenix.add_to_golden_dataset("golden_medical_qa", ...)  │
│  → These become reference examples for future evaluation    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
                  EvolutionResult saved to history
                  Lock released (_is_running = False)
```

### Prompt Version Lifecycle

```
v1 (default)  →  Score drops below 6.5
    │
    ▼
Evolution cycle runs → v2 generated and promoted
    │
    ▼
v2 (active)   →  Score drops again
    │
    ▼
Another evolution → v3 generated and promoted
    │
    ▼
v3 (active)   →  ... and so on (autonomous)
```

### Built-in Prompt Templates

**v1** — Standard medical assistant (Summary → Details → Safety → Sources → Recommendation)

**v2** — Precision clinical protocol (ASSESSMENT → EVIDENCE → CLINICAL DETAILS → SAFETY ALERTS → LIMITATIONS)

**v3** — SOAP-inspired decision support (Situation → Objective → Analysis → Plan) with evidence grading (Level A/B/C)

After evolution, **v4+** are dynamically generated by Gemini targeting specific weaknesses.

---

## 8. Phoenix MCP Client

**File:** `evolution/phoenix_mcp_client.py`

This is the bridge between MedTrace and Arize Phoenix.

### Dual-Mode Design

```
Every method follows this pattern:

    async def some_operation():
        if await self._check_api():        # Phoenix Docker reachable?
            try:
                return api_call()           # Use Phoenix REST API
            except:
                pass
        return in_memory_fallback()         # Always works for demo
```

### Key Operations

| Method | Purpose |
|--------|---------|
| `record_trace()` | Store completed agent trace (called after every query) |
| `get_failure_traces()` | Fetch traces with avg_score < threshold |
| `get_recent_traces()` | Last N traces for dashboard |
| `create_experiment()` | Register A/B experiment with mutations |
| `set_experiment_results()` | Store evaluation results |
| `update_prompt_hub()` | Promote winning prompt |
| `add_to_golden_dataset()` | Curate high-scoring examples (≥8.0) |
| `get_golden_dataset()` | Return curated examples for dashboard |

### In-Memory Storage (Module-Level)

```python
_in_memory_traces: List[Dict] = []        # Last 500 traces
_in_memory_experiments: Dict = {}          # Active experiments
_golden_dataset: List[Dict] = []           # Curated best Q&A
_prompt_hub: Dict = {}                     # Promoted prompts
```

---

## 9. Frontend Dashboard

**File:** `frontend/src/App.jsx`

### Data Polling

```javascript
// Every 3 seconds, fetch all dashboard data:
const [mRes, tRes, eRes, gRes] = await Promise.all([
    fetch('/api/metrics'),           // Live stats
    fetch('/api/traces'),            // Recent traces
    fetch('/api/evolution/history'), // Past evolution cycles
    fetch('/api/golden-dataset')    // Curated examples
])
```

### Component Layout

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER: MedTrace logo + Phoenix/ChromaDB status indicators  │
├────────────────────────────┬─────────────────────────────────┤
│                            │                                 │
│   ChatInterface            │   MetricsPanel                  │
│   ┌──────────────────┐     │   ┌───────────┬───────────┐    │
│   │ Agent messages    │     │   │ Queries:12│ Avg: 7.8  │    │
│   │ + score badges    │     │   │ Evos: 2   │ Golden:15 │    │
│   │ + citations       │     │   └───────────┴───────────┘    │
│   │ + prompt version  │     │   Active Prompt: v3            │
│   │ + processing time │     │   [Trigger Evolution] button   │
│   └──────────────────┘     │                                 │
│   [input box] [send]       │   EvolutionGraph                │
│                            │   ┌───────────────────────┐     │
│                            │   │  Score trend chart     │     │
│                            │   │  (Recharts LineChart)  │     │
│                            │   └───────────────────────┘     │
├────────────────────────────┼─────────────────────────────────┤
│   TraceViewer              │   GoldenDataset                 │
│   Recent traces feed       │   Curated best Q&A table        │
│   with scores & versions   │   (score ≥ 8.0)                │
└────────────────────────────┴─────────────────────────────────┘
```

### Chat Flow (Frontend → Backend → Frontend)

```
User types question → Enter
    │
    ├── Optimistic UI update (show user message)
    ├── Show "Thinking + evaluating…" loader
    │
    ├── POST /api/ask { query: "..." }
    │       │
    │       ▼
    │   Backend runs full 6-node pipeline (3-15 seconds)
    │       │
    │       ▼
    │   Returns: { answer, eval_scores, avg_score,
    │              citations, prompt_version,
    │              evolution_triggered, processing_time_ms }
    │
    ├── Display agent message with:
    │   • Answer text (pre-wrapped)
    │   • Score badges (green ≥7.5, yellow ≥6, red <6)
    │   • Average score (purple badge)
    │   • Citations (BookOpen icon)
    │   • Prompt version label
    │   • "⚡ Evolution triggered" if applicable
    │   • Processing time in ms
    │
    └── Call onNewResult() → triggers dashboard data refresh
```

---

## 10. API Endpoints

**File:** `api/routes.py`

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Health check + app info |
| `GET` | `/api/health` | Service health status |
| `POST` | `/api/ask` | Submit medical question → full pipeline |
| `GET` | `/api/traces?limit=10` | Recent traces with scores |
| `GET` | `/api/metrics` | Dashboard stats (queries, score, evolutions, golden count) |
| `GET` | `/api/evolution/history` | Past evolution cycle results |
| `POST` | `/api/evolution/trigger` | Manually trigger evolution cycle |
| `GET` | `/api/golden-dataset?limit=50` | Curated high-quality Q&A pairs |

### Request/Response Models (`api/models.py`)

```python
# Request
AskRequest { query: str, session_id: Optional[str] }

# Response
AskResponse {
    query, answer, eval_scores: Dict[str, float],
    avg_score, prompt_version, trace_id,
    citations: List[str], evolution_triggered: bool,
    processing_time_ms, error: Optional[str]
}

MetricsResponse {
    total_queries, avg_score, evolutions_run,
    golden_examples, current_prompt_version,
    evolution_running: bool
}
```

---

## 11. Docker & Deployment

### docker-compose.yml

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"   # Phoenix UI Dashboard
      - "4317:4317"   # OTLP gRPC (traces sent here)
    volumes:
      - phoenix_data:/phoenix_data
    healthcheck:
      test: wget --spider http://localhost:6006/healthz
```

### Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:5173 | React Dashboard |
| Backend | http://localhost:8000 | FastAPI API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Phoenix UI | http://localhost:6006 | Trace Explorer |

### Startup Order

```
1. docker compose up          → Phoenix container starts
2. cd backend && uvicorn ...  → Backend connects to Phoenix
3. cd frontend && npm run dev → Frontend proxies to backend
```

---

## 12. Data Flow Summary

### Complete Request Lifecycle

```
[User Browser]
    │ Type question, press Enter
    ▼
[React Frontend] ──POST /api/ask──▶ [FastAPI Backend]
                                        │
                                        ▼
                                   [LangGraph Pipeline]
                                        │
                    ┌───────────────────┤
                    │                   │
                    ▼                   ▼
              [Gemini LLM]        [ChromaDB]
              (understand,         (retrieve
               reason,              medical
               evaluate)            documents)
                    │                   │
                    └───────┬───────────┘
                            │
                            ▼
                    Generate Answer
                    + Self-Evaluate
                    + Record Trace
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        [Phoenix]                   [Frontend]
        (store trace,               (show answer,
         OTEL spans)                 scores, etc.)
              │
              │ If avg_score < 6.5
              ▼
     [Evolution Engine]
     (diagnose → mutate → test → promote)
              │
              ▼
     New prompt version active
     (next query uses improved prompt)
```

### What Makes MedTrace Unique?

1. **Phoenix as ACTIVE Tool** — Not passive monitoring. The agent queries Phoenix for its own failures.
2. **Autonomous Prompt Evolution** — No human needed. Agent diagnoses, experiments, and promotes.
3. **Clean Observability** — `suppress_tracing()` keeps evaluation calls out of Phoenix.
4. **Golden Dataset Curation** — Agent builds its own reference dataset from production successes.
5. **Concurrent Safety** — Global lock prevents multiple evolution cycles from overlapping.

---

*Generated for MedTrace — Google Cloud Rapid Agent Hackathon (Arize Track)*
