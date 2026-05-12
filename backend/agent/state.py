"""
LangGraph State Definition for MedTrace Agent.
Typed dict carried through every node in the graph.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class EvalScores(TypedDict):
    medical_accuracy: float
    completeness: float
    safety: float
    clarity: float
    citation_quality: float


class MedTraceState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    query: str
    session_id: str

    # ── Query Understanding ──────────────────────────────────────────────────
    query_intent: str          # e.g. "drug_interaction", "diagnosis", "dosage"
    query_entities: List[str]  # extracted medical entities (drugs, conditions)

    # ── RAG Retrieval ────────────────────────────────────────────────────────
    retrieved_docs: List[Dict[str, Any]]   # [{content, source, score}]
    retrieval_count: int

    # ── Reasoning & Answer ───────────────────────────────────────────────────
    reasoning: str
    answer: str
    citations: List[str]

    # ── Self-Evaluation ──────────────────────────────────────────────────────
    eval_scores: EvalScores
    avg_score: float
    eval_feedback: str

    # ── Tracing / Versioning ─────────────────────────────────────────────────
    trace_id: str
    prompt_version: str
    span_context: Dict[str, Any]

    # ── Evolution ────────────────────────────────────────────────────────────
    evolution_triggered: bool
    evolution_reason: str

    # ── Errors ───────────────────────────────────────────────────────────────
    error: Optional[str]
    processing_time_ms: float
