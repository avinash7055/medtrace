"""
MedTrace LangGraph Agent Graph.

Builds and compiles the stateful LangGraph agent:
  query_understanding → rag_retrieval → gemini_reasoning
      → answer_generation → self_evaluation → evolution_trigger → END

Every edge is unconditional — all nodes always execute in sequence.
The evolution_trigger node asynchronously fires the evolution engine
without blocking the response path.
"""

import time
import uuid
from typing import Any, Dict

from langgraph.graph import StateGraph, END
from loguru import logger

from agent.state import MedTraceState
from agent.nodes import (
    query_understanding,
    rag_retrieval,
    gemini_reasoning,
    answer_generation,
    self_evaluation,
    evolution_trigger,
)


def build_medtrace_graph():
    """
    Construct the LangGraph StateGraph for MedTrace.
    Returns a compiled, invokable graph.
    """
    graph = StateGraph(MedTraceState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("query_understanding", query_understanding)
    graph.add_node("rag_retrieval", rag_retrieval)
    graph.add_node("gemini_reasoning", gemini_reasoning)
    graph.add_node("answer_generation", answer_generation)
    graph.add_node("self_evaluation", self_evaluation)
    graph.add_node("evolution_trigger", evolution_trigger)

    # ── Define edges (linear pipeline) ───────────────────────────────────────
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "rag_retrieval")
    graph.add_edge("rag_retrieval", "gemini_reasoning")
    graph.add_edge("gemini_reasoning", "answer_generation")
    graph.add_edge("answer_generation", "self_evaluation")
    graph.add_edge("self_evaluation", "evolution_trigger")
    graph.add_edge("evolution_trigger", END)

    compiled = graph.compile()
    logger.info("✅ MedTrace LangGraph compiled successfully")
    return compiled


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_medtrace_graph()
    return _graph


async def run_medtrace_agent(query: str, session_id: str = None) -> Dict[str, Any]:
    """
    Entry point: run the full MedTrace agent pipeline for a medical query.

    Args:
        query: User's medical question
        session_id: Optional session identifier for grouping traces

    Returns:
        Final state dict with answer, scores, trace info
    """
    start_time = time.time()
    session_id = session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    initial_state: MedTraceState = {
        "query": query,
        "session_id": session_id,
        "query_intent": "",
        "query_entities": [],
        "retrieved_docs": [],
        "retrieval_count": 0,
        "reasoning": "",
        "answer": "",
        "citations": [],
        "eval_scores": {
            "medical_accuracy": 0.0,
            "completeness": 0.0,
            "safety": 0.0,
            "clarity": 0.0,
            "citation_quality": 0.0,
        },
        "avg_score": 0.0,
        "eval_feedback": "",
        "trace_id": trace_id,
        "prompt_version": "v1",
        "span_context": {},
        "evolution_triggered": False,
        "evolution_reason": "",
        "error": None,
        "processing_time_ms": 0.0,
    }

    logger.info(f"🚀 MedTrace agent started | session={session_id} | query='{query[:80]}…'")

    try:
        graph = get_graph()
        final_state = await graph.ainvoke(initial_state)
        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        final_state["processing_time_ms"] = elapsed_ms

        logger.info(
            f"✅ Agent complete | avg_score={final_state.get('avg_score')} | "
            f"time={elapsed_ms}ms | evolution={final_state.get('evolution_triggered')}"
        )
        return final_state

    except Exception as exc:
        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        logger.error(f"❌ Agent pipeline failed: {exc}")
        return {
            **initial_state,
            "answer": (
                "I encountered an error processing your medical question. "
                "Please try again or consult a medical professional directly."
            ),
            "error": str(exc),
            "processing_time_ms": elapsed_ms,
        }
