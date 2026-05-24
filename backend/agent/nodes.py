"""
LangGraph Node Implementations for MedTrace.

Each node is a pure async function:  state → partial_state_update
All nodes add custom OTEL span attributes for Phoenix tracing.

Node execution order:
  query_understanding → rag_retrieval → gemini_reasoning
      → answer_generation → self_evaluation → evolution_trigger
"""

import json
import re
import time
import uuid
from typing import Any, Dict
import asyncio

from loguru import logger
from opentelemetry import trace

from agent.state import MedTraceState
from agent.prompts import get_active_prompt, EVALUATION_PROMPT
from agent.tools import retrieve_medical_context, format_context_for_prompt
from config import get_langchain_llm, get_tracer, settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_span_attrs(span, attrs: Dict[str, Any]) -> None:
    """Safe helper — sets OTEL span attributes, skipping None values."""
    for k, v in attrs.items():
        if v is not None:
            try:
                span.set_attribute(k, str(v) if not isinstance(v, (bool, int, float, str)) else v)
            except Exception:
                pass


def _extract_json(text: str) -> Dict:
    """Extract the first JSON object from an LLM response string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Query Understanding
# ─────────────────────────────────────────────────────────────────────────────

async def query_understanding(state: MedTraceState) -> Dict:
    """
    Parse the user's medical question.
    Extracts: intent category + named medical entities (drugs, conditions).
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("medtrace.query_understanding") as span:
        llm = get_langchain_llm()
        query = state["query"]

        prompt = f"""Analyze this medical question and return JSON only.

Question: {query}

Return JSON:
{{
  "intent": "<one of: drug_interaction, symptom_diagnosis, treatment_protocol, dosage_info, emergency, general_medical>",
  "entities": ["<medical entity 1>", "<medical entity 2>"],
  "complexity": "<simple|moderate|complex>"
}}"""

        try:
            response = await llm.ainvoke(prompt)
            parsed = _extract_json(response.content)
            intent = parsed.get("intent", "general_medical")
            entities = parsed.get("entities", [])
        except Exception as exc:
            logger.warning(f"Query understanding LLM call failed: {exc}")
            intent = "general_medical"
            entities = []

        _set_span_attrs(span, {
            "query_intent": intent,
            "query_entities": str(entities),
            "medtrace.node": "query_understanding",
        })

        logger.info(f"Query parsed → intent={intent}, entities={entities}")
        return {
            "query_intent": intent,
            "query_entities": entities,
            "trace_id": state.get("trace_id") or str(uuid.uuid4()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: RAG Retrieval
# ─────────────────────────────────────────────────────────────────────────────

async def rag_retrieval(state: MedTraceState) -> Dict:
    """
    Search ChromaDB with the user query + extracted entities.
    Returns top-5 medical document chunks with similarity scores.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("medtrace.rag_retrieval") as span:
        # Enrich query with entities for better recall
        entities = state.get("query_entities", [])
        enriched_query = state["query"]
        if entities:
            enriched_query += " " + " ".join(entities[:3])

        docs = retrieve_medical_context(enriched_query, top_k=5)

        _set_span_attrs(span, {
            "retrieval_count": len(docs),
            "query_intent": state.get("query_intent", ""),
            "medtrace.node": "rag_retrieval",
        })

        logger.info(f"Retrieved {len(docs)} documents from ChromaDB")
        return {
            "retrieved_docs": docs,
            "retrieval_count": len(docs),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Gemini Reasoning
# ─────────────────────────────────────────────────────────────────────────────

async def gemini_reasoning(state: MedTraceState) -> Dict:
    """
    Send retrieved context + query to Gemini using the active prompt version.
    Records which prompt version produced this reasoning.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("medtrace.gemini_reasoning") as span:
        llm = get_langchain_llm()
        prompt_version, prompt_template = get_active_prompt()

        context = format_context_for_prompt(state.get("retrieved_docs", []))
        full_prompt = prompt_template.format(
            context=context,
            query=state["query"],
        )

        try:
            response = await llm.ainvoke(full_prompt)
            reasoning = response.content
        except Exception as exc:
            logger.error(f"Gemini reasoning failed: {exc}")
            reasoning = (
                "I was unable to process this query due to a technical error. "
                "Please consult a qualified medical professional."
            )

        _set_span_attrs(span, {
            "prompt_version": prompt_version,
            "context_length": len(context),
            "retrieval_count": state.get("retrieval_count", 0),
            "medtrace.node": "gemini_reasoning",
        })

        logger.info(f"Gemini reasoning complete (prompt={prompt_version}, {len(reasoning)} chars)")
        return {
            "reasoning": reasoning,
            "prompt_version": prompt_version,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

async def answer_generation(state: MedTraceState) -> Dict:
    """
    Format the final answer and extract source citations.
    Wraps the raw reasoning into a clean, user-facing response.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("medtrace.answer_generation") as span:
        reasoning = state.get("reasoning", "")
        retrieved_docs = state.get("retrieved_docs", [])

        # Extract citations from retrieved docs
        citations = list({
            doc.get("source", "MedTraceKB")
            for doc in retrieved_docs
            if doc.get("score", 0) > 0.4
        })

        # The reasoning IS the answer (already formatted by Gemini)
        answer = reasoning

        _set_span_attrs(span, {
            "citation_count": len(citations),
            "answer_length": len(answer),
            "medtrace.node": "answer_generation",
        })

        logger.info(f"Answer generated ({len(answer)} chars, {len(citations)} citations)")
        return {
            "answer": answer,
            "citations": citations,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Self-Evaluation (LLM-as-Judge)
# ─────────────────────────────────────────────────────────────────────────────

async def self_evaluation(state: MedTraceState) -> Dict:
    """
    Score the answer on 5 medical quality rubrics using Gemini as judge.
    This is what Phoenix traces — scores become features for evolution.

    Rubrics:
      1. medical_accuracy   2. completeness   3. safety
      4. clarity            5. citation_quality
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("medtrace.self_evaluation") as span:
        from evolution.evaluator import evaluate_answer

        try:
            scores, feedback = await evaluate_answer(
                query=state["query"],
                answer=state.get("answer", ""),
            )
            avg_score = round(sum(scores.values()) / len(scores), 2)
        except Exception as exc:
            logger.error(f"Self-evaluation failed: {exc}")
            scores = {
                "medical_accuracy": 5.0,
                "completeness": 5.0,
                "safety": 5.0,
                "clarity": 5.0,
                "citation_quality": 5.0,
            }
            avg_score = 5.0
            feedback = f"Evaluation error: {exc}"

        _set_span_attrs(span, {
            "eval.medical_accuracy": scores.get("medical_accuracy", 0),
            "eval.completeness": scores.get("completeness", 0),
            "eval.safety": scores.get("safety", 0),
            "eval.clarity": scores.get("clarity", 0),
            "eval.citation_quality": scores.get("citation_quality", 0),
            "eval.avg_score": avg_score,
            "prompt_version": state.get("prompt_version", "v1"),
            "medtrace.node": "self_evaluation",
        })

        logger.info(
            f"Self-eval complete → avg={avg_score} | "
            f"acc={scores.get('medical_accuracy')} "
            f"safe={scores.get('safety')} "
            f"clarity={scores.get('clarity')}"
        )

        return {
            "eval_scores": scores,
            "avg_score": avg_score,
            "eval_feedback": feedback,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: Evolution Trigger
# ─────────────────────────────────────────────────────────────────────────────

# ── Evolution Lock — prevents multiple parallel evolution cycles ──────────────
_evolution_running: bool = False
_evolution_lock: asyncio.Lock = asyncio.Lock()


async def evolution_trigger(state: MedTraceState) -> Dict:
    """
    If avg_score < threshold, fire off an async evolution cycle.

    This is what makes MedTrace unique:
    - Phoenix MCP is called as an ACTIVE TOOL (not passive monitor)
    - The agent queries its own failure traces from Phoenix
    - Generates prompt mutations, runs A/B experiments in Phoenix
    - Promotes winning prompt back to Phoenix Prompt Hub

    A global lock ensures only ONE evolution cycle runs at a time.
    Concurrent low-score queries skip evolution instead of stacking up.
    """
    global _evolution_running

    avg_score = state.get("avg_score", 10.0)
    threshold = settings.evolution_score_threshold

    if avg_score < threshold:
        # Check lock — skip if evolution already running
        if _evolution_running:
            logger.info(
                f"⏭️  Evolution skipped: already running (avg_score={avg_score})"
            )
            return {
                "evolution_triggered": False,
                "evolution_reason": "Evolution already in progress — skipped to prevent overlap.",
            }

        logger.warning(
            f"⚡ Evolution triggered: avg_score={avg_score} < threshold={threshold}"
        )

        from evolution.evolution_engine import run_evolution_cycle

        async def _run_and_unlock():
            """Run evolution cycle and release lock when done."""
            global _evolution_running
            try:
                await run_evolution_cycle()
            except Exception as exc:
                logger.error(f"Evolution cycle failed: {exc}")
            finally:
                _evolution_running = False
                logger.info("🔓 Evolution lock released")

        # Acquire lock and fire evolution in background
        _evolution_running = True
        logger.info("🔒 Evolution lock acquired")
        asyncio.create_task(_run_and_unlock())

        return {
            "evolution_triggered": True,
            "evolution_reason": (
                f"Average score {avg_score} below threshold {threshold}. "
                "Evolution cycle initiated."
            ),
        }
    else:
        logger.debug(f"No evolution needed: avg_score={avg_score} ≥ {threshold}")
        return {
            "evolution_triggered": False,
            "evolution_reason": "",
        }
