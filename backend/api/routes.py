"""FastAPI routes for MedTrace."""
import asyncio
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from api.models import (
    AskRequest, AskResponse, TraceItem, EvolutionRecord,
    MetricsResponse, GoldenExample
)
from agent.main_agent import run_medtrace_agent
from agent.prompts import get_active_prompt
from evolution.phoenix_mcp_client import get_phoenix_client
from evolution.evolution_engine import (
    run_evolution_cycle, get_evolution_history,
    is_evolution_running, get_current_avg_score
)

router = APIRouter()
_total_queries = 0

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "MedTrace"}

@router.post("/ask", response_model=AskResponse)
async def ask_medical_question(req: AskRequest):
    global _total_queries
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    _total_queries += 1
    logger.info(f"Query #{_total_queries}: {req.query[:80]}")
    state = await run_medtrace_agent(req.query, req.session_id)
    phoenix = get_phoenix_client()
    phoenix.record_trace(state)
    return AskResponse(
        query=state.get("query", ""),
        answer=state.get("answer", ""),
        eval_scores=state.get("eval_scores", {}),
        avg_score=state.get("avg_score", 0.0),
        prompt_version=state.get("prompt_version", "v1"),
        trace_id=state.get("trace_id", ""),
        citations=state.get("citations", []),
        evolution_triggered=state.get("evolution_triggered", False),
        processing_time_ms=state.get("processing_time_ms", 0.0),
        error=state.get("error"),
    )

@router.get("/traces", response_model=List[TraceItem])
async def get_traces(limit: int = 10):
    phoenix = get_phoenix_client()
    traces = phoenix.get_recent_traces(limit=limit)
    return [
        TraceItem(
            trace_id=t.get("trace_id", ""),
            timestamp=t.get("timestamp", ""),
            query=t.get("query", "")[:120],
            avg_score=t.get("avg_score", 0.0),
            prompt_version=t.get("prompt_version", "v1"),
            evolution_triggered=t.get("evolution_triggered", False),
        )
        for t in traces
    ]

@router.get("/evolution/history", response_model=List[EvolutionRecord])
async def evolution_history():
    return [EvolutionRecord(**r) for r in get_evolution_history()]

@router.post("/evolution/trigger")
async def trigger_evolution(background_tasks: BackgroundTasks):
    if is_evolution_running():
        return {"status": "already_running", "message": "Evolution cycle in progress"}
    background_tasks.add_task(run_evolution_cycle)
    return {"status": "triggered", "message": "Evolution cycle started"}

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    phoenix = get_phoenix_client()
    golden_stats = phoenix.get_golden_dataset_stats()
    version, _ = get_active_prompt()
    return MetricsResponse(
        total_queries=_total_queries,
        avg_score=round(get_current_avg_score(), 2),
        evolutions_run=len(get_evolution_history()),
        golden_examples=golden_stats["total"],
        current_prompt_version=version,
        evolution_running=is_evolution_running(),
    )

@router.get("/golden-dataset", response_model=List[GoldenExample])
async def golden_dataset(limit: int = 50):
    phoenix = get_phoenix_client()
    examples = phoenix.get_golden_dataset(limit=limit)
    return [
        GoldenExample(
            id=e.get("id", ""),
            query=e.get("query", ""),
            answer=e.get("answer", "")[:300],
            avg_score=e.get("avg_score", 0.0),
            prompt_version=e.get("prompt_version", "v1"),
            added_at=e.get("added_at", ""),
        )
        for e in examples
    ]
