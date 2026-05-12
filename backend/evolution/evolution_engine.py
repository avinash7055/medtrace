"""
MedTrace Evolution Engine — The Heart of Self-Improvement.

This is the world's first agent that uses Phoenix MCP as an ACTIVE tool:
  1. Query own failure traces from Phoenix
  2. Diagnose root causes with Gemini
  3. Generate 3 prompt mutations
  4. Create Phoenix A/B experiment
  5. Evaluate all mutations on failure cases
  6. Promote winner to Phoenix Prompt Hub
  7. Curate Golden Dataset from successes

No human intervention required — fully autonomous improvement loop.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.prompts import (
    get_active_prompt,
    update_active_prompt,
    list_prompt_versions,
    SYSTEM_PROMPTS,
)
from config import get_langchain_llm, settings
from evolution.evaluator import batch_evaluate
from evolution.phoenix_mcp_client import get_phoenix_client

# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PromptMutation:
    name: str
    prompt: str
    rationale: str
    evaluated_score: float = 0.0


@dataclass
class EvolutionResult:
    cycle_id: str
    timestamp: str
    old_version: str
    new_version: str
    old_score: float
    new_score: float
    improvement: float
    root_cause: str
    mutations_tested: int
    winner_name: str
    golden_examples_added: int
    duration_seconds: float
    status: str = "success"
    error: Optional[str] = None


# Global evolution history (persisted in-memory, shown on dashboard)
_evolution_history: List[EvolutionResult] = []
_is_running: bool = False
_current_avg_score: float = 7.0  # updated after each agent run


# ─────────────────────────────────────────────────────────────────────────────
# Core Evolution Functions
# ─────────────────────────────────────────────────────────────────────────────


async def diagnose_failures(failure_traces: List[Dict]) -> str:
    """
    Use Gemini to identify the root cause of low-scoring answers.
    Analyzes patterns across multiple failure traces.
    """
    if not failure_traces:
        return "Insufficient failure traces for diagnosis"

    llm = get_langchain_llm()

    # Sample up to 10 failures for analysis
    sample = failure_traces[:10]
    trace_summaries = []
    for t in sample:
        scores = t.get("eval_scores", {})
        trace_summaries.append(
            f"Q: {t.get('query', '')[:100]}\n"
            f"Scores: {json.dumps(scores)}\n"
            f"Avg: {t.get('avg_score', 0):.1f}"
        )

    analysis_prompt = f"""You are a medical AI quality analyst. Analyze these {len(sample)} 
low-scoring medical Q&A responses and identify the PRIMARY root cause of failure.

Failed Traces:
{chr(10).join(trace_summaries)}

Identify the single most impactful root cause from:
- Missing safety warnings
- Incomplete drug interaction coverage
- Poor citation quality
- Insufficient clinical detail
- Unclear structure/formatting
- Lacking dosage adjustment guidance
- Missing emergency escalation cues

Return a concise 1-2 sentence root cause diagnosis."""

    try:
        response = await llm.ainvoke(analysis_prompt)
        diagnosis = response.content.strip()
        logger.info(f"Root cause diagnosed: {diagnosis[:100]}")
        return diagnosis
    except Exception as exc:
        logger.error(f"Failure diagnosis failed: {exc}")
        return "Unable to diagnose root cause — using generic improvement strategy"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
async def generate_prompt_mutations(
    current_prompt: str,
    root_cause: str,
    count: int = 3,
) -> List[PromptMutation]:
    """
    Generate N prompt mutations targeting the diagnosed root cause.
    Each mutation is a distinct approach to improving the prompt.
    """
    llm = get_langchain_llm()

    mutation_prompt = f"""You are a prompt engineering expert specializing in medical AI systems.

CURRENT PROMPT (first 500 chars):
{current_prompt[:500]}

ROOT CAUSE OF FAILURES:
{root_cause}

Generate exactly {count} improved prompt variants that address this root cause.
Each variant should take a DIFFERENT approach to solving the problem.

Return ONLY valid JSON:
{{
  "mutations": [
    {{
      "name": "variant_1_<brief_label>",
      "rationale": "<1 sentence why this helps>",
      "key_change": "<the main change made>",
      "full_prompt": "<complete improved system prompt with {{context}} and {{query}} placeholders>"
    }},
    ...
  ]
}}"""

    response = await llm.ainvoke(mutation_prompt)
    raw = response.content

    try:
        # Extract JSON
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            mutations = []
            for m in data.get("mutations", [])[:count]:
                mutations.append(PromptMutation(
                    name=m.get("name", f"variant_{len(mutations)+1}"),
                    prompt=m.get("full_prompt", current_prompt),
                    rationale=m.get("rationale", ""),
                ))
            logger.info(f"Generated {len(mutations)} prompt mutations")
            return mutations
    except Exception as exc:
        logger.error(f"Mutation parsing failed: {exc}")

    # Fallback: create simple variations of current prompt
    return [
        PromptMutation(
            name=f"variant_{i+1}_safety_enhanced",
            prompt=current_prompt + f"\n\n[ENHANCEMENT {i+1}]: {root_cause}. Always address this explicitly.",
            rationale=f"Variation {i+1} targeting: {root_cause}",
        )
        for i in range(count)
    ]


async def evaluate_mutations_on_failures(
    mutations: List[PromptMutation],
    failure_traces: List[Dict],
) -> List[PromptMutation]:
    """
    Run each prompt mutation against the failure test cases.
    Uses Gemini to generate answers and evaluates them.
    """
    from agent.tools import retrieve_medical_context, format_context_for_prompt

    llm = get_langchain_llm()
    test_cases = failure_traces[:5]  # Use 5 failure cases for speed

    for mutation in mutations:
        scores_across_cases = []

        for case in test_cases:
            query = case.get("query", "")
            if not query:
                continue

            try:
                # Get context for this query
                docs = retrieve_medical_context(query, top_k=3)
                context = format_context_for_prompt(docs)

                # Generate answer with this mutation's prompt
                full_prompt = mutation.prompt.format(
                    context=context, query=query
                )
                response = await llm.ainvoke(full_prompt)
                answer = response.content

                # Evaluate the answer
                from evolution.evaluator import evaluate_answer
                scores, _ = await evaluate_answer(query, answer)
                avg = sum(scores.values()) / len(scores)
                scores_across_cases.append(avg)

            except Exception as exc:
                logger.warning(f"Mutation eval case failed: {exc}")
                scores_across_cases.append(5.0)

        mutation.evaluated_score = (
            round(sum(scores_across_cases) / len(scores_across_cases), 2)
            if scores_across_cases
            else 5.0
        )
        logger.info(f"Mutation '{mutation.name}': avg_score={mutation.evaluated_score}")

    return mutations


# ─────────────────────────────────────────────────────────────────────────────
# Main Evolution Cycle
# ─────────────────────────────────────────────────────────────────────────────


async def run_evolution_cycle() -> Optional[EvolutionResult]:
    """
    Full autonomous evolution cycle using Phoenix MCP as the active tool.

    Step 1: Get failure traces from Phoenix
    Step 2: Diagnose root cause with Gemini
    Step 3: Generate 3 prompt mutations
    Step 4: Create Phoenix A/B experiment
    Step 5: Evaluate mutations on failure cases
    Step 6: Select winning prompt
    Step 7: Promote winner to Phoenix Prompt Hub
    Step 8: Curate Golden Dataset from successes
    """
    global _is_running, _current_avg_score

    if _is_running:
        logger.info("Evolution already in progress — skipping duplicate trigger")
        return None

    _is_running = True
    start_time = time.time()
    cycle_id = f"evo-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    current_version, current_prompt = get_active_prompt()

    logger.info(f"🔄 Evolution cycle started: {cycle_id}")

    try:
        phoenix = get_phoenix_client()

        # ── Step 1: Get failure traces ────────────────────────────────────────
        logger.info("Step 1: Fetching failure traces from Phoenix…")
        failure_traces = await phoenix.get_failure_traces(
            score_threshold=settings.evolution_score_threshold,
            limit=20,
        )

        if len(failure_traces) < 2:
            logger.info(f"Insufficient failures ({len(failure_traces)}) — skipping evolution")
            _is_running = False
            return None

        old_avg = sum(t.get("avg_score", 5.0) for t in failure_traces) / len(failure_traces)
        logger.info(f"Found {len(failure_traces)} failure traces (avg={old_avg:.2f})")

        # ── Step 2: Diagnose root cause ───────────────────────────────────────
        logger.info("Step 2: Diagnosing failure root causes…")
        root_cause = await diagnose_failures(failure_traces)

        # ── Step 3: Generate prompt mutations ────────────────────────────────
        logger.info("Step 3: Generating 3 prompt mutations…")
        mutations = await generate_prompt_mutations(
            current_prompt, root_cause, count=3
        )

        # ── Step 4: Create Phoenix experiment ────────────────────────────────
        logger.info("Step 4: Creating Phoenix A/B experiment…")
        experiment_id = await phoenix.create_experiment(
            mutations=[{"name": m.name, "prompt": m.prompt} for m in mutations],
            test_cases=failure_traces[:5],
        )

        # ── Step 5: Evaluate mutations ────────────────────────────────────────
        logger.info("Step 5: Evaluating mutations on failure cases…")
        evaluated_mutations = await evaluate_mutations_on_failures(
            mutations, failure_traces
        )

        # Store results in Phoenix experiment
        results = [
            {
                "mutation_index": i,
                "name": m.name,
                "prompt": m.prompt,
                "avg_score": m.evaluated_score,
            }
            for i, m in enumerate(evaluated_mutations)
        ]
        phoenix.set_experiment_results(experiment_id, results)

        # ── Step 6: Select winner ─────────────────────────────────────────────
        logger.info("Step 6: Selecting winning prompt variant…")
        winner = max(evaluated_mutations, key=lambda m: m.evaluated_score)
        logger.info(
            f"🏆 Winner: '{winner.name}' | score={winner.evaluated_score} "
            f"(current={old_avg:.2f})"
        )

        # Only promote if it's actually better
        if winner.evaluated_score <= old_avg:
            logger.info("No improvement found — keeping current prompt")
            _is_running = False
            return EvolutionResult(
                cycle_id=cycle_id,
                timestamp=datetime.utcnow().isoformat(),
                old_version=current_version,
                new_version=current_version,
                old_score=round(old_avg, 2),
                new_score=round(winner.evaluated_score, 2),
                improvement=0.0,
                root_cause=root_cause,
                mutations_tested=len(mutations),
                winner_name="none",
                golden_examples_added=0,
                duration_seconds=round(time.time() - start_time, 1),
                status="no_improvement",
            )

        # ── Step 7: Promote to Phoenix Prompt Hub ─────────────────────────────
        existing_versions = list_prompt_versions()
        new_version = f"v{len(existing_versions) + 1}"

        logger.info(f"Step 7: Promoting '{winner.name}' as {new_version} to Phoenix Prompt Hub…")

        update_active_prompt(new_version, winner.prompt)
        await phoenix.update_prompt_hub(
            prompt_name="medtrace_system_prompt",
            new_prompt=winner.prompt,
            version=new_version,
        )

        # Update current score tracking
        _current_avg_score = winner.evaluated_score

        # ── Step 8: Curate Golden Dataset ────────────────────────────────────
        logger.info("Step 8: Curating Golden Dataset from high-scoring traces…")
        all_traces = phoenix.get_recent_traces(limit=100)
        high_quality = [t for t in all_traces if t.get("avg_score", 0) >= 8.0]
        golden_added = await phoenix.add_to_golden_dataset(
            "golden_medical_qa", high_quality
        )

        # Record evolution result
        result = EvolutionResult(
            cycle_id=cycle_id,
            timestamp=datetime.utcnow().isoformat(),
            old_version=current_version,
            new_version=new_version,
            old_score=round(old_avg, 2),
            new_score=round(winner.evaluated_score, 2),
            improvement=round(winner.evaluated_score - old_avg, 2),
            root_cause=root_cause,
            mutations_tested=len(mutations),
            winner_name=winner.name,
            golden_examples_added=golden_added,
            duration_seconds=round(time.time() - start_time, 1),
            status="success",
        )

        _evolution_history.append(result)
        logger.info(
            f"✅ Evolution cycle complete: {current_version}→{new_version} | "
            f"Score {old_avg:.2f}→{winner.evaluated_score:.2f} "
            f"(+{winner.evaluated_score - old_avg:.2f})"
        )
        return result

    except Exception as exc:
        logger.error(f"❌ Evolution cycle failed: {exc}")
        _evolution_history.append(
            EvolutionResult(
                cycle_id=cycle_id,
                timestamp=datetime.utcnow().isoformat(),
                old_version=current_version,
                new_version=current_version,
                old_score=0.0,
                new_score=0.0,
                improvement=0.0,
                root_cause="Error during analysis",
                mutations_tested=0,
                winner_name="none",
                golden_examples_added=0,
                duration_seconds=round(time.time() - start_time, 1),
                status="error",
                error=str(exc),
            )
        )
        return None
    finally:
        _is_running = False


def get_evolution_history() -> List[Dict]:
    """Return serializable evolution history for the dashboard."""
    return [
        {
            "cycle_id": r.cycle_id,
            "timestamp": r.timestamp,
            "old_version": r.old_version,
            "new_version": r.new_version,
            "old_score": r.old_score,
            "new_score": r.new_score,
            "improvement": r.improvement,
            "root_cause": r.root_cause,
            "mutations_tested": r.mutations_tested,
            "winner_name": r.winner_name,
            "golden_examples_added": r.golden_examples_added,
            "duration_seconds": r.duration_seconds,
            "status": r.status,
            "error": r.error,
        }
        for r in _evolution_history
    ]


def update_current_score(score: float) -> None:
    """Called after each agent run to track current performance."""
    global _current_avg_score
    _current_avg_score = score


def get_current_avg_score() -> float:
    return _current_avg_score


def is_evolution_running() -> bool:
    return _is_running
