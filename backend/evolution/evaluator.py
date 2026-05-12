"""
LLM-as-Judge Evaluator for MedTrace.

Scores every Gemini answer on 5 medical quality rubrics using Gemini itself.
Rubrics: medical_accuracy, completeness, safety, clarity, citation_quality

Returns structured float scores (0–10) and textual feedback.
Low scores (avg < 6.5) trigger the autonomous evolution engine.
"""

import json
import re
from typing import Dict, Tuple

from phoenix.trace import suppress_tracing

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.prompts import EVALUATION_PROMPT
from config import get_langchain_llm


def _parse_scores(text: str) -> Dict[str, float]:
    """
    Robustly extract a JSON score block from LLM output.
    Falls back to regex extraction if strict JSON parse fails.
    """
    # Try strict JSON first
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group())
            scores = {}
            for key in ["medical_accuracy", "completeness", "safety", "clarity", "citation_quality"]:
                val = raw.get(key, 5.0)
                scores[key] = max(0.0, min(10.0, float(val)))
            return scores
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Regex fallback — extract "key": value patterns
    scores = {}
    keys = ["medical_accuracy", "completeness", "safety", "clarity", "citation_quality"]
    for key in keys:
        pattern = rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        m = re.search(pattern, text)
        if m:
            scores[key] = max(0.0, min(10.0, float(m.group(1))))
        else:
            scores[key] = 5.0  # neutral default

    return scores


def _parse_feedback(text: str) -> str:
    """Extract overall_feedback string from LLM response."""
    match = re.search(r'"overall_feedback"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1)
    return "Evaluation completed."


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def evaluate_answer(
    query: str, answer: str
) -> Tuple[Dict[str, float], str]:
    """
    Evaluate a medical answer using Gemini as judge.

    Args:
        query:  The original medical question
        answer: The generated answer to evaluate

    Returns:
        Tuple of (scores_dict, feedback_string)
        scores_dict keys: medical_accuracy, completeness, safety, clarity, citation_quality
    """
    llm = get_langchain_llm()

    eval_prompt = EVALUATION_PROMPT.format(query=query, answer=answer)

    try:
        with suppress_tracing():  # Don't trace evaluation calls — keeps Phoenix clean
            response = await llm.ainvoke(eval_prompt)
        raw_text = response.content

        scores = _parse_scores(raw_text)
        feedback = _parse_feedback(raw_text)

        avg = sum(scores.values()) / len(scores)
        logger.debug(
            f"Eval scores → avg={avg:.2f} | "
            + " | ".join(f"{k}={v}" for k, v in scores.items())
        )

        return scores, feedback

    except Exception as exc:
        logger.error(f"Evaluator LLM call failed: {exc}")
        raise


async def batch_evaluate(
    qa_pairs: list[Dict],
) -> list[Dict]:
    """
    Batch-evaluate a list of {query, answer} dicts.
    Used by the evolution engine to assess prompt mutations at scale.
    """
    results = []
    for pair in qa_pairs:
        try:
            scores, feedback = await evaluate_answer(
                pair["query"], pair["answer"]
            )
            avg = sum(scores.values()) / len(scores)
            results.append({
                **pair,
                "eval_scores": scores,
                "avg_score": round(avg, 2),
                "eval_feedback": feedback,
            })
        except Exception as exc:
            logger.warning(f"Batch eval skipped one pair: {exc}")
            results.append({
                **pair,
                "eval_scores": {},
                "avg_score": 0.0,
                "eval_feedback": str(exc),
            })
    return results
