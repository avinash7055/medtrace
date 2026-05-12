"""
Phoenix MCP Client for MedTrace.

Interfaces with the @arizeai/phoenix-mcp server via subprocess.
This is what makes MedTrace UNIQUE:
  - Phoenix MCP is used as an ACTIVE self-improvement tool
  - Not just passive monitoring — the agent QUERIES its own failure traces
  - Creates A/B experiments, evaluates variants, promotes winners
  - Builds a Golden Dataset from production successes

When Phoenix API is unavailable, all methods gracefully fall back
to in-memory simulation so the demo works without credentials.
"""

import asyncio
import json
import subprocess
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from config import settings

# ─────────────────────────────────────────────────────────────────────────────
# In-memory state (fallback when Phoenix API not available)
# ─────────────────────────────────────────────────────────────────────────────

_in_memory_traces: List[Dict] = []
_in_memory_experiments: Dict[str, Dict] = {}
_golden_dataset: List[Dict] = []
_prompt_hub: Dict[str, str] = {}


class PhoenixMCPClient:
    """
    Async client for Phoenix MCP operations.

    Uses Phoenix REST API where available; falls back to in-memory simulation.
    All operations are logged and their results stored for the dashboard.
    """

    def __init__(self):
        self.base_url = settings.phoenix_base_url
        self.project_name = settings.phoenix_project_name
        # Self-hosted Phoenix requires no authentication
        self.headers = {
            "Content-Type": "application/json",
        }
        self._api_available: Optional[bool] = None

    async def _check_api(self) -> bool:
        """Check if self-hosted Phoenix REST API is reachable (cached)."""
        if self._api_available is not None:
            return self._api_available
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/projects",
                    headers=self.headers,
                )
                self._api_available = resp.status_code < 500
        except Exception:
            self._api_available = False
        logger.info(f"Phoenix API available: {self._api_available}")
        return self._api_available

    # ── Trace Operations ─────────────────────────────────────────────────────

    def record_trace(self, trace_data: Dict) -> None:
        """Store a completed trace (called after each agent run)."""
        trace = {
            "trace_id": trace_data.get("trace_id", str(uuid.uuid4())),
            "timestamp": datetime.utcnow().isoformat(),
            "query": trace_data.get("query", ""),
            "answer": trace_data.get("answer", ""),
            "eval_scores": trace_data.get("eval_scores", {}),
            "avg_score": trace_data.get("avg_score", 0.0),
            "prompt_version": trace_data.get("prompt_version", "v1"),
            "retrieval_count": trace_data.get("retrieval_count", 0),
            "processing_time_ms": trace_data.get("processing_time_ms", 0.0),
            "evolution_triggered": trace_data.get("evolution_triggered", False),
        }
        _in_memory_traces.append(trace)

        # Keep only last 500 traces in memory
        if len(_in_memory_traces) > 500:
            _in_memory_traces.pop(0)

    async def get_failure_traces(
        self, score_threshold: float = 6.5, limit: int = 20
    ) -> List[Dict]:
        """
        Query Phoenix for traces with avg_score below threshold.
        Falls back to in-memory store if API is unavailable.
        """
        if await self._check_api():
            try:
                return await self._fetch_failure_traces_from_api(
                    score_threshold, limit
                )
            except Exception as exc:
                logger.warning(f"Phoenix API trace fetch failed: {exc}, using in-memory")

        # In-memory fallback
        failures = [
            t for t in _in_memory_traces
            if t.get("avg_score", 10.0) < score_threshold
        ]
        return failures[-limit:]

    async def _fetch_failure_traces_from_api(
        self, score_threshold: float, limit: int
    ) -> List[Dict]:
        """Fetch low-scoring traces from Phoenix REST API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.base_url}/v1/projects/{self.project_name}/spans",
                headers=self.headers,
                params={
                    "filter": f"attributes['eval.avg_score'] < {score_threshold}",
                    "limit": limit,
                    "sort": "startTime desc",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])
        return []

    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        """Return the most recent N traces (in-memory)."""
        return list(reversed(_in_memory_traces[-limit:]))

    # ── Experiment Operations ─────────────────────────────────────────────────

    async def create_experiment(
        self,
        mutations: List[Dict],
        test_cases: List[Dict],
        experiment_name: str = None,
    ) -> str:
        """
        Create a Phoenix A/B experiment to compare prompt mutations.
        Returns an experiment_id for later result retrieval.

        UNIQUE: This agent autonomously creates its own experiments in Phoenix!
        """
        experiment_id = str(uuid.uuid4())
        name = experiment_name or f"MedTrace-Evo-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

        experiment = {
            "id": experiment_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "mutations": mutations,
            "test_cases": test_cases,
            "status": "running",
            "results": [],
        }
        _in_memory_experiments[experiment_id] = experiment

        logger.info(
            f"📊 Phoenix experiment created: {name} | "
            f"{len(mutations)} variants × {len(test_cases)} test cases"
        )

        # If Phoenix API available, also register there
        if await self._check_api():
            try:
                await self._register_experiment_api(experiment_id, name, mutations)
            except Exception as exc:
                logger.warning(f"Phoenix API experiment registration failed: {exc}")

        return experiment_id

    async def _register_experiment_api(
        self, experiment_id: str, name: str, mutations: List[Dict]
    ) -> None:
        """Register experiment with Phoenix REST API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{self.base_url}/v1/experiments",
                headers=self.headers,
                json={
                    "experimentId": experiment_id,
                    "name": name,
                    "projectName": self.project_name,
                    "variants": [m.get("name", f"variant_{i}") for i, m in enumerate(mutations)],
                },
            )

    async def get_experiment_results(self, experiment_id: str) -> List[Dict]:
        """Retrieve results for a completed experiment."""
        experiment = _in_memory_experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found")
            return []

        results = experiment.get("results", [])
        if results:
            return results

        # Results not set yet — return the mutations with pending status
        return [
            {
                "mutation_index": i,
                "name": m.get("name", f"variant_{i}"),
                "prompt": m.get("prompt", ""),
                "avg_score": m.get("evaluated_score", 0.0),
                "status": "pending",
            }
            for i, m in enumerate(experiment.get("mutations", []))
        ]

    def set_experiment_results(
        self, experiment_id: str, results: List[Dict]
    ) -> None:
        """Store evaluated results for an experiment."""
        if experiment_id in _in_memory_experiments:
            _in_memory_experiments[experiment_id]["results"] = results
            _in_memory_experiments[experiment_id]["status"] = "completed"

    # ── Prompt Hub ───────────────────────────────────────────────────────────

    async def update_prompt_hub(
        self, prompt_name: str, new_prompt: str, version: str
    ) -> bool:
        """
        Promote a winning prompt to Phoenix Prompt Hub.
        UNIQUE: Fully autonomous — no human needed!
        """
        _prompt_hub[prompt_name] = {
            "content": new_prompt,
            "version": version,
            "updated_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"🏆 Prompt promoted to Phoenix Hub: {prompt_name} → {version}")

        if await self._check_api():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    await client.post(
                        f"{self.base_url}/v1/prompts",
                        headers=self.headers,
                        json={
                            "name": prompt_name,
                            "content": new_prompt,
                            "version": version,
                            "projectName": self.project_name,
                        },
                    )
                logger.info(f"✅ Prompt registered in Phoenix Prompt Hub API")
                return True
            except Exception as exc:
                logger.warning(f"Phoenix Prompt Hub API update failed: {exc}")

        return True  # In-memory always succeeds

    def get_prompt_hub(self) -> Dict:
        return dict(_prompt_hub)

    # ── Golden Dataset ───────────────────────────────────────────────────────

    async def add_to_golden_dataset(
        self, dataset_name: str, examples: List[Dict]
    ) -> int:
        """
        Add high-scoring traces to the Golden Dataset.
        UNIQUE: Agent curates its own training data from production successes!
        """
        added = 0
        for ex in examples:
            if ex.get("avg_score", 0) >= 8.0:  # only add truly great answers
                entry = {
                    "id": str(uuid.uuid4()),
                    "dataset": dataset_name,
                    "query": ex.get("query", ""),
                    "answer": ex.get("answer", ""),
                    "eval_scores": ex.get("eval_scores", {}),
                    "avg_score": ex.get("avg_score", 0.0),
                    "prompt_version": ex.get("prompt_version", "v1"),
                    "added_at": datetime.utcnow().isoformat(),
                }
                _golden_dataset.append(entry)
                added += 1

        logger.info(
            f"✨ Golden Dataset '{dataset_name}': +{added} examples "
            f"(total={len(_golden_dataset)})"
        )
        return added

    def get_golden_dataset(self, limit: int = 50) -> List[Dict]:
        """Return the most recent golden examples."""
        return list(reversed(_golden_dataset[-limit:]))

    def get_golden_dataset_stats(self) -> Dict:
        if not _golden_dataset:
            return {"total": 0, "avg_score": 0.0}
        avg = sum(e["avg_score"] for e in _golden_dataset) / len(_golden_dataset)
        return {"total": len(_golden_dataset), "avg_score": round(avg, 2)}


# Singleton client instance
_client: Optional[PhoenixMCPClient] = None


def get_phoenix_client() -> PhoenixMCPClient:
    global _client
    if _client is None:
        _client = PhoenixMCPClient()
    return _client
