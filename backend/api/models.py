"""Pydantic models for MedTrace API."""
from typing import Dict, List, Optional
from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class EvalScores(BaseModel):
    medical_accuracy: float = 0.0
    completeness: float = 0.0
    safety: float = 0.0
    clarity: float = 0.0
    citation_quality: float = 0.0

class AskResponse(BaseModel):
    query: str
    answer: str
    eval_scores: Dict[str, float]
    avg_score: float
    prompt_version: str
    trace_id: str
    citations: List[str]
    evolution_triggered: bool
    processing_time_ms: float
    error: Optional[str] = None

class TraceItem(BaseModel):
    trace_id: str
    timestamp: str
    query: str
    avg_score: float
    prompt_version: str
    evolution_triggered: bool

class EvolutionRecord(BaseModel):
    cycle_id: str
    timestamp: str
    old_version: str
    new_version: str
    old_score: float
    new_score: float
    improvement: float
    root_cause: str
    mutations_tested: int
    status: str

class MetricsResponse(BaseModel):
    total_queries: int
    avg_score: float
    evolutions_run: int
    golden_examples: int
    current_prompt_version: str
    evolution_running: bool

class GoldenExample(BaseModel):
    id: str
    query: str
    answer: str
    avg_score: float
    prompt_version: str
    added_at: str
