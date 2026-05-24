"""
MedTrace Configuration Module
Central configuration for all services: Gemini, Phoenix, ChromaDB, FastAPI.
Phoenix tracing is set up here and instruments the entire LangChain pipeline.
"""

import os
import sys
from typing import Optional
from functools import lru_cache

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Loguru configuration — pretty, coloured, structured logs
# ─────────────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/medtrace.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} | {message}",
)


class Settings(BaseSettings):
    """All application settings — loaded from environment / .env file."""

    # ── Gemini ──────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # ── Phoenix (self-hosted Docker) ──────────────────────────────────────────
    phoenix_base_url: str = "http://localhost:6006"
    phoenix_collector_endpoint: str = "http://localhost:4317"  # gRPC OTLP
    phoenix_project_name: str = "medtrace-agent"

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "medical_knowledge"

    # ── Evolution Engine ────────────────────────────────────────────────────
    evolution_score_threshold: float = 6.5
    evolution_min_failures: int = 5
    evolution_check_interval: int = 300  # seconds between auto-checks

    # ── FastAPI ─────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "*"]

    # ── Google Cloud ────────────────────────────────────────────────────────
    google_cloud_project: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Phoenix / OpenTelemetry Tracing Setup
# ─────────────────────────────────────────────────────────────────────────────
_tracer_provider = None


def setup_phoenix_tracing():
    """
    Registers Phoenix OTEL tracer and instruments the LangChain pipeline.
    Called once at application startup.

    Uses the self-hosted Phoenix Docker container (no API key required).
    Traces are exported via gRPC OTLP to port 4317.

    Phoenix is used as an ACTIVE SELF-IMPROVEMENT TOOL:
      - Failure traces are queried by the Evolution Engine
      - A/B experiments are created and evaluated via Phoenix
      - Winning prompts are promoted to Phoenix Prompt Hub
    """
    global _tracer_provider

    if _tracer_provider is not None:
        return _tracer_provider

    try:
        from phoenix.otel import register
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        # Self-hosted Phoenix requires no auth headers — clear any leftover cloud key
        os.environ.pop("OTEL_EXPORTER_OTLP_HEADERS", None)

        _tracer_provider = register(
            project_name=settings.phoenix_project_name,
            # gRPC OTLP collector on the Docker Phoenix container
            endpoint=settings.phoenix_collector_endpoint,
            set_global_tracer_provider=True,
            auto_instrument=True,  # covers LangChain + LangGraph node spans automatically
        )

        # Instrument direct google.generativeai SDK calls (outside LangChain)
        GoogleGenAIInstrumentor().instrument(tracer_provider=_tracer_provider)

        logger.info(
            f"✅ Phoenix tracing initialised → project='{settings.phoenix_project_name}' "
            f"endpoint='{settings.phoenix_collector_endpoint}'"
        )
    except Exception as exc:
        logger.warning(
            f"⚠️  Phoenix tracing setup failed (running without tracing): {exc}"
        )
        _tracer_provider = None

    return _tracer_provider


def get_tracer():
    """Return an OTEL tracer for manual span creation inside nodes."""
    from opentelemetry import trace

    return trace.get_tracer("medtrace.agent")


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client factory
# ─────────────────────────────────────────────────────────────────────────────
_gemini_client = None


def get_gemini_client():
    """Singleton Gemini generative model client."""
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        _gemini_client = genai.GenerativeModel(settings.gemini_model)
        logger.info(f"✅ Gemini client ready → model='{settings.gemini_model}'")
    return _gemini_client


def get_langchain_llm():
    """Return a LangChain-wrapped Gemini LLM for use inside LangGraph nodes."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )


from langchain_core.embeddings import Embeddings
from typing import List

class GeminiBatchEmbeddings(Embeddings):
    """Custom high-performance embedding class using google-genai SDK for batching."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch
                )
                for emb in response.embeddings:
                    results.append(emb.values)
            except Exception as e:
                logger.error(f"Batch embedding failed at chunk {i}: {e}")
                raise e
        return results

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise e


def get_embedding_model():
    """Return LangChain compatible Gemini batch embedding model for ChromaDB."""
    return GeminiBatchEmbeddings(
        model_name=settings.gemini_embedding_model,
        api_key=settings.gemini_api_key,
    )
