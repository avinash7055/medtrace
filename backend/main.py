"""
MedTrace FastAPI application entry point.
Initialises Phoenix tracing, loads the knowledge base, mounts all routes.
"""
import os, asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

os.makedirs("logs", exist_ok=True)

from config import setup_phoenix_tracing, settings
from api.routes import router
from knowledge_base.loader import initialize_knowledge_base

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MedTrace starting up…")
    setup_phoenix_tracing()
    try:
        count = initialize_knowledge_base()
        logger.info(f"Knowledge base: {count} documents ready")
    except Exception as e:
        logger.warning(f"KB init warning: {e} — continuing without full KB")
    yield
    logger.info("MedTrace shutting down")

app = FastAPI(
    title="MedTrace API",
    description="Self-Evolving Medical Information Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {
        "name": "MedTrace",
        "description": "Self-Evolving Medical Information Agent",
        "version": "1.0.0",
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
