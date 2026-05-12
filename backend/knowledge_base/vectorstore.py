"""ChromaDB vector store singleton for MedTrace."""
import os
from loguru import logger
from config import settings

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    from langchain_chroma import Chroma
    from config import get_embedding_model
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    embeddings = get_embedding_model()
    _vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    count = _vectorstore._collection.count()
    logger.info(f"ChromaDB ready: {count} documents in '{settings.chroma_collection_name}'")
    return _vectorstore

def reset_vectorstore():
    global _vectorstore
    _vectorstore = None
