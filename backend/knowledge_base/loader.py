"""
Knowledge base loader for MedTrace.
Loads medical Q&A data, chunks it, embeds it into ChromaDB.
"""
import json, os, re
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger
from knowledge_base.vectorstore import get_vectorstore

DATA_DIR = Path(__file__).parent / "data"

def load_json_qa_files() -> List[Dict]:
    """Load all JSON Q&A files from data/ directory."""
    pairs = []
    for f in DATA_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                pairs.extend(data)
            elif isinstance(data, dict) and "qa_pairs" in data:
                pairs.extend(data["qa_pairs"])
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")
    logger.info(f"Loaded {len(pairs)} Q&A pairs from JSON files")
    return pairs

def qa_pairs_to_documents(pairs: List[Dict]) -> List[Document]:
    """Convert Q&A dicts to LangChain Documents."""
    docs = []
    for p in pairs:
        q = p.get("question", p.get("Q", ""))
        a = p.get("answer", p.get("A", ""))
        topic = p.get("topic", p.get("category", "General Medicine"))
        source = p.get("source", "MedTraceKB")
        if q and a:
            content = f"Question: {q}\n\nAnswer: {a}"
            docs.append(Document(
                page_content=content,
                metadata={"source": source, "topic": topic, "confidence": "high"}
            ))
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} docs → {len(chunks)} chunks")
    return chunks

def initialize_knowledge_base(force_reload: bool = False) -> int:
    """Load data into ChromaDB. Skips if already populated (unless force_reload)."""
    vs = get_vectorstore()
    count = vs._collection.count()
    if count > 0 and not force_reload:
        logger.info(f"Knowledge base already has {count} docs — skipping reload")
        return count

    pairs = load_json_qa_files()
    if not pairs:
        logger.warning("No data files found in knowledge_base/data/")
        return 0

    docs = qa_pairs_to_documents(pairs)
    chunks = chunk_documents(docs)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vs.add_documents(batch)
        logger.info(f"Indexed batch {i//batch_size+1}: {len(batch)} chunks")

    total = vs._collection.count()
    logger.info(f"Knowledge base ready: {total} chunks indexed")
    return total
