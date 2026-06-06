"""
Knowledge base loader for MedTrace.
Loads medical Q&A data, chunks it, embeds it into ChromaDB.
"""
import json, os, re, sys
from pathlib import Path
from typing import List, Dict

# Ensure the backend root is on sys.path so `knowledge_base` is importable
# whether this file is run directly or imported as a module.
_backend_root = str(Path(__file__).parent.parent)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger
from knowledge_base.vectorstore import get_vectorstore, reset_vectorstore

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
    # For medical Q&A, we keep each Q&A pair intact. Splitting them makes 
    # answers lose association with the questions, which hurts retrieval.
    logger.info(f"Preserving full Q&A pairs as individual documents → {len(docs)} docs")
    return docs

def initialize_knowledge_base(force_reload: bool = False) -> int:
    """Load data into ChromaDB. Resumes and only embeds missing documents if force_reload is False."""
    vs = get_vectorstore()
    count = vs._collection.count()
    
    existing_contents = set()
    if count > 0:
        if force_reload:
            logger.info("Force reload enabled — resetting ChromaDB collection to prevent duplicates")
            try:
                vs.delete_collection()
                reset_vectorstore()
                vs = get_vectorstore()
                count = 0
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
        else:
            logger.info(f"ChromaDB has {count} existing documents. Checking for documents to resume...")
            try:
                existing = vs.get()
                existing_contents = set(existing.get("documents", []) or [])
                logger.info(f"Loaded {len(existing_contents)} unique documents already in database.")
            except Exception as e:
                logger.warning(f"Failed to fetch existing documents, starting fresh: {e}")

    pairs = load_json_qa_files()
    if not pairs:
        logger.warning("No data files found in knowledge_base/data/")
        return count

    docs = qa_pairs_to_documents(pairs)
    
    # Filter out already indexed documents to support resuming
    if existing_contents:
        new_docs = [d for d in docs if d.page_content not in existing_contents]
        logger.info(f"Filter: {len(docs)} total source docs | {len(docs) - len(new_docs)} already in DB | {len(new_docs)} new docs to index.")
        docs = new_docs
    
    chunks = chunk_documents(docs)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        logger.info("[Success] All documents are already indexed in ChromaDB! Resume complete.")
        return vs._collection.count()
        
    logger.info(f"Starting database indexing: remaining chunks to embed = {total_chunks} of {len(pairs)} total.")

    batch_size = 90
    processed = 0
    import time
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        
        # Auto-retry logic with 60s delay to let sliding window reset
        retries = 5
        for attempt in range(retries):
            try:
                vs.add_documents(batch)
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    logger.warning(f"[Warning] Network / Rate limit issue: {err_str}. Waiting 60s before retry (attempt {attempt+1}/{retries})...")
                    time.sleep(60)
                else:
                    raise e
        else:
            raise RuntimeError("Exhausted all retries due to rate limits.")

        processed += len(batch)
        pct = (processed / total_chunks) * 100
        logger.info(f"Progress: {processed}/{total_chunks} chunks indexed ({pct:.1f}%) | Remaining: {total_chunks - processed} chunks")
        
        # No rate-limiting sleep needed for local Ollama embeddings
        pass


    total = vs._collection.count()
    logger.info(f"Knowledge base ready: {total} chunks indexed")
    return total

if __name__ == "__main__":
    print("Indexing real medical Q&A database into ChromaDB...")
    initialize_knowledge_base(force_reload=False)
    print("Done!")

