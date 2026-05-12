"""
RAG + Search Tools for MedTrace.
Wraps ChromaDB retrieval as a LangChain tool usable inside LangGraph nodes.
"""

import time
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_core.documents import Document
from loguru import logger


def get_retriever(top_k: int = 5):
    """Lazy-load ChromaDB retriever to avoid circular imports."""
    from knowledge_base.vectorstore import get_vectorstore

    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.3},
    )


def retrieve_medical_context(
    query: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant medical documents for a query.

    Returns a list of dicts with:
      - content: the document text
      - source: metadata source field
      - topic: topic category
      - score: similarity score (estimated)
    """
    try:
        from knowledge_base.vectorstore import get_vectorstore

        vs = get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=top_k)

        docs = []
        for doc, score in results:
            docs.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "MedTraceKB"),
                    "topic": doc.metadata.get("topic", "General Medicine"),
                    "confidence": doc.metadata.get("confidence", "medium"),
                    "score": round(float(score), 4),
                }
            )

        logger.debug(f"Retrieved {len(docs)} docs for query: '{query[:60]}…'")
        return docs

    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}")
        return []


def format_context_for_prompt(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a prompt-ready context string."""
    if not docs:
        return "No relevant medical documents found in knowledge base."

    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Document {i}] Source: {doc['source']} | Topic: {doc['topic']}\n"
            f"{doc['content']}\n"
        )
    return "\n---\n".join(parts)


@tool
def medical_rag_search(query: str) -> str:
    """
    Search the MedTrace medical knowledge base for relevant information.
    Returns formatted context from ChromaDB vector store.
    """
    docs = retrieve_medical_context(query, top_k=5)
    return format_context_for_prompt(docs)


@tool
def check_drug_interaction(drug_a: str, drug_b: str) -> str:
    """
    Check for known interactions between two medications.
    Searches the medical knowledge base for interaction data.
    """
    query = f"drug interaction between {drug_a} and {drug_b}"
    docs = retrieve_medical_context(query, top_k=3)
    return format_context_for_prompt(docs)


@tool
def get_dosage_info(medication: str, condition: str = "") -> str:
    """
    Retrieve dosage information for a medication.
    Optionally specify the condition being treated.
    """
    query = f"dosage {medication}" + (f" for {condition}" if condition else "")
    docs = retrieve_medical_context(query, top_k=3)
    return format_context_for_prompt(docs)


# Registry of all tools available to the agent
MEDTRACE_TOOLS = [medical_rag_search, check_drug_interaction, get_dosage_info]
