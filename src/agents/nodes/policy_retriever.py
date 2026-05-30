"""Node: Retrieves relevant policy documents from ChromaDB vector store for decision context."""
import os
import glob

import chromadb
from chromadb.utils import embedding_functions

from src.agents.state import AgentState
from src.agents.observability import traced

POLICIES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "policies")

_client = chromadb.Client()
_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    ef = embedding_functions.DefaultEmbeddingFunction()
    _collection = _client.get_or_create_collection(
        name="risk_policies",
        embedding_function=ef,
    )

    if _collection.count() > 0:
        return _collection

    policy_files = glob.glob(os.path.join(POLICIES_DIR, "*.txt"))
    documents = []
    metadatas = []
    ids = []

    for fpath in policy_files:
        with open(fpath, "r") as f:
            text = f.read()

        fname = os.path.basename(fpath)
        chunks = _split_into_chunks(text, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": fname, "chunk_index": i})
            ids.append(f"{fname}_{i}")

    if documents:
        _collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return _collection


def _split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _build_query(state: AgentState) -> str:
    parts = []

    risk = state.get("risk_score", 0)
    if risk > 0.8:
        parts.append("high risk merchant block criteria")
    elif risk > 0.5:
        parts.append("review threshold investigation criteria")

    merchant = state.get("merchant_info", {})
    mcc = merchant.get("mcc_description", "")
    if mcc:
        parts.append(f"MCC {merchant.get('mcc_code', '')} {mcc} risk requirements")

    txn = state.get("transactions_summary", {})
    if txn.get("pct_card_present", 1) < 0.6:
        parts.append("card not present CNP laundering indicators")
    if txn.get("pct_international", 0) > 0.15:
        parts.append("foreign card international transaction policy")

    kyb = state.get("kyb_status", {})
    if kyb.get("flags"):
        parts.append("KYB verification flags business age concern")

    context = state.get("context_findings", [])
    if any(f.get("classification") == "EXPLAINS_ANOMALY" for f in context):
        parts.append("legitimate event seasonal pattern approval criteria")

    if not parts:
        parts.append("risk scoring thresholds decision criteria")

    return " ".join(parts)


@traced(name="policy_retriever")
def policy_retriever(state: AgentState) -> dict:
    try:
        collection = _get_collection()
        query = _build_query(state)

        results = collection.query(query_texts=[query], n_results=5)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        policy_sections = []
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "unknown")
            policy_sections.append(f"[{source}]\n{doc}")

        policy_context = "\n\n---\n\n".join(policy_sections)

        sources = list({m.get("source", "unknown") for m in metas})
        return {
            "policy_context": policy_context,
            "reasoning_trace": [
                f"[PolicyRetriever] Retrieved {len(docs)} policy chunks from {len(sources)} documents: {', '.join(sources)}"
            ],
        }

    except Exception as e:
        return {
            "policy_context": "",
            "reasoning_trace": [
                f"[PolicyRetriever] Failed ({type(e).__name__}: {str(e)[:80]}), proceeding without policy context"
            ],
        }
