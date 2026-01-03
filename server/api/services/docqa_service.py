import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from langchain_huggingface import ChatHuggingFace

from docqa_agent.ingest import load_documents_from_folder
from docqa_agent.chunking import chunk_documents
from docqa_agent.vectorstore import (
    build_or_load_chroma,
    rebuild_index_fresh,
    similarity_search_with_scores,
    build_embeddings,
    build_embeddings_hf,
)
from docqa_agent.structured_rag import (
    build_structured_answer,
    build_llm,
    build_llm_hf,
    INSUFFICIENT_MSG,
)
from docqa_agent.rag import answer_question

# ---- process-wide singletons (per Django worker) ----
_LOCK = threading.Lock()

_cached: Dict[Tuple[str, str], Any] = {}   # (index_dir, collection_name) -> vectordb
_cached_embeddings: Dict[str, Any] = {}    # "google"|"hf" -> embeddings
_cached_llm: Dict[str, Any] = {}           # "google"|"hf" -> llm


@dataclass
class DocQAConfig:
    index_dir: str
    collection_name: str


def _get_embeddings(embedding: str):
    if embedding in _cached_embeddings:
        return _cached_embeddings[embedding]

    if embedding == "google":
        emb = build_embeddings()
    elif embedding == "hf":
        emb = build_embeddings_hf()
    else:
        raise ValueError("Unsupported embedding")

    _cached_embeddings[embedding] = emb
    return emb


def _get_llm(llm_model: str):
    if llm_model in _cached_llm:
        return _cached_llm[llm_model]

    if llm_model == "google":
        llm = build_llm()
    elif llm_model == "hf":
        llm = ChatHuggingFace(llm=build_llm_hf())
    else:
        raise ValueError("Unsupported llm_model")

    _cached_llm[llm_model] = llm
    return llm


def _get_vectordb(cfg: DocQAConfig, embeddings):
    key = (cfg.index_dir, cfg.collection_name)
    if key in _cached:
        return _cached[key]

    db = build_or_load_chroma(
        persist_dir=cfg.index_dir,
        collection_name=cfg.collection_name,
        embeddings=embeddings,
    )
    _cached[key] = db
    return db


def rebuild_index(cfg: DocQAConfig, docs_path: str, embedding: str) -> Dict[str, Any]:
    """
    Rebuild index safely (lock prevents concurrent rebuilds).
    """
    with _LOCK:
        embeddings = _get_embeddings(embedding)
        docs = load_documents_from_folder(docs_path)
        chunks = chunk_documents(docs)

        db = rebuild_index_fresh(
            persist_dir=cfg.index_dir,
            collection_name=cfg.collection_name,
            embeddings=embeddings,
            chunks=chunks,
        )

        # update cache
        _cached[(cfg.index_dir, cfg.collection_name)] = db

        return {"status": "ok", "documents": len(docs), "chunks": len(chunks)}


def ask(cfg: DocQAConfig, question: str, k: int, embedding: str, llm_model: str) -> Dict[str, Any]:
    embeddings = _get_embeddings(embedding)
    db = _get_vectordb(cfg, embeddings)
    llm = _get_llm(llm_model)

    scored = similarity_search_with_scores(db, question, k=k)
    docs = [d for (d, s) in scored]
    # scores = [float(s) for (d, s) in scored]
    
    resp = answer_question(
        llm=llm,
        retrieved_docs=docs,
        question=question,
    )
    # Human-friendly response + citations (no JSON schema changes here)

    # For /ask: return answer + minimal citations list (still chunk_id-based)
    return {
            "answer": resp.answer_text,
            "citations": resp.citations,
            "insufficient_evidence": resp.insufficient_evidence,
        }



def ask_json(cfg: DocQAConfig, question: str, k: int, embedding: str, llm_model: str) -> Dict[str, Any]:
    embeddings = _get_embeddings(embedding)
    db = _get_vectordb(cfg, embeddings)
    llm = _get_llm(llm_model)

    scored = similarity_search_with_scores(db, question, k=k)
    docs = [d for (d, s) in scored]
    scores = [float(s) for (d, s) in scored]

    resp = build_structured_answer(
        llm=llm,
        question=question,
        retrieved_docs=docs,
        retrieved_scores=scores,
    )

    return resp.model_dump()
