"""test cli command: python -m main index --docs ./data --rebuild-index"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logger = logging.getLogger("docqa_agent.vectorstore")


def _ensure_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def wipe_persist_dir(persist_dir: str) -> None:
    p = Path(persist_dir).expanduser().resolve()
    if p.exists() and p.is_dir():
        logger.info("Deleting persist directory for clean rebuild: %s", p)
        shutil.rmtree(p)


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def build_embeddings_hf() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def build_or_load_chroma(persist_dir: str, collection_name: str, embeddings) -> Chroma:
    persist_dir = _ensure_dir(persist_dir)
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

def rebuild_index_fresh(
    persist_dir: str,
    collection_name: str,
    embeddings,
    chunks: List[Document],
) -> Chroma:
    wipe_persist_dir(persist_dir)

    vectordb = build_or_load_chroma(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    ids = []
    for c in chunks:
        cid = c.metadata.get("chunk_id")
        if not cid:
            raise ValueError("Missing chunk_id in chunk metadata.")
        ids.append(str(cid))

    logger.info("Adding %d chunks to vector store...", len(chunks))
    vectordb.add_documents(chunks, ids=ids)
    logger.info("Index rebuild complete.")

    return vectordb

def similarity_search(vectordb: Chroma, query: str, k: int = 3) -> List[Document]:
    return vectordb.similarity_search(query, k=k)

def similarity_search_with_scores(vectordb: Chroma, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    Returns (Document, relevance_score) where score is usually in [0, 1] for Chroma.
    Exact scaling can vary, so we treat it as a heuristic.
    """
    return vectordb.similarity_search_with_relevance_scores(query, k=k)