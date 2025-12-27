"""test cli command: python -m main index --docs ./data --rebuild-index"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("docqa_agent.vectorstore")


def _ensure_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    # Uses GOOGLE_API_KEY from environment (.env)
    # Model names can evolve; this is the common Gemini embedding model name.
    # If your account rejects it, we’ll swap to the exact supported name from the error message.
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def build_or_load_chroma(persist_dir: str, collection_name: str, embeddings) -> Chroma:
    persist_dir = _ensure_dir(persist_dir)
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def rebuild_index(vectordb: Chroma, chunks: List[Document]) -> None:
    logger.info("Rebuilding index: clearing existing vectors...")
    try:
        vectordb._collection.delete(where={})
    except Exception:
        logger.warning("Could not clear collection; continuing with add_documents.")

    logger.info("Adding %d chunks to vector store...", len(chunks))
    vectordb.add_documents(chunks)
    logger.info("Index rebuild complete.")


def similarity_search(vectordb: Chroma, query: str, k: int = 3) -> List[Document]:
    return vectordb.similarity_search(query, k=k)
