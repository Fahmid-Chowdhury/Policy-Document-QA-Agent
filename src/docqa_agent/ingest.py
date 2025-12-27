"""test cli command: python -m main ingest --docs ./data"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

logger = logging.getLogger("docqa_agent.ingest")


SUPPORTED_EXTS = {".pdf", ".txt", ".docx"}


def _detect_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".txt":
        return "txt"
    if ext == ".docx":
        return "docx"
    return "unknown"


def _load_one_file(path: Path) -> List[Document]:
    file_type = _detect_file_type(path)

    if file_type == "pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    if file_type == "txt":
        # autodetect_encoding=True helps avoid common Windows encoding issues
        loader = TextLoader(str(path), autodetect_encoding=True)
        return loader.load()

    if file_type == "docx":
        loader = Docx2txtLoader(str(path))
        return loader.load()

    return []


def _normalize_metadata(
    doc: Document,
    source_root: Path,
    source_path: Path,
    file_type: str
) -> Document:
    # Keep existing metadata too, but ensure we have our standard keys.
    meta = dict(doc.metadata) if doc.metadata else {}
    print(int(meta["page"]))

    # Store relative path for portability across machines
    try:
        rel = source_path.relative_to(source_root)
        source_file = str(rel).replace("\\", "/")
    except ValueError:
        source_file = str(source_path.name)

    page: Optional[int] = None
    if "page" in meta:
        # PyPDFLoader sets 'page' as int (0-based). We'll keep it as-is.
        try:
            page = int(meta["page"])
        except Exception:
            page = None

    meta["source_file"] = source_file
    meta["file_type"] = file_type
    meta["page"] = page

    return Document(page_content=doc.page_content, metadata=meta)


def load_documents_from_folder(folder_path: str) -> List[Document]:
    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Docs folder not found or not a directory: {root}")

    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)

    if not paths:
        logger.warning("No supported documents found under: %s", root)
        return []

    all_docs: List[Document] = []

    for path in sorted(paths):
        file_type = _detect_file_type(path)
        logger.info("Loading: %s", path.name)

        try:
            docs = _load_one_file(path)
        except Exception as e:
            logger.exception("Failed to load %s (%s): %s", path, file_type, e)
            continue

        for d in docs:
            all_docs.append(_normalize_metadata(d, root, path, file_type))

    return all_docs