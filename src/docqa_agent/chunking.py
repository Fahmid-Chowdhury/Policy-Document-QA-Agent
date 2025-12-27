import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    # Baseline settings: good for general document QA.
    # We'll tune later based on your retrieval sanity tests.
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )


def _make_chunk_id(source_file: str, page: int, chunk_index: int, text: str) -> str:
    # Stable + short-ish ID. If content changes, chunk_id changes (good).
    # If same content appears, it gets same hash (also fine).
    base = f"{source_file}|{page}|{chunk_index}|{text[:200]}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return h[:12]


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = build_text_splitter()

    chunks: List[Document] = []
    split_docs = splitter.split_documents(docs)

    # Assign chunk_id in a predictable way.
    # We'll group by (source_file, page) so chunk indices are local.
    counters = {}

    for d in split_docs:
        meta = dict(d.metadata) if d.metadata else {}

        source_file = str(meta.get("source_file", "unknown"))
        page_val = meta.get("page")
        page = int(page_val) if isinstance(page_val, int) else -1

        key = (source_file, page)
        counters[key] = counters.get(key, 0) + 1
        chunk_index = counters[key]

        chunk_id = _make_chunk_id(source_file, page, chunk_index, d.page_content)
        meta["chunk_id"] = chunk_id
        meta["chunk_index"] = chunk_index

        chunks.append(Document(page_content=d.page_content, metadata=meta))

    return chunks
