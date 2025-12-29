from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

INSUFFICIENT_MSG = "Insufficient evidence in the provided documents."


@dataclass
class RagAnswer:
    answer_text: str
    citations: List[dict]
    insufficient_evidence: bool


def build_llm() -> ChatGoogleGenerativeAI:
    # Keep it deterministic-ish for QA
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )
    
def build_llm_hf() -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id = "openai/gpt-oss-20b",
        task = "text-generation"
    )

def _format_context(docs: List[Document]) -> Tuple[str, List[dict]]:
    """
    Build a context string and a citation table.
    Each chunk is assigned a small local id: C1, C2, ...
    """
    lines = []
    citations = []

    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        cid = meta.get("chunk_id")
        src = meta.get("source_file")
        page = meta.get("page")

        chunk_tag = f"C{i}"
        text = d.page_content.strip()

        lines.append(f"[{chunk_tag}] source={src} page={page} chunk_id={cid}\n{text}")

        citations.append(
            {
                "chunk_tag": chunk_tag,
                "source_file": src,
                "page": page,
                "chunk_id": cid,
            }
        )

    return "\n\n".join(lines), citations


def _evidence_is_sufficient(docs: List[Document], min_total_chars: int = 600) -> bool:
    """
    Simple, deterministic gate:
    - if no docs -> insufficient
    - if total retrieved text is too short -> likely weak retrieval
    This is intentionally conservative; we’ll refine later.
    """
    if not docs:
        return False

    total = 0
    for d in docs:
        total += len((d.page_content or "").strip())

    return total >= min_total_chars


def answer_question(
    llm: ChatGoogleGenerativeAI,
    retrieved_docs: List[Document],
    question: str,
) -> RagAnswer:
    if not _evidence_is_sufficient(retrieved_docs):
        return RagAnswer(
            answer_text=INSUFFICIENT_MSG,
            citations=[],
            insufficient_evidence=True,
        )

    context_text, citation_table = _format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful Document QA assistant.\n"
                "Rules:\n"
                "1) Answer ONLY using the provided context.\n"
                "2) If the context does not contain enough evidence, reply exactly:\n"
                f"{INSUFFICIENT_MSG}\n"
                "3) When you make a claim, cite the supporting chunk tags like [C1], [C2].\n"
                "4) Do not use outside knowledge.\n",
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Write a helpful answer. Include chunk-tag citations inline like [C1].",
            ),
        ]
    )

    msg = prompt.format_messages(question=question, context=context_text)
    resp = llm.invoke(msg)
    text = str(resp.content).strip()

    # If the model refuses, we keep it as refusal.
    if text == INSUFFICIENT_MSG:
        return RagAnswer(
            answer_text=text,
            citations=[],
            insufficient_evidence=True,
        )

    # For now, citations output is the retrieved set. Phase 6 will extract exact quotes.
    return RagAnswer(
        answer_text=text,
        citations=citation_table,
        insufficient_evidence=False,
    )
