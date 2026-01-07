import json
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from docqa_agent.schema import QAResponse, Citation

load_dotenv()

INSUFFICIENT_MSG = "Insufficient evidence in the provided documents."


def build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )
    
def build_llm_hf() -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id = "meta-llama/Llama-3.1-8B-Instruct",
        task = "text-generation"
    )

def _evidence_is_sufficient(docs: List[Document], min_total_chars: int = 600) -> bool:
    if not docs:
        return False
    total = 0
    for d in docs:
        total += len((d.page_content or "").strip())
    return total >= min_total_chars


def _make_chunk_map(docs: List[Document]) -> Dict[str, Document]:
    """
    Map chunk_id -> Document for reliable quote filling and citation validation.
    """
    m: Dict[str, Document] = {}
    for d in docs:
        meta = d.metadata or {}
        cid = meta.get("chunk_id")
        if cid:
            m[str(cid)] = d
    return m


def _short_quote(text: str, max_len: int = 240) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "..."


def _format_context(docs: List[Document]) -> str:
    """
    Provide context with stable identifiers so the model can cite correctly.
    """
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source_file")
        page = meta.get("page")
        chunk_id = meta.get("chunk_id")
        text = (d.page_content or "").strip()

        blocks.append(
            f"CHUNK\n"
            f"source_file: {src}\n"
            f"page: {page}\n"
            f"chunk_id: {chunk_id}\n"
            f"text: {text}\n"
        )
    return "\n".join(blocks)


def _compute_confidence(scores: Optional[List[float]]) -> float:
    """
    Returns:
      - float in [0,1] if scores are provided
      - None if scores are not available (e.g., MMR)
    """
    if scores is None:
        return None
    if not scores:
        return 0.0

    top = scores[:5]
    cleaned = []
    for x in top:
        try:
            x = float(x)
        except Exception:
            continue
        # clamp score into [0, 1]
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        cleaned.append(x)

    if not cleaned:
        return 0.0

    avg = sum(cleaned) / len(cleaned)
    conf = avg * 0.95  # conservative scaling

    # FINAL clamp (most important)
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0
    return conf


def build_structured_answer(
    llm: ChatGoogleGenerativeAI,
    question: str,
    retrieved_docs: List[Document],
    retrieved_scores: Optional[List[float]] = None,
) -> QAResponse:
    """
    Always returns a valid QAResponse (strict schema).
    If parsing fails, returns fallback JSON (insufficient evidence).
    """
    confidence_opt = _compute_confidence(retrieved_scores)
    confidence = 0.0 if confidence_opt is None else confidence_opt

    # Always clamp
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    if not _evidence_is_sufficient(retrieved_docs):
        return QAResponse(
            question=question,
            answer=INSUFFICIENT_MSG,
            citations=[],
            confidence=min(confidence, 0.25),
            insufficient_evidence=True,
        )

    parser = PydanticOutputParser(pydantic_object=QAResponse)
    format_instructions = parser.get_format_instructions()

    context = _format_context(retrieved_docs)
    chunk_map = _make_chunk_map(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful Document QA assistant.\n"
                "Rules:\n"
                "1) Use ONLY the provided chunks.\n"
                f"2) If evidence is insufficient, answer MUST be exactly: {INSUFFICIENT_MSG}\n"
                "3) Citations must reference ONLY chunk_id values present in the context.\n"
                "4) Each citation must include a short quote copied from the cited chunk text.\n"
                "5) Output MUST follow the given JSON schema exactly.\n",
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "{format_instructions}\n",
            ),
        ]
    )

    messages = prompt.format_messages(
        question=question,
        context=context,
        format_instructions=format_instructions,
    )

    try:
        raw = llm.invoke(messages)
        parsed: QAResponse = parser.parse(str(raw.content))

        # If model refused, force canonical refusal shape
        if parsed.answer.strip() == INSUFFICIENT_MSG:
            return QAResponse(
                question=question,
                answer=INSUFFICIENT_MSG,
                citations=[],
                confidence=min(confidence, 0.25),
                insufficient_evidence=True,
            )

        # Validate/repair citations:
        fixed_citations: List[Citation] = []
        
        for c in parsed.citations:
            # Drop invented chunk_ids
            if c.chunk_id not in chunk_map:
                continue

            doc = chunk_map[c.chunk_id]
            meta = doc.metadata or {}
            src = str(meta.get("source_file"))
            page = meta.get("page")

            # Ensure quote is actually from the chunk (or fill it)
            # quote = c.quote.strip() if c.quote else ""
            # if not quote:
            #     quote = _short_quote(doc.page_content)
            # else:
            #     # Keep it short; don’t trust model to be concise
            #     quote = _short_quote(quote)

            fixed_citations.append(
                Citation(
                    source_file=src,
                    page=page if isinstance(page, int) else None,
                    chunk_id=str(meta.get("chunk_id")),
                    # quote=quote,
                )
            )
            
        MAX_CITATIONS = 6

        # de-duplicate by chunk_id, keep first occurrence
        seen = set()
        deduped: List[Citation] = []
        for c in fixed_citations:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            deduped.append(c)

        # cap
        fixed_citations = deduped[:MAX_CITATIONS]

        # If the model answered but gave no valid citations, treat as insufficient.
        if not fixed_citations:
            return QAResponse(
                question=question,
                answer=INSUFFICIENT_MSG,
                citations=[],
                confidence=min(confidence, 0.25),
                insufficient_evidence=True,
            )
            
        # If scores were unavailable (MMR), derive a conservative confidence from evidence.
        if confidence_opt is None:
            # Stronger when more citations exist, but keep conservative bounds.
            confidence = 0.15 + 0.05 * len(fixed_citations)
            if confidence > 0.45:
                confidence = 0.45

        # Final response
        return QAResponse(
            question=question,
            answer=parsed.answer.strip(),
            citations=fixed_citations,
            confidence=confidence,
            insufficient_evidence=False,
        )

    except Exception:
        # Hard fallback: always valid JSON, always safe.
        return QAResponse(
            question=question,
            answer=INSUFFICIENT_MSG,
            citations=[],
            confidence=min(confidence, 0.1),
            insufficient_evidence=True,
        )
        
