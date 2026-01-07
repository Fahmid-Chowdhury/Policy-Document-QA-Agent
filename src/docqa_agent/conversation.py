from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from docqa_agent.structured_rag import build_structured_answer, INSUFFICIENT_MSG


_HISTORY_STORE: Dict[str, ChatMessageHistory] = {}


def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _HISTORY_STORE:
        _HISTORY_STORE[session_id] = ChatMessageHistory()
    return _HISTORY_STORE[session_id]


def _looks_like_policy_question(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "policy", "leave", "sick", "annual", "study", "unpaid",
        "benefit", "salary", "termination", "notice", "disciplinary",
        "hours", "work", "overtime", "holiday", "procedure", "apply",
        "form", "manager", "approval", "certificate", "documentation",
    ]
    return any(k in t for k in keywords)


def contextualize_question(llm, history: ChatMessageHistory, question: str) -> str:
    """
    Rewrite the last question into a standalone question using conversation context.
    Do NOT answer. Output only the rewritten question.
    """
    msgs: List = []
    msgs.append(
        HumanMessage(
            content=(
                "You are rewriting a follow-up question into a standalone question for document retrieval.\n"
                "The documents are internal company policies.\n\n"
                "Rules:\n"
                "1) Output ONLY one standalone question. Do not add explanations or extra text.\n"
                "2) If the question uses references like 'it', 'that', 'this', or 'they', replace them with the specific policy topic implied by the conversation.\n"
                "3) Reuse concrete terms already mentioned in the conversation when possible.\n"
                "4) Do NOT invent new entities, facts, or policy topics.\n"
                "5) If the question is already standalone, return it unchanged.\n"
                "6) Keep the question concise (under 25 words).\n"
            )
        )
    )

    for m in history.messages[-8:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        msgs.append(HumanMessage(content=f"{role}: {m.content}"))

    msgs.append(HumanMessage(content=f"User: {question}\nStandalone question:"))

    rewritten = (llm.invoke(msgs).content or "").strip()
    if not rewritten or len(rewritten) > 800:
        return question
    return rewritten


def retrieve_for_question(
    vectordb,
    question: str,
    *,
    k: int,
    mmr: bool,
    fetch_k: int,
) -> Tuple[List, Optional[List[float]]]:
    """
    Returns (docs, scores_or_none).
    - Similarity: tries relevance scores (0..1), else returns None scores.
    - MMR: returns docs, scores=None (MMR typically doesn't expose per-doc scores).
    """
    if mmr:
        docs = vectordb.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=fetch_k,
        )
        return docs, None

    # Similarity with scores when available
    if hasattr(vectordb, "similarity_search_with_relevance_scores"):
        pairs = vectordb.similarity_search_with_relevance_scores(question, k=k)
        docs = [d for (d, s) in pairs]
        scores = [float(s) for (d, s) in pairs]
        return docs, scores

    # Fallback: similarity without scores
    docs = vectordb.similarity_search(question, k=k)
    return docs, None


def conversational_answer(
    *,
    llm,
    vectordb,
    question: str,
    history: ChatMessageHistory,
    k: int,
    mmr: bool,
    fetch_k: int,
):
    """
    Conversational RAG:
    1) rewrite follow-up -> standalone
    2) retrieve using standalone
    3) answer from retrieved docs
    4) refuse if off-topic AND insufficient evidence
    """
    if not history.messages:
        standalone = question
    else:
        standalone = contextualize_question(llm, history, question)

    docs, scores = retrieve_for_question(
        vectordb,
        standalone,
        k=k,
        mmr=mmr,
        fetch_k=fetch_k,
    )

    resp = build_structured_answer(
        llm=llm,
        question=standalone,
        retrieved_docs=docs,
        retrieved_scores=scores,  # may be None for MMR
    )

    # Extra policy-scope gate: only refuse harder when it's clearly off-topic
    if resp.insufficient_evidence and (not _looks_like_policy_question(question)):
        resp.answer = INSUFFICIENT_MSG
        resp.citations = []
        resp.insufficient_evidence = True
        resp.confidence = 0.0

    # Update conversation history with original user question and final answer
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=resp.answer))

    return resp, standalone, docs, scores
