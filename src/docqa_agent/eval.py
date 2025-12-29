from dataclasses import dataclass
from typing import List, Optional

from langchain_huggingface import ChatHuggingFace

from docqa_agent.structured_rag import build_llm, build_llm_hf, build_structured_answer, INSUFFICIENT_MSG
from docqa_agent.vectorstore import (
    build_embeddings,
    build_embeddings_hf,
    build_or_load_chroma,
    similarity_search_with_scores,
)
from docqa_agent.schema import QAResponse


@dataclass
class EvalCase:
    question: str
    expect_answerable: bool


@dataclass
class EvalResult:
    question: str
    passed: bool
    insufficient_evidence: bool
    num_citations: int
    confidence: float
    answer_preview: str
    reason: Optional[str] = None


def _is_refusal(resp: QAResponse) -> bool:
    return resp.insufficient_evidence or resp.answer.strip() == INSUFFICIENT_MSG


def run_evaluation(index_dir: str, collection_name: str, k: int = 6, embedding: str = "google", llm_model: str = "google") -> List[EvalResult]:
    cases = [
        # Tune these to match your actual policy docs; keep 5 for the assignment requirement.
        EvalCase("What are the leave policies?", expect_answerable=True),
        EvalCase("How does sick leave work and what documentation is required?", expect_answerable=True),
        EvalCase("Is annual leave payout allowed on termination?", expect_answerable=True),
        EvalCase("What is the capital of Japan?", expect_answerable=False),
        EvalCase("What is the CEO's favorite color?", expect_answerable=False),
    ]

    if embedding == "google":
        embeddings = build_embeddings()
    elif embedding == "hf":
        embeddings = build_embeddings_hf()
    # embeddings = build_embeddings()
    vectordb = build_or_load_chroma(
        persist_dir=index_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    if llm_model == "google":
        llm = build_llm()
    elif llm_model == "hf":
        llm = ChatHuggingFace(llm = build_llm_hf())
    # llm = build_llm()
    results: List[EvalResult] = []

    for case in cases:
        scored = similarity_search_with_scores(vectordb, case.question, k=k)
        docs = [d for (d, s) in scored]
        scores = [float(s) for (d, s) in scored]

        resp = build_structured_answer(
            llm=llm,
            question=case.question,
            retrieved_docs=docs,
            retrieved_scores=scores,
        )

        num_citations = len(resp.citations)
        refusal = _is_refusal(resp)

        passed = True
        reason = None

        if case.expect_answerable:
            # Must not refuse, must have citations
            if refusal:
                passed = False
                reason = "Expected answerable, but got refusal/insufficient evidence."
            elif num_citations == 0:
                passed = False
                reason = "Expected citations for an answerable question, but citations list is empty."
        else:
            # Must refuse
            if not refusal:
                passed = False
                reason = "Expected refusal/insufficient evidence, but got an answer."
            elif resp.answer.strip() != INSUFFICIENT_MSG:
                passed = False
                reason = "Refusal text mismatch (must be exact)."

        preview = (resp.answer or "").strip().replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:140] + "..."

        results.append(
            EvalResult(
                question=case.question,
                passed=passed,
                insufficient_evidence=resp.insufficient_evidence,
                num_citations=num_citations,
                confidence=float(resp.confidence),
                answer_preview=preview,
                reason=reason,
            )
        )

    return results


def print_report(results: List[EvalResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    print("\n=== Evaluation Report ===")
    print(f"Passed: {passed}/{total}\n")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] Q: {r.question}")
        print(f"  insufficient_evidence: {r.insufficient_evidence}")
        print(f"  citations: {r.num_citations}")
        print(f"  confidence: {r.confidence:.2f}")
        print(f"  answer: {r.answer_preview}")
        if r.reason:
            print(f"  reason: {r.reason}")
        print("")


def main(index_dir: str, collection_name: str, k: int = 6, embedding: str = "google", llm_model: str = "google") -> int:
    results = run_evaluation(index_dir=index_dir, collection_name=collection_name, k=k, embedding=embedding, llm_model=llm_model)
    print_report(results)

    all_pass = all(r.passed for r in results)
    return 0 if all_pass else 1
