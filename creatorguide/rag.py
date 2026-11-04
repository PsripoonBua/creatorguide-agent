"""Single-shot baselines: no retrieval, and plain retrieve-then-generate (no tools, no loop)."""
from __future__ import annotations

from .llm import LLM
from .prompts import NO_RETRIEVAL_SYSTEM, RAG_SYSTEM, format_evidence
from .retrieval import Retriever, doc_ranking, passes_gate
from .schema import ValidationError
from .structured import Result, fallback, generate_final


def _check_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("question must be a non-empty string")
    if len(question) > 600:
        raise ValidationError("question is longer than 600 characters")
    return question.strip()


def answer_no_retrieval(llm: LLM, question: str) -> Result:
    question = _check_question(question)
    usage: dict[str, int] = {}
    messages = [{"role": "system", "content": NO_RETRIEVAL_SYSTEM}, {"role": "user", "content": question}]
    final, error, _ = generate_final(llm, messages, allowed_ids=set(), usage=usage, require_citations=False)
    if final is None:
        return fallback("no_retrieval", "invalid_output", usage)
    final.citations = []
    return Result(final=final, stop_reason="final", usage=usage, system="no_retrieval")


def answer_rag(llm: LLM, retriever: Retriever, question: str, top_k: int = 5, min_score: float = 0.0,
               system: str = "rag") -> Result:
    question = _check_question(question)
    usage: dict[str, int] = {}
    hits = retriever.search(question, top_k)
    docs = doc_ranking(hits)
    if not passes_gate(hits, retriever.gate_key, min_score):
        return fallback(system, "below_min_score", usage, retrieved_docs=docs)  # no LLM call at all
    allowed = {h["chunk_id"] for h in hits} | {h["doc_id"] for h in hits}
    messages = [{"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Evidence:\n{format_evidence(hits)}\n\nQuestion: {question}"}]
    final, error, _ = generate_final(llm, messages, allowed_ids=allowed, usage=usage)
    if final is None:
        return fallback(system, "invalid_output", usage, retrieved_docs=docs)
    return Result(final=final, stop_reason="final", usage=usage, system=system, retrieved_docs=docs)
