"""Wires settings, corpus, retriever and model together. Used by the API server and the CLI."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

from .agent import Agent
from .config import ROOT, Settings
from .corpus import Corpus
from .llm import LLM, OpenAILLM
from .rag import answer_rag
from .retrieval import Retriever, build_retriever

log = logging.getLogger("creatorguide")


@dataclass
class Service:
    settings: Settings
    corpus: Corpus
    retriever: Retriever
    retrieval_used: str
    llm: LLM

    def ask(self, question: str, mode: str = "agent") -> dict[str, Any]:
        if mode == "rag":
            return answer_rag(self.llm, self.retriever, question, self.settings.top_k, self.settings.min_score,
                              system="rag_ours").to_dict()
        return Agent(self.llm, self.corpus, self.retriever, self.settings).run(question).to_dict()

    def stream(self, question: str, mode: str = "agent") -> Iterator[dict[str, Any]]:
        if mode == "rag":
            result = self.ask(question, "rag")
            yield {"type": "final", **{k: result[k] for k in ("answer", "answerable", "citations", "stop_reason",
                                                                "usage", "mode", "tool_calls", "retrieved_docs")},
                   "sources": self._sources(result["citations"])}
            return
        yield from Agent(self.llm, self.corpus, self.retriever, self.settings).stream(question)

    def _sources(self, citations: list[str]) -> list[dict[str, str]]:
        return Agent(self.llm, self.corpus, self.retriever, self.settings)._sources(citations)


def build_service(settings: Settings, llm: LLM | None = None) -> Service:
    corpus = Corpus.load(settings.docs_path)
    used = settings.retrieval
    try:
        retriever = build_retriever(settings.retrieval, settings.index_dir, embed_backend=settings.embed_backend,
                                    embed_model=settings.embed_model, remove_stopwords=True)
    except (ImportError, OSError) as exc:
        log.warning("dense retrieval unavailable (%s); falling back to BM25", exc)
        retriever = build_retriever("bm25", settings.index_dir, remove_stopwords=True)
        used = "bm25 (fallback)"
    llm = llm or OpenAILLM(settings.llm_model, settings.temperature, settings.max_tokens)
    return Service(settings, corpus, retriever, used, llm)
