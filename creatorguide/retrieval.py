"""BM25, dense and hybrid (RRF) retrieval over the shared chunk index.

Every retriever returns hits shaped like
    {"rank", "score", "bm25_score"?, "dense_score"?, chunk_id, doc_id, title, heading, url, text, ...}
and exposes `gate_key`: the score used for the relevance floor (`min_score`).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .corpus import read_jsonl

TOKEN = re.compile(r"[a-z0-9]+")


# Question words and function words only. Deliberately keeps "not", "no", "under", "over", "at", "least":
# in policy text those change the meaning ("not allowed", "under 18", "at least 18").
STOPWORDS = frozenset(
    "a an and are as be been but by can could did do does for from had has have how i if in into is it its "
    "me my of on or our should so than that the their them then there these they this to was we were what "
    "when where which who whom whose why will with would you your".split()
)


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Lowercase letter/digit runs. English corpus, so no stemming (keeps matches explainable)."""
    tokens = TOKEN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS] if remove_stopwords else tokens


def _require_query(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


def load_index(index_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_dir = Path(index_dir)
    chunks = read_jsonl(index_dir / "chunks.jsonl")
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return chunks, manifest


class Retriever(Protocol):
    gate_key: str
    chunks: list[dict[str, Any]]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class BM25Retriever:
    """Okapi BM25 (Lucene-style IDF, always positive). Written out so every term is explainable."""

    gate_key = "bm25_score"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75,
                 remove_stopwords: bool = False) -> None:
        self.chunks = chunks
        self.k1, self.b, self.remove_stopwords = k1, b, remove_stopwords
        self.tf = [Counter(tokenize(c["text"], remove_stopwords)) for c in chunks]
        self.dl = np.array([sum(t.values()) for t in self.tf], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if len(chunks) else 0.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, t in enumerate(self.tf):
            for w in t:
                self.postings.setdefault(w, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks), dtype=np.float64)
        for w in set(tokenize(query, self.remove_stopwords)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings[w]:
                f = self.tf[i][w]
                norm = 1 - self.b + self.b * self.dl[i] / self.avgdl
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        _require_query(query, top_k)
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        hits = []
        for i in order:
            if scores[i] <= 0:  # no shared word: not a retrieval result, just an arbitrary pick
                break
            hits.append({"rank": len(hits) + 1, "score": round(float(scores[i]), 6),
                         "bm25_score": round(float(scores[i]), 6), **self.chunks[i]})
        return hits


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbedder:
    """sentence-transformers on CPU. Must be the same model/revision the index was built with."""

    def __init__(self, model: str, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import kept lazy

        self.model = SentenceTransformer(model, device="cpu", revision=revision)

    def embed_query(self, text: str) -> np.ndarray:
        v = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


class ProxyEmbedder:
    """OpenAI-compatible /embeddings endpoint. Only valid if the index was built with the same model."""

    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model = client, model

    def embed_query(self, text: str) -> np.ndarray:
        v = np.asarray(self.client.embeddings.create(model=self.model, input=text).data[0].embedding,
                       dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)


class DenseRetriever:
    gate_key = "dense_score"

    def __init__(self, chunks: list[dict[str, Any]], embeddings: np.ndarray, embedder: Embedder) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks are not aligned")
        self.chunks, self.embeddings, self.embedder = chunks, embeddings, embedder

    def scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.embedder.embed_query(query)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        _require_query(query, top_k)
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [{"rank": r, "score": round(float(scores[i]), 6), "dense_score": round(float(scores[i]), 6),
                 **self.chunks[i]} for r, i in enumerate(order, 1)]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_lists 1/(k + rank). Uses ranks only, because cosine
    (about -1..1) and BM25 (0..~30) live on different scales. Deterministic tie-break."""
    if k < 0:
        raise ValueError("k must be >= 0")
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            best[cid] = min(best.get(cid, rank), rank)
    order = sorted(score, key=lambda c: (-score[c], best[c], c))
    return [(c, score[c]) for c in order]


class HybridRetriever:
    """BM25 + dense fused with RRF. Hits also carry both native scores (used for the gate)."""

    gate_key = "dense_score"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, candidates: int = 50) -> None:
        if [c["chunk_id"] for c in bm25.chunks] != [c["chunk_id"] for c in dense.chunks]:
            raise ValueError("BM25 and dense must index the same chunks")
        self.bm25, self.dense, self.rrf_k, self.candidates = bm25, dense, rrf_k, candidates
        self.chunks = bm25.chunks
        self._pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        _require_query(query, top_k)
        n = min(self.candidates, len(self.chunks))
        d_scores, b_scores = self.dense.scores(query), self.bm25.scores(query)
        d_order = [self.chunks[i]["chunk_id"] for i in np.argsort(-d_scores, kind="stable")[:n]]
        b_order = [self.chunks[i]["chunk_id"] for i in np.argsort(-b_scores, kind="stable")[:n] if b_scores[i] > 0]
        hits = []
        for rank, (cid, score) in enumerate(rrf_fuse([d_order, b_order], self.rrf_k)[:top_k], 1):
            i = self._pos[cid]
            hits.append({"rank": rank, "score": round(score, 6), "dense_score": round(float(d_scores[i]), 6),
                         "bm25_score": round(float(b_scores[i]), 6), **self.chunks[i]})
        return hits


def passes_gate(hits: list[dict[str, Any]], gate_key: str, min_score: float) -> bool:
    """Relevance floor: answer only if the best hit on the native score reaches min_score."""
    return bool(hits) and max(h.get(gate_key, 0.0) for h in hits) >= min_score


def build_retriever(kind: str, index_dir: Path, embedder: Embedder | None = None,
                    embed_backend: str = "local", embed_model: str | None = None,
                    remove_stopwords: bool = False) -> Retriever:
    chunks, manifest = load_index(index_dir)
    if kind == "bm25":
        return BM25Retriever(chunks, remove_stopwords=remove_stopwords)
    if embedder is None:
        if embed_backend == "proxy":
            embedder = ProxyEmbedder(embed_model or manifest["model"])
        else:
            embedder = LocalEmbedder(embed_model or manifest["model"], manifest.get("model_revision"))
    dense = DenseRetriever(chunks, np.load(Path(index_dir) / "embeddings.npy"), embedder)
    return dense if kind == "dense" else HybridRetriever(BM25Retriever(chunks, remove_stopwords=remove_stopwords), dense)


def doc_ranking(hits: list[dict[str, Any]]) -> list[str]:
    """Collapse chunk hits to a document ranking (first occurrence wins)."""
    seen: list[str] = []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.append(h["doc_id"])
    return seen
