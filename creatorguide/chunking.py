"""Word-based chunking used only for the chunk-size sweep (eval/run_chunk_sweep.py).

The committed dense index uses model-token chunks (scripts/build_index.py); this lighter chunker
lets BM25 experiments run with no model download. Words are not tokens, so sizes are not comparable.
"""
from __future__ import annotations

from typing import Any

from .corpus import Corpus


def chunk_words(corpus: Corpus, size: int, overlap: int = 0) -> list[dict[str, Any]]:
    if size < 16 or not 0 <= overlap < size:
        raise ValueError("size must be >= 16 and 0 <= overlap < size")
    chunks: list[dict[str, Any]] = []
    for doc_id, doc in corpus.docs.items():
        n = 0
        for heading, body in corpus.sections(doc_id):
            words = body.split()
            start = 0
            while start < len(words):
                piece = " ".join(words[start:start + size])
                n += 1
                chunks.append({"chunk_id": f"{doc_id}:{n:04d}", "doc_id": doc_id, "title": doc["title"],
                               "heading": heading, "category": doc["category"], "url": doc["url"],
                               "text": f"Title: {doc['title']}\nSection: {heading}\n\n{piece}"})
                if start + size >= len(words):
                    break
                start += size - overlap
    return chunks


def chunk_sections(corpus: Corpus) -> list[dict[str, Any]]:
    """One chunk per '## ' section (no size limit)."""
    out = []
    for doc_id, doc in corpus.docs.items():
        for n, (heading, body) in enumerate(corpus.sections(doc_id), 1):
            out.append({"chunk_id": f"{doc_id}:{n:04d}", "doc_id": doc_id, "title": doc["title"],
                        "heading": heading, "category": doc["category"], "url": doc["url"],
                        "text": f"Title: {doc['title']}\nSection: {heading}\n\n{body}"})
    return out
