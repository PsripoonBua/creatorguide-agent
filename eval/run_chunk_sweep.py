"""Chunk-size sweep with BM25 (+stopword filter) on word-based chunks. Runs offline, no model needed.

Words are not model tokens, so these sizes are NOT comparable to the 224-token dense index; the sweep only
answers 'which granularity helps lexical retrieval on this corpus and question set'.
"""
from __future__ import annotations

from _common import ROOT, meta, write_json
from check_questions import load_questions
from metrics import retrieval_row

from creatorguide.chunking import chunk_sections, chunk_words
from creatorguide.corpus import Corpus
from creatorguide.retrieval import BM25Retriever, doc_ranking, load_index

CONFIGS = [("words-60", 60, 15), ("words-100", 100, 25), ("words-150", 150, 25), ("words-250", 250, 50),
           ("words-400", 400, 50), ("sections", None, None)]


def evaluate(chunks: list[dict], questions: list[dict]) -> dict:
    retriever = BM25Retriever(chunks, remove_stopwords=True)
    rows = [retrieval_row(doc_ranking(retriever.search(q["question"], 30)), set(q["gold_doc_ids"])) for q in questions]
    return {"n_chunks": len(chunks), **{k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}}


def main() -> None:
    questions = [q for q in load_questions() if q["answerable"]]
    corpus = Corpus.load(ROOT / "data" / "docs.jsonl")
    out = {"committed_224_token_index": evaluate(load_index(ROOT / "data" / "index")[0], questions)}
    for name, size, overlap in CONFIGS:
        chunks = chunk_sections(corpus) if size is None else chunk_words(corpus, size, overlap)
        out[name] = evaluate(chunks, questions)
    for name, r in out.items():
        print(f"{name:28s} chunks={r['n_chunks']:4d} hit@3={r['hit@3']:.2f} recall@5={r['recall@5']:.2f} mrr@10={r['mrr@10']:.3f}")
    write_json("chunk_sweep.json", {"meta": meta(retriever="bm25+stopwords"), "results": out})


if __name__ == "__main__":
    main()
