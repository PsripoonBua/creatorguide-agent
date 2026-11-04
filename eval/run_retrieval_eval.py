"""Retrieval evaluation at document level on the answerable questions.

    python eval/run_retrieval_eval.py
Writes eval/results/retrieval.json and retrieval_per_question.csv. Systems that need the dense model are
skipped (and recorded as skipped, with the reason) if sentence-transformers is not installed.
"""
from __future__ import annotations

import csv

from _common import RESULTS, meta, try_build, write_json
from check_questions import load_questions
from metrics import paired_bootstrap, retrieval_row

from creatorguide.config import load_settings
from creatorguide.retrieval import doc_ranking

SYSTEMS = {  # name -> (retriever kind, remove_stopwords)
    "bm25": ("bm25", False),                 # baseline: naive default
    "bm25_stopwords": ("bm25", True),
    "dense": ("dense", False),
    "hybrid": ("hybrid", False),
    "hybrid_stopwords": ("hybrid", True),
}
DEPTH = 30  # chunks retrieved per query before collapsing to documents


def main() -> None:
    settings = load_settings()
    questions = [q for q in load_questions() if q["answerable"]]
    summary, per_q, skipped, rows_by_system = {}, [], {}, {}
    for name, (kind, stop) in SYSTEMS.items():
        retriever, reason = try_build(kind, settings, remove_stopwords=stop)
        if retriever is None:
            skipped[name] = reason
            print(f"skip {name}: {reason}")
            continue
        rows = []
        for q in questions:
            ranked = doc_ranking(retriever.search(q["question"], DEPTH))
            row = retrieval_row(ranked, set(q["gold_doc_ids"]))
            rows.append(row)
            per_q.append({"system": name, "qid": q["qid"], "type": q["type"], "top3_docs": " ".join(ranked[:3]),
                          **row})
        rows_by_system[name] = rows
        summary[name] = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}
        print(name, summary[name])
    comparisons = {}
    if "bm25" in rows_by_system:
        for name, rows in rows_by_system.items():
            if name != "bm25":
                comparisons[f"{name}_vs_bm25_mrr@10"] = {
                    k: round(v, 4) for k, v in paired_bootstrap([r["mrr@10"] for r in rows_by_system["bm25"]],
                                                                 [r["mrr@10"] for r in rows]).items()}
    write_json("retrieval.json", {"meta": meta(n_questions=len(questions), depth_chunks=DEPTH),
                                  "summary": summary, "paired_bootstrap": comparisons, "skipped": skipped})
    with (RESULTS / "retrieval_per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)


if __name__ == "__main__":
    main()
