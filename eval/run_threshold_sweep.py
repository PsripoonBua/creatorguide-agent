"""Choose `min_score` (the relevance floor) from data instead of guessing.

For each retriever, take the best gate score among the top-5 hits of every question, then sweep thresholds:
  TPR = answerable questions that pass the gate, TNR = unanswerable questions that are rejected.
Operating point: the threshold that rejects the most unanswerable questions while still keeping at least 90% of
the answerable ones (a wrongly refused real question is worse than an unnecessary attempt). The balanced-accuracy
optimum is reported too. With only 5 unanswerable questions this is a rough estimate; the report says so.
"""
from __future__ import annotations

from _common import meta, try_build, write_json
from check_questions import load_questions

from creatorguide.config import load_settings


MIN_TPR = 0.9


def sweep(scores_ans: list[float], scores_una: list[float]) -> dict:
    candidates = sorted({0.0, *scores_ans, *scores_una})
    table = []
    for t in candidates:
        tpr = sum(s >= t for s in scores_ans) / len(scores_ans)
        tnr = sum(s < t for s in scores_una) / len(scores_una)
        table.append({"threshold": round(t, 4), "tpr_answerable_kept": round(tpr, 4),
                      "tnr_unanswerable_rejected": round(tnr, 4), "balanced_accuracy": round((tpr + tnr) / 2, 4)})
    balanced = max(table, key=lambda r: (r["balanced_accuracy"], -r["threshold"]))
    safe = [r for r in table if r["tpr_answerable_kept"] >= MIN_TPR]
    best = max(safe, key=lambda r: (r["tnr_unanswerable_rejected"], -r["threshold"]))
    return {"table": table, "best": best, "best_balanced": balanced, "no_gate": table[0], "min_tpr": MIN_TPR}


def main() -> None:
    settings = load_settings()
    questions = load_questions()
    out, skipped = {}, {}
    for name, (kind, stop) in {"bm25": ("bm25", False), "bm25_stopwords": ("bm25", True),
                               "dense": ("dense", False), "hybrid": ("hybrid", False)}.items():
        retriever, reason = try_build(kind, settings, remove_stopwords=stop)
        if retriever is None:
            skipped[name] = reason
            continue
        scores = {True: [], False: []}
        for q in questions:
            hits = retriever.search(q["question"], 5)
            scores[q["answerable"]].append(max((h[retriever.gate_key] for h in hits), default=0.0))
        out[name] = {"gate_key": retriever.gate_key, **sweep(scores[True], scores[False]),
                     "answerable_scores": [round(s, 4) for s in sorted(scores[True])],
                     "unanswerable_scores": [round(s, 4) for s in sorted(scores[False])]}
        b = out[name]["best"]
        print(f"{name}: min_score={b['threshold']} keeps {b['tpr_answerable_kept']:.0%} of answerable, "
              f"rejects {b['tnr_unanswerable_rejected']:.0%} of unanswerable (no gate: rejects 0%); "
              f"balanced optimum {out[name]['best_balanced']['threshold']}")
    write_json("threshold_sweep.json", {"meta": meta(), "results": out, "skipped": skipped})


if __name__ == "__main__":
    main()
