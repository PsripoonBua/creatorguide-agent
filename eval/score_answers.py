"""Score saved answers (offline, deterministic). Writes answers_summary.json, answers_per_question.csv, failures.json."""
from __future__ import annotations

import csv
import json

from _common import RESULTS, meta, write_json
from check_questions import load_questions
from metrics import paired_bootstrap, score_answer, summarize_answers

ORDER = ["no_retrieval", "rag_bm25", "rag_ours", "agent"]
ERROR_RESULT = {"answer": "", "answerable": False, "citations": [], "stop_reason": "error", "usage": {}, "tool_calls": 0}


def load_system(system: str) -> dict[str, dict]:
    path = RESULTS / f"answers_{system}.jsonl"
    if not path.exists():
        return {}
    return {r["qid"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main(settings_meta: dict | None = None) -> None:
    questions = load_questions()
    summary, per_q, failures, rows_by_system = {}, [], [], {}
    for system in ORDER:
        saved = load_system(system)
        if not saved:
            continue
        rows = []
        for q in questions:
            rec = saved.get(q["qid"])
            result = (rec or {}).get("result") or ERROR_RESULT
            row = score_answer(q, result)
            row["system"], row["error"] = system, (rec or {}).get("error") or ("missing" if rec is None else "")
            rows.append(row)
            if row["correct"] == 0.0:
                failures.append({"system": system, "qid": q["qid"], "type": q["type"], "question": q["question"],
                                 "expected_key_facts": q["key_facts"], "answerable_expected": q["answerable"],
                                 "got_answer": result["answer"], "got_answerable": result["answerable"],
                                 "citations": result.get("citations", []), "stop_reason": result.get("stop_reason"),
                                 "error": row["error"]})
        rows_by_system[system] = rows
        per_q += rows
        summary[system] = {**summarize_answers(rows), "n_errors": sum(1 for r in rows if r["error"])}
    comparisons = {}
    for base in ("no_retrieval", "rag_bm25"):
        for system, rows in rows_by_system.items():
            if base in rows_by_system and system != base:
                comparisons[f"{system}_vs_{base}_accuracy_all"] = {
                    k: round(v, 4) for k, v in paired_bootstrap([r["correct"] for r in rows_by_system[base]],
                                                                 [r["correct"] for r in rows]).items()}
    if "rag_ours" in rows_by_system and "agent" in rows_by_system:
        comparisons["agent_vs_rag_ours_accuracy_all"] = {
            k: round(v, 4) for k, v in paired_bootstrap([r["correct"] for r in rows_by_system["rag_ours"]],
                                                         [r["correct"] for r in rows_by_system["agent"]]).items()}
    write_json("answers_summary.json", {"meta": meta(**(settings_meta or {})), "summary": summary,
                                        "paired_bootstrap": comparisons})
    write_json("failures.json", failures)
    if per_q:
        with (RESULTS / "answers_per_question.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_q[0]))
            w.writeheader()
            w.writerows(per_q)
    for system, s in summary.items():
        print(f"{system:13s} accuracy={s['accuracy_all']:.3f} answerable={s['accuracy_answerable']:.3f} "
              f"multi={s['accuracy_multi']:.3f} abstain_ok={s['abstain_correct_unanswerable']:.2f} errors={s['n_errors']}")


if __name__ == "__main__":
    main()
