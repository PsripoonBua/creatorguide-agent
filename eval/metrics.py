"""Metrics for retrieval and for answers. Pure functions, unit-tested, no I/O."""
from __future__ import annotations

import random
import re
from typing import Any, Sequence


# ---------------------------------------------------------------- retrieval (document level)
def hit_at_k(ranked_docs: Sequence[str], gold: set[str], k: int) -> float:
    return float(any(d in gold for d in ranked_docs[:k]))


def recall_at_k(ranked_docs: Sequence[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked_docs[:k])) / len(gold) if gold else 0.0


def reciprocal_rank(ranked_docs: Sequence[str], gold: set[str], k: int = 10) -> float:
    for rank, d in enumerate(ranked_docs[:k], 1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def retrieval_row(ranked_docs: Sequence[str], gold: set[str]) -> dict[str, float]:
    return {"hit@1": hit_at_k(ranked_docs, gold, 1), "hit@3": hit_at_k(ranked_docs, gold, 3),
            "hit@5": hit_at_k(ranked_docs, gold, 5), "recall@3": recall_at_k(ranked_docs, gold, 3),
            "recall@5": recall_at_k(ranked_docs, gold, 5), "mrr@10": reciprocal_rank(ranked_docs, gold, 10)}


# ---------------------------------------------------------------- answers
def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)       # 3,000 -> 3000
    return re.sub(r"\s+", " ", text)


def _contains(haystack: str, needle: str) -> bool:
    needle = normalize(needle).strip()
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def key_fact_groups(answer: str, key_facts: list[list[str]]) -> list[bool]:
    """Each group is a list of acceptable variants; a group is matched if ANY variant appears as a whole token/phrase."""
    text = normalize(answer)
    return [any(_contains(text, v) for v in group) for group in key_facts]


def score_answer(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Deterministic per-question scoring. `result` is Result.to_dict()-shaped."""
    answerable_pred = bool(result["answerable"])
    row: dict[str, Any] = {"qid": question["qid"], "type": question["type"], "pred_answerable": answerable_pred,
                           "stop_reason": result.get("stop_reason", ""), "tool_calls": result.get("tool_calls", 0),
                           "llm_calls": result.get("usage", {}).get("llm_calls", 0),
                           "tokens": result.get("usage", {}).get("prompt_tokens", 0)
                           + result.get("usage", {}).get("completion_tokens", 0)}
    if not question["answerable"]:
        row.update(correct=float(not answerable_pred), kf_recall=None, grounded=None, citation_precision=None)
        return row
    groups = key_fact_groups(result["answer"], question["key_facts"]) if answerable_pred else [False] * len(question["key_facts"])
    gold = set(question["gold_doc_ids"])
    cited = {c.split(":", 1)[0] for c in result.get("citations", [])}
    correct = answerable_pred and all(groups)
    row.update(correct=float(correct), kf_recall=sum(groups) / len(groups),
               citation_precision=(len(cited & gold) / len(cited)) if cited else None,
               grounded=float(correct and bool(cited & gold)))
    return row


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    out = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
           "accuracy_all": _mean([r["correct"] for r in rows]),
           "accuracy_answerable": _mean([r["correct"] for r in ans]),
           "accuracy_single": _mean([r["correct"] for r in ans if r["type"] == "single"]),
           "accuracy_multi": _mean([r["correct"] for r in ans if r["type"] == "multi"]),
           "key_fact_recall": _mean([r["kf_recall"] for r in ans]),
           "grounded_rate": _mean([r["grounded"] for r in ans]),
           "citation_precision": _mean([r["citation_precision"] for r in ans]),
           "abstain_correct_unanswerable": _mean([r["correct"] for r in una]),
           "false_abstain_rate": _mean([float(not r["pred_answerable"]) for r in ans]),
           "mean_llm_calls": _mean([r["llm_calls"] for r in rows]),
           "mean_tool_calls": _mean([r["tool_calls"] for r in rows]),
           "mean_tokens": _mean([r["tokens"] for r in rows])}
    return out


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 42) -> dict[str, float]:
    """Mean of (b - a) with a 95% percentile CI, resampling questions in pairs. n is tiny, so read the CI."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty lists")
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return {"mean_diff": sum(diffs) / len(diffs), "ci_low": means[int(0.025 * n)], "ci_high": means[int(0.975 * n) - 1]}
