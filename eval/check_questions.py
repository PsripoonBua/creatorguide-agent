"""Validate the evaluation set: schema, doc ids, and that every evidence quote really occurs in its article.

    python eval/check_questions.py            # exits non-zero on any problem
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from creatorguide.corpus import Corpus, read_jsonl  # noqa: E402

TYPES = {"single", "multi", "unanswerable"}


def norm(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def load_questions(path: Path | None = None) -> list[dict]:
    return read_jsonl(path or ROOT / "eval" / "questions.jsonl")


def check(path: Path | None = None) -> list[str]:
    corpus = Corpus.load(ROOT / "data" / "docs.jsonl")
    texts = {d: norm(doc["text"]) for d, doc in corpus.docs.items()}
    problems: list[str] = []
    questions = load_questions(path)
    ids = Counter(q["qid"] for q in questions)
    problems += [f"duplicate qid {k}" for k, v in ids.items() if v > 1]
    for q in questions:
        qid = q["qid"]
        if q.get("type") not in TYPES:
            problems.append(f"{qid}: bad type")
        if not q.get("question", "").strip():
            problems.append(f"{qid}: empty question")
        if q["answerable"] != (q["type"] != "unanswerable"):
            problems.append(f"{qid}: answerable flag contradicts type")
        if q["answerable"]:
            if not q["gold_doc_ids"] or not q["key_facts"] or not q["evidence"]:
                problems.append(f"{qid}: answerable question needs gold_doc_ids, key_facts and evidence")
            for g in q["gold_doc_ids"]:
                if g not in corpus:
                    problems.append(f"{qid}: unknown gold doc {g}")
            for e in q["evidence"]:
                if e["doc_id"] not in texts:
                    problems.append(f"{qid}: evidence doc {e['doc_id']} unknown")
                elif norm(e["quote"]) not in texts[e["doc_id"]]:
                    problems.append(f"{qid}: quote not found verbatim in {e['doc_id']}: {e['quote']!r}")
                elif e["doc_id"] not in q["gold_doc_ids"]:
                    problems.append(f"{qid}: evidence doc {e['doc_id']} missing from gold_doc_ids")
            for group in q["key_facts"]:
                if not group or not all(isinstance(v, str) and v.strip() for v in group):
                    problems.append(f"{qid}: malformed key_facts group {group}")
        elif q["gold_doc_ids"] or q["key_facts"] or q["evidence"]:
            problems.append(f"{qid}: unanswerable question must have empty gold/evidence/key_facts")
    return problems


def main() -> int:
    questions = load_questions()
    problems = check()
    print(f"{len(questions)} questions: {dict(Counter(q['type'] for q in questions))}")
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
