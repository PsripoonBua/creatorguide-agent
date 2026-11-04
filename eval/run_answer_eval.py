"""Generate answers for all evaluation questions with each system, then score them.

    export OPENAI_API_KEY=... OPENAI_BASE_URL=...      # class proxy
    python eval/run_answer_eval.py                      # 4 systems x 30 questions
    python eval/run_answer_eval.py --retrieval bm25     # if the dense model is not installed
    python eval/run_answer_eval.py --systems agent --force

Systems
  no_retrieval  baseline: the model alone, no documents
  rag_bm25      naive default: BM25 (no stopword filter), top-3, no relevance floor, single shot
  rag_ours      single shot with our retriever (hybrid + stopword filter, top-5, MIN_SCORE)
  agent         the tool-calling agent (search_docs / read_doc / calculator) on our retriever
Raw answers are saved per system (answers_<system>.jsonl, resumable); scoring is a separate offline step
(score_answers.py) so metrics can be recomputed without calling the model again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from _common import RESULTS, ROOT, try_build
from check_questions import load_questions

from creatorguide.agent import Agent
from creatorguide.config import load_settings
from creatorguide.corpus import Corpus
from creatorguide.llm import OpenAILLM
from creatorguide.rag import answer_no_retrieval, answer_rag
from creatorguide.retrieval import BM25Retriever, load_index

ALL = ["no_retrieval", "rag_bm25", "rag_ours", "agent"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=",".join(ALL))
    ap.add_argument("--retrieval", choices=["hybrid", "dense", "bm25"], default=None,
                    help="retriever for rag_ours/agent (default: RETRIEVAL from env, 'hybrid')")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--force", action="store_true", help="regenerate even if a saved answer exists")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (and OPENAI_BASE_URL for the class proxy). See .env.example.")

    settings = load_settings()
    kind = args.retrieval or settings.retrieval
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(ALL)
    if unknown:
        sys.exit(f"unknown systems: {sorted(unknown)}")
    questions = load_questions()[: args.limit]
    corpus = Corpus.load(ROOT / "data" / "docs.jsonl")
    llm = OpenAILLM(settings.llm_model, settings.temperature, settings.max_tokens)

    chunks, _ = load_index(ROOT / "data" / "index")
    naive = BM25Retriever(chunks)
    ours = None
    if {"rag_ours", "agent"} & set(systems):
        ours, reason = try_build(kind, settings, remove_stopwords=True)
        if ours is None:
            sys.exit(f"cannot build the '{kind}' retriever ({reason}). Install requirements.txt or use --retrieval bm25.")

    runners = {
        "no_retrieval": lambda q: answer_no_retrieval(llm, q),
        "rag_bm25": lambda q: answer_rag(llm, naive, q, top_k=3, min_score=0.0, system="rag_bm25"),
        "rag_ours": lambda q: answer_rag(llm, ours, q, top_k=settings.top_k, min_score=settings.min_score, system="rag_ours"),
        "agent": lambda q: Agent(llm, corpus, ours, settings).run(q),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    for system in systems:
        path = RESULTS / f"answers_{system}.jsonl"
        done = {}
        if path.exists() and not args.force:
            done = {json.loads(l)["qid"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        with path.open("w", encoding="utf-8") as out:
            for q in questions:
                if q["qid"] in done and not done[q["qid"]].get("error"):
                    record = done[q["qid"]]
                else:
                    started = time.time()
                    try:
                        record = {"qid": q["qid"], "question": q["question"], "error": None,
                                  "result": runners[system](q["question"]).to_dict()}
                    except Exception as exc:  # noqa: BLE001 - recorded, scored as a failure, never hidden
                        record = {"qid": q["qid"], "question": q["question"], "error": f"{type(exc).__name__}: {exc}"[:300],
                                  "result": None}
                    record["seconds"] = round(time.time() - started, 2)
                    print(f"[{system}] {q['qid']} {'ERROR ' + record['error'] if record['error'] else 'ok'} ({record['seconds']}s)")
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
    import score_answers

    score_answers.main(settings_meta={"llm_model": settings.llm_model, "temperature": settings.temperature,
                                      "retrieval": kind, "min_score": settings.min_score, "top_k": settings.top_k,
                                      "agent_mode": settings.agent_mode, "max_steps": settings.max_steps})
    return 0


if __name__ == "__main__":
    sys.exit(main())
