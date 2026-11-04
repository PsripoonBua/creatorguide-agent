# CreatorGuide Agent

A grounded, tool-using assistant that answers YouTube-creator policy questions (strikes, appeals, copyright, Partner Program thresholds) **only from 62 YouTube Help articles**, cites the passages it used, does arithmetic with a safe calculator, and says "not in the articles" when it cannot answer.

Capstone for the LLM-application course. The model is called through the class proxy with the OpenAI SDK (`OpenAI()`, model `gemma-4-E4B-it`).

## Course capability families covered

| # | Family | Where |
|---|---|---|
| 1 | Structured output + schema validation | `creatorguide/schema.py`, `structured.py` (validate, repair up to 2x, else abstain) |
| 2 | Retrieval / RAG with citations | `creatorguide/retrieval.py` (BM25, MiniLM dense, RRF hybrid), `rag.py` |
| 3 | **Retrieval + answer evaluation** | `eval/` (30 labelled questions, metrics, baselines, sweeps, bootstrap CIs) |
| 4 | **Agent loop with tools** | `creatorguide/agent.py`, `tools.py` (`search_docs`, `read_doc`, `calculator`) |
| 6 | Served interface | `server/api.py` (FastAPI JSON + SSE stream), `server/static/index.html` |

## Architecture

```
Browser (EventSource) -> FastAPI /api/ask/stream -> Agent loop -> LLM (class proxy, gemma-4-E4B-it)
                                                     |-> search_docs -> hybrid retriever (BM25 + MiniLM, RRF, min_score floor)
                                                     |-> read_doc    -> corpus (allow-listed doc ids)
                                                     |-> calculator  -> AST arithmetic (no eval)
Final answer = JSON {answer, answerable, citations}, validated; citations must be chunks the model was shown.
```

Layers: `config` -> `llm` -> `corpus`/`retrieval` -> `tools` -> `agent`/`rag` -> `service` -> `server`/`cli`. Nothing below `service` imports FastAPI; everything below `llm` is testable with a scripted fake model.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # fill OPENAI_API_KEY and OPENAI_BASE_URL (class proxy)
python -m creatorguide.cli "How long do I have to appeal a Community Guidelines strike?"
uvicorn server.api:app --reload                         # open http://127.0.0.1:8000/
```

Keys are read from the environment only (`.env` is git-ignored). Without `sentence-transformers` the server falls back to BM25 and `/health` says so.

## Reproduce the evaluation

```bash
make test             # 41 offline tests (no network, no key)
make check-data       # every evidence quote must occur verbatim in its article
make eval-offline     # retrieval + min_score sweep + chunk sweep (dense rows need sentence-transformers)
make eval-answers     # 4 systems x 30 questions through the proxy, then scoring
make report           # rebuilds report/CreatorGuide_Report.pdf from eval/results
```

Systems compared: `no_retrieval` (model alone), `rag_bm25` (naive default: BM25, top-3, no floor), `rag_ours` (hybrid + stopword filter + floor), `agent`. Metrics: Hit@k / Recall@k / MRR (documents), answer accuracy by deterministic key-fact matching, grounded rate, abstention accuracy, cost; paired bootstrap 95% CIs. No LLM judge (see report, limitations).

### Results (30 questions: 20 single-fact, 5 multi-step, 5 unanswerable; model `gemma-4-E4B-it`, temperature 0)

> **Evaluation history.** After a first full run I reviewed every failure and revised the question set once: s02 and m02 were reworded because they used a phrase ("early-access tier") that the articles never use, the key facts of s05 were widened (a correct answer, "two additional weeks", had been marked wrong), and the over-permissive variant "no" was replaced in s15 and m05. All systems were then re-run on all 30 questions; first-run results are kept in `eval/results_v1/`. Because the revisions were made after seeing results on the same set, the numbers below are somewhat optimistic compared with a fresh held-out set.

**Retrieval** (document level, 25 answerable questions, 224-token chunks):

| System | Hit@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| bm25 (baseline: naive default) | 0.880 | 0.980 | 0.933 |
| bm25 + stopword filter | 0.920 | 0.960 | 0.953 |
| dense (MiniLM) | 0.800 | 0.940 | 0.885 |
| hybrid (BM25 + dense, RRF) | 0.920 | 0.980 | 0.960 |
| hybrid + stopword filter (used by `rag_ours` and `agent`) | 0.880 | 0.980 | 0.940 |

Every difference from the BM25 baseline has a paired-bootstrap 95% CI that includes 0 (e.g. hybrid +0.027 MRR, CI [-0.060, +0.120]), so with 25 questions no retrieval variant is established as better; dense alone is the weakest, and the stopword filter does not help once BM25 is fused with dense. A raw BM25 score is a poor relevance gate (keeping 96% of answerable questions rejects only 40% of unanswerable ones), while the dense cosine separates the two groups well.

**Relevance floor.** `min_score = 0.5233` (best dense cosine among the top-5 hits) keeps 25/25 answerable questions and rejects 5/5 unanswerable ones. This perfect separation was chosen and measured on the same 30 questions, so treat it as optimistic; with only 5 unanswerable questions it is a rough estimate. Set `MIN_SCORE=0.5233` in `.env` to reproduce.

**Answers:**

| System | Accuracy (30) | Single (20) | Multi (5) | Abstain ok (5) | Grounded | False abstain | LLM calls / q | Tokens / q |
|---|---|---|---|---|---|---|---|---|
| no_retrieval (baseline 1) | 30% | 15% | 40% | 80% | 0% | 32% | 1.0 | 224 |
| rag_bm25 (naive default, baseline 2) | 70% | 70% | 40% | 100% | 60% | 20% | 1.0 | 873 |
| rag_ours (hybrid + stopwords + floor, single shot) | 93% | 100% | 60% | 100% | 92% | 0% | 0.8 | 1,023 |
| agent (tools) | 90% | 95% | 60% | 100% | 84% | 8% | 2.7 | 2,809 |

Paired-bootstrap 95% CIs on overall accuracy (difference in percentage points):

| Comparison | Difference | 95% CI |
|---|---|---|
| rag_ours vs no_retrieval | +63.3 | [+46.7, +80.0] |
| agent vs no_retrieval | +60.0 | [+43.3, +76.7] |
| rag_bm25 vs no_retrieval | +40.0 | [+16.7, +60.0] |
| rag_ours vs rag_bm25 | +23.3 | [+10.0, +40.0] |
| agent vs rag_bm25 | +20.0 | [0.0, +40.0] |
| agent vs rag_ours | -3.3 | [-13.3, +6.7] |

What the numbers support: retrieving documents clearly helps (every retrieval system beats `no_retrieval` with intervals far from 0), and `rag_ours` beats the naive default `rag_bm25` by 23 points with an interval that excludes 0. That comparison bundles several changes (hybrid retrieval, stopword filter, top-5 instead of top-3, relevance floor), and the retrieval table shows the retrieval changes alone are small, so I have not isolated which change causes the gain. What they do **not** support: that the agent is better. It answers one question fewer correctly than single-shot RAG (an indistinguishable difference), ties it on the 5 multi-step questions (3/5 each, so the calculator shows no accuracy benefit on these simple sums), and uses about 2.7x the tokens and 2.7 vs 0.8 model calls per question. Its value here is transparency (every tool call is visible in the UI) rather than accuracy. The floor's measured effect is fewer model calls (5 of 30 questions answered "not in the articles" without calling the model); its accuracy benefit is not demonstrated, because `rag_bm25` also abstains correctly on all 5 unanswerable questions without any floor.

## Safety and correctness choices

- **Allow-listed tools**, arguments validated (types, length, range, unknown keys rejected); unknown tools are refused, never dispatched.
- **Calculator** parses an AST; only numbers and arithmetic. No `eval`, names, attributes or calls other than `abs/min/max/round/ceil/floor`.
- **Untrusted text**: retrieved passages and tool results are framed as quoted data in every prompt; citations outside the evidence shown are rejected.
- **Stop conditions**: valid answer, `MAX_STEPS` tool rounds (then a forced final answer), repeated identical calls blocked, 2 repairs then abstain.
- `temperature=0` for everything evaluated; tool results capped at 5,000 characters and always valid JSON.
- `AGENT_MODE=auto` uses native function calling and falls back to a JSON action protocol if the endpoint rejects `tools=`.

## Layout

```
creatorguide/   core package (config, llm, corpus, retrieval, schema, tools, agent, rag, service, cli)
server/         FastAPI app + static SSE page
eval/           questions.jsonl, metrics, run_*.py, score_answers.py, results/
report/         build_report.py -> CreatorGuide_Report.pdf
scripts/        scrape.py (re-fetch corpus), build_index.py (re-embed)
data/           docs.jsonl (62 articles), sources.csv, index/ (686 chunks + MiniLM embeddings)
tests/          unittest suite with a scripted fake LLM
docs/           DEMO_SCRIPT.md
```

## Provenance and attribution

- The corpus (YouTube Help articles from support.google.com, fetched 5 and 12 Oct 2026) and the chunk index were first built for an earlier course project by the author's team and are reused here **only as a document corpus**; the 30 evaluation questions, all code in `creatorguide/`, `server/`, `eval/` and `report/` are new for this capstone. Article text is Google's; it is included for coursework and carries source URLs in `data/sources.csv`.
- Developed with AI assistance (Claude). The author is expected to be able to explain every design choice and evaluation number in the demo.

## Limitations

30 questions, one annotator, revised once after seeing first-run results (see Evaluation history); the corpus is a dated snapshot that already announces rule changes for 1 Feb 2027; a 4B model can ignore tools or format (validation + abstention contain this at extra cost); prompt-injection is mitigated, not eliminated; the API has no auth or rate limiting. See the report for failure cases.
