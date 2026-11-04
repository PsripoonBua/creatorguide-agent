# Demo script (3-6 minutes)

Record the screen with narration. Run `uvicorn server.api:app` first; check `/health` shows `retrieval: hybrid`.

1. **0:00 Problem (20 s)** - creators need checkable answers to policy questions; wrong answers have real cost.
2. **0:20 Architecture (40 s)** - show README diagram: agent loop, three tools, validated JSON, class proxy.
3. **1:00 Success, single fact (45 s)** - ask "How long do I have to appeal a Community Guidelines strike?" Point at the streamed `search_docs` step, the answer, and the cited article link.
4. **1:45 Success, multi-step (60 s)** - ask "I have 650 subscribers and 3,200 watch hours... how many more of each do I need?" Show `search_docs` then `calculator`. Say why the calculator exists (small-model arithmetic).
5. **2:45 Honest failure (60 s)** - pick a real failure from `eval/results/failures.json` (the report lists them) or the unanswerable "average CPM" question if the agent wrongly answers it. Show it, explain the cause.
6. **3:45 Evaluation (60 s)** - open the report results table: baseline vs agent, the 30 labelled questions, one CI, one honest caveat (n = 30).
7. **4:45 Design choice + number (30 s)** - e.g. the stopword filter (MRR 0.893 -> 0.933, CI includes 0) or the min_score chosen from the sweep. Be ready for the same question live.

Live-defence prep: be able to explain `validate_final` (why citations must be in the evidence shown), the repeat-call guard, why RRF uses ranks, and how `key_facts` scoring works.
