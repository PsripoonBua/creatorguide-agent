PY ?= python

.PHONY: test check-data eval-retrieval eval-offline eval-answers report serve
test:            ## unit + smoke tests (no network, no API key)
	$(PY) -m unittest discover -s tests -t . -v
check-data:      ## verify the 30 labelled questions against the corpus
	$(PY) eval/check_questions.py
eval-retrieval:  ## retrieval metrics (dense/hybrid rows need sentence-transformers)
	$(PY) eval/run_retrieval_eval.py
eval-offline:    ## everything that needs no LLM: retrieval, min_score sweep, chunk sweep
	$(PY) eval/run_retrieval_eval.py && $(PY) eval/run_threshold_sweep.py && $(PY) eval/run_chunk_sweep.py
eval-answers:    ## answers from the LLM for all systems, then scoring (needs OPENAI_API_KEY/OPENAI_BASE_URL)
	$(PY) eval/run_answer_eval.py
report:          ## rebuild report/CreatorGuide_Report.pdf from eval/results
	$(PY) report/build_report.py
serve:           ## API + streaming UI on http://127.0.0.1:8000
	uvicorn server.api:app --reload
