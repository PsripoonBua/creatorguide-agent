# Data

- `docs.jsonl` - 62 YouTube Help articles (id, url, title, category, text, fetched_at). Snapshot 2026-10-05, sha256 recorded in `index/manifest.json` and checked by `tests/test_basics.py`.
- `sources.csv` - source table (id, title, category, url, words, fetched_at).
- `index/` - 686 chunks (`chunks.jsonl`, 224 tokens, 32 overlap), L2-normalised `all-MiniLM-L6-v2` embeddings (`embeddings.npy`, pinned revision in `manifest.json`).
- Re-fetch: `pip install -r requirements-scrape.txt && python scripts/scrape.py`. Re-embed: `python scripts/build_index.py build`.
