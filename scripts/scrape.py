"""
Step 1 — Scrape YouTube Help articles (policy, enforcement, monetization).

Usage:
    pip install requests beautifulsoup4
    python scrape.py                  # target 60 docs (leaves buffer above 50)
    python scrape.py --target 80 --delay 1.5

Outputs (in data/):
    raw/<answer_id>.html   raw HTML as fetched (for reproducibility / re-parsing)
    docs.jsonl             one cleaned document per line (id, url, title, category, text, ...)
    sources.csv            source table: id, title, category, url, words, fetched_at, status
    rejected.csv           pages dropped (too short, duplicate, off-topic, error) and why
"""

import argparse
import csv
import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://support.google.com/youtube/answer/{}?hl=en"
ANSWER_RE = re.compile(r"/youtube/answer/(\d+)")
MIN_WORDS = 150  # pages shorter than this are flagged "too short" and skipped

# Seed article IDs collected from the three hub pages (5 Oct 2026):
#   Community Guidelines hub (9288567), Monetization policies hub (1311392),
#   Community Guidelines strike basics (2802032)
SEEDS = {
    "policy": [
        9288567, 2801973, 2801947, 9054257, 3399767, 9713446, 2801981, 2802002,
        9229980, 2801999, 2802245, 10072685, 2801964, 2802008, 9229472, 2801939,
        2802268, 9229611, 7667605, 10834785, 10835034, 13813322, 6345162,
        10767342, 2802027, 10774223, 9528076, 7671399, 12950271, 2797466,
    ],
    "enforcement": [
        2802032, 2814000, 2802168, 185111, 9890437, 9891124, 2802167,
        7650329, 6013276,
    ],
    "monetization": [
        1311392, 6162278, 12504220, 13195878, 72857, 12496325, 1727191, 72851,
        10285842, 12843009, 16271309, 9208564,
    ],
}
SEEDS["policy"].append(10008196)

# Manual review (5 Oct 2026, see AI_AUDIT.md): off-topic how-to pages, and crawled pages
# the keyword classifier put in the wrong category.
EXCLUDE = {11585688, 157177, 7336634}
CATEGORY_OVERRIDES = {17081657: "monetization", 6154218: "policy", 10726080: "policy"}
ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")

# Used to keep crawled (non-seed) pages on topic and to guess their category.
CATEGORY_KEYWORDS = {
    "monetization": ["monetiz", "partner program", "ypp", "adsense", "ads ", "advertiser",
                     "revenue", "earn", "membership", "super chat", "super thanks", "shopping"],
    "enforcement": ["strike", "terminat", "appeal", "warning", "removed", "suspend",
                    "restrict", "penalt", "age-restrict", "age restrict"],
    "policy": ["policy", "policies", "guideline", "prohibited", "not allowed", "misinformation",
               "harass", "hate", "spam", "violent", "copyright", "privacy"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (student NLP course project; RAG over YouTube Help policies)",
    "Accept-Language": "en-US,en;q=0.9",
}

OUT = Path(__file__).resolve().parent.parent / "data"
RAW = OUT / "raw"


def fetch(session, answer_id, retries=3):
    url = BASE.format(answer_id)
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.url, r.text
            if r.status_code in (404, 410):
                return url, None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return url, None


def parse(html):
    """Return (title, clean_text, linked_answer_ids)."""
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("h1")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    # Article body: try known Help Center containers, fall back to <article>/<main>.
    body = (soup.select_one("div.article-content-container")
            or soup.select_one("section.article-container")
            or soup.select_one("div.cc")
            or soup.find("article")
            or soup.find("main")
            or soup.body)

    links = set()
    if body is None:
        return title, "", links

    for a in body.find_all("a", href=True):
        m = ANSWER_RE.search(a["href"])
        if m:
            links.add(int(m.group(1)))

    for tag in body(["script", "style", "noscript", "svg", "button", "form", "nav", "footer"]):
        tag.decompose()
    # Drop the "Was this helpful?" / feedback widgets and breadcrumbs if present.
    for sel in [".article-survey", ".feedback", ".breadcrumbs", '[role="navigation"]']:
        for el in body.select(sel):
            el.decompose()

    # Keep block structure: one line per heading / paragraph / list item.
    lines = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "div"]):
        if el.name == "div" and el.find(["div", "h1", "h2", "h3", "h4", "p", "li", "td"]):
            continue
        if el.find(["p", "li"]):          # skip containers whose children we'll visit anyway
            continue
        t = el.get_text(" ", strip=True)
        if not t:
            continue
        if el.name in ("h2", "h3", "h4"):
            t = "## " + t
        elif el.name == "li":
            t = "- " + t
        lines.append(t)

    text = "\n".join(lines)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text, links


def guess_category(title, text):
    blob = (title + " " + text[:1500]).lower()
    scores = {c: sum(blob.count(k) for k in kws) for c, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60, help="stop after this many kept docs")
    ap.add_argument("--max-fetch", type=int, default=250, help="hard cap on page requests")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    queue = deque()
    seed_cat = {}
    for cat, ids in SEEDS.items():
        for i in ids:
            if i not in seed_cat:
                seed_cat[i] = cat
                queue.append(i)

    seen, kept, rejected, hashes = set(), [], [], {}
    fetched = 0

    while queue and len(kept) < args.target and fetched < args.max_fetch:
        aid = queue.popleft()
        if aid in seen:
            continue
        seen.add(aid)
        if aid in EXCLUDE:
            rejected.append({"id": aid, "url": BASE.format(aid), "reason": "off_topic (manual review, see AI_AUDIT.md)"})
            continue

        url, html = fetch(session, aid)
        fetched += 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        time.sleep(args.delay)

        if html is None:
            rejected.append({"id": aid, "url": url, "reason": "fetch_failed"})
            continue
        (RAW / f"{aid}.html").write_text(html, encoding="utf-8")

        title, text, links = parse(html)
        title, text = ZERO_WIDTH.sub("", title).strip(), ZERO_WIDTH.sub("", text)
        for l in links:
            if l not in seen:
                queue.append(l)

        words = len(text.split())
        if words < MIN_WORDS:
            rejected.append({"id": aid, "url": url, "reason": f"too_short ({words} words)"})
            continue

        h = hashlib.sha1(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()
        if h in hashes:
            rejected.append({"id": aid, "url": url, "reason": f"duplicate_of {hashes[h]}"})
            continue

        category = CATEGORY_OVERRIDES.get(aid) or seed_cat.get(aid) or guess_category(title, text)
        if category is None:
            rejected.append({"id": aid, "url": url, "reason": "off_topic"})
            continue

        hashes[h] = aid
        kept.append({
            "id": f"yt-{aid}",
            "answer_id": aid,
            "url": BASE.format(aid),
            "title": title,
            "category": category,
            "source": "seed" if aid in seed_cat else "crawled",
            "words": words,
            "fetched_at": now,
            "text": text,
        })
        print(f"[{len(kept):3d}] {category:12s} {words:5d}w  {title[:70]}")

    with open(OUT / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(OUT / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "category", "source", "url",
                                          "words", "fetched_at"])
        w.writeheader()
        for d in kept:
            w.writerow({k: d[k] for k in w.fieldnames})

    with open(OUT / "rejected.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "url", "reason"])
        w.writeheader()
        w.writerows(rejected)

    by_cat = {}
    for d in kept:
        by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
    print(f"\nKept {len(kept)} docs from {fetched} requests | by category: {by_cat}")
    print(f"Rejected {len(rejected)} (see data/rejected.csv)")
    if len(kept) < 50:
        print("WARNING: fewer than 50 docs — raise --max-fetch or add seeds.")


if __name__ == "__main__":
    main()
