"""Build report/CreatorGuide_Report.pdf from the saved evaluation results.

    python report/build_report.py --author "Your Name"

Every number in the PDF is read from eval/results/*.json|csv. Sections whose results do not exist yet
(dense/hybrid retrieval, LLM answers) say so explicitly instead of showing placeholders or invented figures.
Optional: put your own observations in report/manual_notes.md and they are appended to the analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "CreatorGuide_Report.pdf"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
ACCENT = colors.HexColor("#c4302b")


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------ figures
def box(ax, x, y, w, h, text, fc="#f3f4f6", ec="#6b7280", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.3))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.2, 2.5, 1.7, 1.0, "Browser UI\n(SSE stream)", "#fde8e7", "#c4302b", bold=True)
    box(ax, 2.6, 2.5, 1.9, 1.0, "FastAPI\n/api/ask\n/api/ask/stream", "#e0ecff", "#2563eb", bold=True)
    box(ax, 5.0, 2.2, 2.7, 1.6, "Agent loop\nstep limit, repeat guard,\nvalidate + repair", "#e7f6ec", "#16a34a", fs=7.5, bold=True)
    box(ax, 8.4, 4.3, 3.3, 1.2, "search_docs\nBM25 + MiniLM dense + RRF\n(min_score floor)", "#fff7e0", "#d97706")
    box(ax, 8.4, 2.6, 3.3, 1.0, "read_doc\n(allow-listed doc ids)", "#fff7e0", "#d97706")
    box(ax, 8.4, 1.0, 3.3, 1.0, "calculator\n(AST arithmetic, no eval)", "#fff7e0", "#d97706")
    box(ax, 5.2, 0.2, 2.4, 1.1, "gemma-4-E4B-it\nvia class proxy (OpenAI SDK)", "#f3e8ff", "#7c3aed")
    box(ax, 0.2, 4.4, 4.3, 1.2, "Corpus: 62 YouTube Help articles\n686 chunks (224 tokens, 32 overlap)", "#f3f4f6", "#6b7280")
    box(ax, 0.2, 0.2, 4.3, 1.3, "Structured output  {answer, answerable, citations}\nschema-validated, citations must be\nchunks the model was actually shown", "#e7f6ec", "#16a34a", fs=8)
    for a in [(1.9, 3.0, 2.6, 3.0), (4.5, 3.0, 5.2, 3.0), (7.6, 3.4, 8.4, 4.7), (7.6, 3.0, 8.4, 3.1), (7.6, 2.6, 8.4, 1.6),
              (6.4, 2.2, 6.4, 1.3), (8.4, 4.9, 4.5, 5.0), (5.2, 0.8, 4.5, 0.85)]:
        arrow(ax, *a)
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / "architecture.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def fig_retrieval(summary):
    names = list(summary)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xs = range(len(names)); w = 0.27
    for i, (metric, color) in enumerate([("hit@1", "#93c5fd"), ("recall@5", "#2563eb"), ("mrr@10", "#c4302b")]):
        vals = [summary[n][metric] for n in names]
        bars = ax.bar([x + (i - 1) * w for x in xs], vals, w, label=metric, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("score (25 answerable questions)", fontsize=8); ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    p = FIG / "retrieval.png"; fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def fig_gate(entry, name):
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    for y, key, color, label in [(1, "answerable_scores", "#16a34a", "answerable (25)"),
                                 (0, "unanswerable_scores", "#c4302b", "unanswerable (5)")]:
        ax.scatter(entry[key], [y] * len(entry[key]), s=38, color=color, alpha=0.75, label=label)
    ax.axvline(entry["best"]["threshold"], color="#374151", ls="--", lw=1)
    ax.text(entry["best"]["threshold"], 1.32, f" chosen min_score = {entry['best']['threshold']}", fontsize=7.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["unanswerable", "answerable"], fontsize=8); ax.set_ylim(-0.4, 1.6)
    ax.set_xlabel(f"best {entry['gate_key']} among the top-5 hits ({name})", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    p = FIG / "gate.png"; fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def fig_answers(summary):
    names = list(summary)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xs = range(len(names)); w = 0.2
    for i, (metric, color) in enumerate([("accuracy_all", "#c4302b"), ("accuracy_single", "#93c5fd"),
                                         ("accuracy_multi", "#2563eb"), ("abstain_correct_unanswerable", "#16a34a")]):
        vals = [summary[n][metric] or 0 for n in names]
        bars = ax.bar([x + (i - 1.5) * w for x in xs], vals, w, label=metric.replace("_", " "), color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15)); ax.spines[["top", "right"]].set_visible(False)
    p = FIG / "answers.png"; fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


# ------------------------------------------------------------------ pdf helpers
def styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.4, alignment=TA_LEFT, spaceAfter=5)
    return {"body": base,
            "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14.5, textColor=ACCENT, spaceBefore=10, spaceAfter=5),
            "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=7, spaceAfter=3),
            "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=ACCENT, spaceAfter=2, alignment=TA_LEFT),
            "sub": ParagraphStyle("s", parent=base, fontSize=10.5, textColor=colors.HexColor("#4b5563")),
            "small": ParagraphStyle("sm", parent=base, fontSize=8.2, leading=11, textColor=colors.HexColor("#4b5563")),
            "cell": ParagraphStyle("c", parent=base, fontSize=8, leading=10, spaceAfter=0),
            "warn": ParagraphStyle("w", parent=base, backColor=colors.HexColor("#fff4e5"), borderColor=colors.HexColor("#f5a524"),
                                   borderWidth=0.8, borderPadding=5, spaceBefore=4, spaceAfter=8)}


def table(rows, widths, st, header=True, font=8):
    data = [[Paragraph(esc(c), st["cell"]) if isinstance(c, str) else c for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("FONTSIZE", (0, 0), (-1, -1), font)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6"))]
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)



def compare_sentence(s):
    """Describe agent vs single-shot RAG strictly from the numbers (no assumed advantage)."""
    a, r = s["agent"], s["rag_ours"]
    d_all = round((a["accuracy_all"] - r["accuracy_all"]) * a["n"])
    d_multi = round((a["accuracy_multi"] - r["accuracy_multi"]) * 5)
    if d_all > 0:
        head = f"The agent answers {d_all} more question(s) correctly than single-shot RAG overall"
    elif d_all < 0:
        head = f"The agent answers {-d_all} fewer question(s) correctly than single-shot RAG overall, so tool use did not improve accuracy here"
    else:
        head = "The agent and single-shot RAG answer the same number of questions correctly"
    if d_multi > 0:
        multi = f"; on the 5 multi-step questions it is ahead by {d_multi}"
    elif d_multi < 0:
        multi = f"; on the 5 multi-step questions it is behind by {-d_multi}"
    else:
        multi = "; on the 5 multi-step questions the two tie, so the calculator tool shows no accuracy benefit on this set"
    cost = ""
    if r.get("mean_tokens"):
        cost = f". The agent uses {a['mean_tokens'] / r['mean_tokens']:.1f}x the tokens and {a['mean_llm_calls']:.1f} vs {r['mean_llm_calls']:.1f} LLM calls per question"
    return head + multi + cost + ". "

# ------------------------------------------------------------------ content
def build(author: str) -> None:
    st = styles()
    ret, thr, chunk = load("retrieval.json"), load("threshold_sweep.json"), load("chunk_sweep.json")
    ans, fails = load("answers_summary.json"), load("failures.json") or []
    qs = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n_ans = sum(q["answerable"] for q in qs)
    S: list = []

    S += [Paragraph("CreatorGuide Agent", st["title"]),
          Paragraph("A grounded, tool-using assistant for YouTube creator policies - capstone report", st["sub"]),
          Paragraph(f"{esc(author) + ' &middot; ' if author else ''}LLM application capstone &middot; built {date.today():%d %B %Y}", st["small"]), Spacer(1, 6)]

    # 1 ---------------------------------------------------------------------------------------
    S += [Paragraph("1. Problem and users", st["h1"]),
          Paragraph("YouTube creators constantly ask rule questions: how long a strike lasts, whether a strike can still be appealed, what a counter "
                    "notification is, how many subscribers and watch hours the Partner Program needs. The authoritative answers live in dozens of Help articles, "
                    "and the questions are often multi-step (\"I have 650 subscribers and 3,200 watch hours - how many more do I need?\"). "
                    "A model answering from memory is risky here: thresholds change, and a wrong answer about strikes or monetization has real cost for a creator.", st["body"]),
          Paragraph("<b>CreatorGuide Agent</b> answers such questions only from a frozen corpus of 62 English YouTube Help articles (policy, enforcement, monetization; "
                    "snapshot of 5 and 12 Oct 2026). Every answer is a validated JSON object with citations to the passages the model actually saw, and the system "
                    "declines when the articles do not cover the question. It is for creators and for support staff who need a quick, checkable answer.", st["body"])]

    # 2 ---------------------------------------------------------------------------------------
    S += [Paragraph("2. Design", st["h1"]), img(fig_architecture(), 15.5),
          Paragraph("Figure 1. Architecture. The agent is the only component that talks to the model; tools never run anything the allow-list does not name.", st["small"])]
    rows = [["Course technique", "Where it is used", "Why this choice"],
            ["Structured output + schema validation", "Final answer {answer, answerable, citations}; tool arguments",
             "Small models drift from format. Validation rejects wrong types, unknown keys and citations the model was never shown, then sends the error back for up to 2 repairs; otherwise the system abstains."],
            ["Retrieval / RAG with citations", "search_docs: BM25 + MiniLM dense, fused with Reciprocal Rank Fusion",
             "Policy text mixes exact terms (\"90-day\", \"YPP\") with paraphrases. BM25 catches the former, dense the latter; RRF fuses ranks, not incomparable scores."],
            ["Retrieval evaluation", "eval/: Hit@k, Recall@k, MRR on labelled documents; min_score and chunk-size sweeps",
             "Measures retrieval separately from generation, so a bad answer can be traced to its cause."],
            ["Agent loop with tools (agentic)", "search_docs, read_doc, calculator; max 6 tool rounds; repeat-call guard",
             "Multi-step questions need search + arithmetic. A deterministic calculator removes small-model arithmetic errors."],
            ["Served interface", "FastAPI JSON endpoint and an SSE stream rendered by a small web page",
             "Streaming each tool call makes the agent's reasoning visible, which is also the demo."]]
    S += [table(rows, [3.7 * cm, 5.1 * cm, 8.2 * cm], st), Spacer(1, 4),
          Paragraph("<b>Safety and robustness.</b> API key and base URL come only from the environment (the OpenAI SDK reads them). Tool names are allow-listed and arguments are validated "
                    "(types, lengths, ranges, unknown keys rejected). The calculator parses an AST and permits only numbers and arithmetic operators. Retrieved text and tool results are "
                    "framed as untrusted quoted data in every prompt. Temperature is 0 for everything that is evaluated. Tool results are capped at 5,000 characters and always remain valid JSON.", st["body"])]

    # 3 ---------------------------------------------------------------------------------------
    S += [Paragraph("3. Evaluation design", st["h1"]),
          Paragraph(f"<b>Test set.</b> {len(qs)} questions written for this project: {sum(q['type']=='single' for q in qs)} single-fact, "
                    f"{sum(q['type']=='multi' for q in qs)} multi-step (three need the calculator, two need facts from two different articles), and "
                    f"{sum(q['type']=='unanswerable' for q in qs)} unanswerable (off-corpus, including one adversarial case where \"CPM\" means Commerce Product Module in the corpus). "
                    "Every answerable question has gold document ids, a verbatim evidence quote from the article (checked automatically by <font face='Courier'>eval/check_questions.py</font>), "
                    "and key-fact groups: each group lists acceptable variants of one required fact (e.g. 350 and 800 for question m01).", st["body"]),
          Paragraph("<b>Systems compared.</b> <i>no_retrieval</i> (model alone, baseline 1); <i>rag_bm25</i> (naive default: BM25 without stopword filter, top-3, no relevance floor, single shot - baseline 2); "
                    "<i>rag_ours</i> (single shot with hybrid retrieval, stopword filter, top-5, calibrated floor); <i>agent</i> (tool-calling loop on the same retriever).", st["body"]),
          Paragraph("<b>Metrics.</b> Retrieval (document level, answerable questions): Hit@1/3/5, Recall@3/5, MRR@10. Answers (deterministic, no LLM judge): <i>accuracy</i> = every key-fact group "
                    "matched and the system chose to answer (for unanswerable questions: it abstained); <i>grounded</i> = correct and cites a gold document; <i>key-fact recall</i>; "
                    "<i>abstention accuracy</i> on unanswerable questions; <i>false-abstain rate</i>; cost (LLM calls, tokens). Differences between systems are reported with a "
                    "paired bootstrap 95% interval (10,000 resamples, seed 42) because with 30 questions a single question moves accuracy by 3.3 points.", st["body"])]

    # 4 ---------------------------------------------------------------------------------------
    S += [Paragraph("4. Results", st["h1"]), Paragraph("4.1 Retrieval", st["h2"])]
    if ret:
        hdr = ["System", "Hit@1", "Hit@3", "Hit@5", "Recall@3", "Recall@5", "MRR@10"]
        rows = [hdr] + [[n] + [f3(v[k]) for k in ("hit@1", "hit@3", "hit@5", "recall@3", "recall@5", "mrr@10")] for n, v in ret["summary"].items()]
        S += [table(rows, [4.2 * cm] + [2.1 * cm] * 6, st), Spacer(1, 3), img(fig_retrieval(ret["summary"]), 12.5),
              Paragraph(f"Table 1 / Figure 2. Retrieval on {n_ans} answerable questions, 224-token chunks, 62 documents.", st["small"])]
        if ret.get("skipped"):
            S.append(Paragraph("<b>Not run when this PDF was built:</b> " + esc(", ".join(ret["skipped"])) + " (the dense model sentence-transformers is not installed in the build environment). "
                               "Run <font face='Courier'>make eval-offline</font> on a machine with the requirements installed, then <font face='Courier'>make report</font>.", st["warn"]))
        for k, v in (ret.get("paired_bootstrap") or {}).items():
            S.append(Paragraph(f"{esc(k)}: mean difference {v['mean_diff']:+.3f}, 95% CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
                               + (" - the interval includes 0, so this is not a reliable gain." if v["ci_low"] <= 0 <= v["ci_high"] else "."), st["small"]))
    else:
        S.append(Paragraph("Retrieval results not found. Run make eval-offline.", st["warn"]))

    S.append(Paragraph("4.2 Measured improvements over naive defaults", st["h2"]))
    if ret and "bm25" in ret["summary"] and "bm25_stopwords" in ret["summary"]:
        a, b = ret["summary"]["bm25"], ret["summary"]["bm25_stopwords"]
        S.append(Paragraph(f"<b>Stopword filter for BM25.</b> Naive BM25 lets question words (\"how long do I have to\") outweigh content words. Removing a short list of function and question words "
                           f"(keeping \"not\", \"no\", \"under\", \"at\", \"least\", which change policy meaning) moves Hit@1 {f3(a['hit@1'])} to {f3(b['hit@1'])} and MRR@10 {f3(a['mrr@10'])} to {f3(b['mrr@10'])}. "
                           "The paired interval below shows whether this difference is reliable.", st["body"]))
    if thr and thr["results"]:
        name = "hybrid" if "hybrid" in thr["results"] else ("dense" if "dense" in thr["results"] else "bm25_stopwords")
        e = thr["results"][name]; b, nog = e["best"], e["no_gate"]
        S += [Paragraph(f"<b>Relevance floor (min_score), chosen from data.</b> For {esc(name)} the gate signal is <font face='Courier'>{e['gate_key']}</font> of the best top-5 hit. "
                        f"Without a floor the system rejects {pct(nog['tnr_unanswerable_rejected'])} of unanswerable questions. The chosen operating point, min_score = {b['threshold']}, "
                        f"keeps {pct(b['tpr_answerable_kept'])} of answerable questions and rejects {pct(b['tnr_unanswerable_rejected'])} of unanswerable ones "
                        f"(constraint: keep at least {pct(e['min_tpr'])} of answerable). Only 5 unanswerable questions exist, so the estimate is rough.", st["body"]),
              img(fig_gate(e, name), 12.5), Paragraph("Figure 3. Gate score of every question; unanswerable questions are not cleanly separable by score alone.", st["small"])]
        if name == "bm25_stopwords":
            S.append(Paragraph("A raw BM25 score is a weak gate (it depends on query length), which is why the dense cosine / hybrid rows matter; they are filled in once the dense model is available.", st["small"]))
    if chunk:
        rows = [["Chunking (BM25 + stopwords)", "Chunks", "Hit@3", "Recall@5", "MRR@10"]] + [[n, str(v["n_chunks"]), f3(v["hit@3"]), f3(v["recall@5"]), f3(v["mrr@10"])] for n, v in chunk["results"].items()]
        vals = [v["mrr@10"] for v in chunk["results"].values()]
        S += [Spacer(1, 4), table(rows, [6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm], st),
              Paragraph(f"Table 2. Chunk-size sweep (word-based chunks, offline). MRR@10 ranges only from {min(vals):.3f} to {max(vals):.3f} across sizes: on 25 questions this is within noise, "
                        "so I keep the 224-token index and do <b>not</b> claim a chunk-size improvement.", st["small"])]

    S.append(Paragraph("4.3 Answer quality (LLM)", st["h2"]))
    if ans:
        order = list(ans["summary"])
        hdr = ["System", "Accuracy (30)", "Single (20)", "Multi (5)", "Abstain ok (5)", "Grounded", "False abstain", "LLM calls", "Tokens"]
        rows = [hdr] + [[n, pct(s["accuracy_all"]), pct(s["accuracy_single"]), pct(s["accuracy_multi"]), pct(s["abstain_correct_unanswerable"]),
                         pct(s["grounded_rate"]), pct(s["false_abstain_rate"]), f"{s['mean_llm_calls']:.1f}", f"{s['mean_tokens']:.0f}"] for n, s in ans["summary"].items()]
        S += [table(rows, [3.0 * cm, 2.0 * cm, 1.8 * cm, 1.6 * cm, 2.0 * cm, 1.8 * cm, 1.9 * cm, 1.5 * cm, 1.4 * cm], st), Spacer(1, 3),
              img(fig_answers(ans["summary"]), 12.5), Paragraph(f"Table 3 / Figure 4. Answer accuracy with {esc(ans['meta'].get('llm_model', 'the model'))}, "
                                                                   f"temperature {ans['meta'].get('temperature', 0)}, retrieval = {esc(ans['meta'].get('retrieval', '?'))}.", st["small"])]
        for k, v in ans["paired_bootstrap"].items():
            S.append(Paragraph(f"{esc(k)}: {v['mean_diff']:+.3f} (95% CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}])"
                               + (" - includes 0." if v["ci_low"] <= 0 <= v["ci_high"] else " - excludes 0."), st["small"]))
    else:
        S.append(Paragraph("<b>LLM answer results were not available when this PDF was built.</b> They require the class proxy, which is not reachable from the build environment. "
                           "Run <font face='Courier'>make eval-answers</font> and then <font face='Courier'>make report</font>; this section, the analysis and the failure cases are generated from those files.", st["warn"]))

    # 5 ---------------------------------------------------------------------------------------
    S += [Paragraph("5. Analysis and failure cases", st["h1"])]
    if ans:
        s = ans["summary"]
        ranked = sorted(s.items(), key=lambda kv: -(kv[1]["accuracy_all"] or 0))
        S.append(Paragraph("Systems ranked by overall accuracy: " + ", ".join(f"{n} {pct(v['accuracy_all'])}" for n, v in ranked) + ". "
                           + (compare_sentence(s) if "agent" in s and "rag_ours" in s else "")
                           + "With 30 questions each question is 3.3 points, so read the intervals above before claiming that one system is better.", st["body"]))
        pb = ans["paired_bootstrap"].get("rag_ours_vs_rag_bm25_accuracy_all")
        if pb:
            S.append(Paragraph(f"Our single-shot RAG configuration (hybrid retrieval, stopword filter, top-5, relevance floor) vs the naive default (BM25, top-3, no floor): {pb['mean_diff'] * 100:+.1f} points "
                               f"(95% CI [{pb['ci_low'] * 100:+.1f}, {pb['ci_high'] * 100:+.1f}])" + ("; the interval excludes 0" if pb["ci_low"] > 0 or pb["ci_high"] < 0 else "; the interval includes 0")
                               + ". This compares a bundle of changes, so it does not show which of them matters; the retrieval-only comparison in 4.1 shows small differences.", st["body"]))
        n_err = {n: v["n_errors"] for n, v in s.items() if v["n_errors"]}
        if n_err:
            S.append(Paragraph(f"Runs with API errors or missing answers (counted as wrong): {esc(n_err)}.", st["warn"]))
        shown = 0
        for fcase in [f for f in fails if f["system"] == "agent"] + [f for f in fails if f["system"] == "rag_ours"]:
            if shown >= 4:
                break
            shown += 1
            exp = "; ".join("/".join(g) for g in fcase["expected_key_facts"]) or "(should abstain)"
            S.append(KeepTogether([Paragraph(f"<b>Failure {shown} - {esc(fcase['system'])} / {esc(fcase['qid'])}.</b> Q: {esc(fcase['question'])}<br/>"
                                             f"Expected: {esc(exp)}<br/>Got ({esc(fcase['stop_reason'])}, answerable={fcase['got_answerable']}): {esc(fcase['got_answer'][:260])}", st["body"])]))
        if not shown:
            S.append(Paragraph("No failures were recorded for agent or rag_ours; check failures.json for the baselines.", st["body"]))
    # always: a retrieval failure that is measured here
    if ret:
        import csv as _csv
        rows = list(_csv.DictReader((RES / "retrieval_per_question.csv").open(encoding="utf-8")))
        miss = [r for r in rows if r["system"] == "bm25" and float(r["hit@1"]) == 0.0]
        fixed = {r["qid"] for r in rows if r["system"] == "bm25_stopwords" and float(r["hit@1"]) == 1.0}
        qtext = {q["qid"]: q["question"] for q in qs}
        if miss:
            lines = [f"{esc(r['qid'])} (\"{esc(qtext[r['qid']])}\", top-3: {esc(r['top3_docs'])}; fixed by the stopword filter: {'yes' if r['qid'] in fixed else 'no'})" for r in miss[:4]]
            S.append(Paragraph("<b>Retrieval failure analysis (measured).</b> Naive BM25 ranks a gold document first for all but " + str(len(miss)) + " of " + str(n_ans) + " questions. Misses: " + "; ".join(lines)
                               + ". The pattern is lexical: chatty question words and generic terms (\"strike\", \"channel\") match many articles, so a document about a neighbouring topic can win.", st["body"]))
    notes = ROOT / "report" / "manual_notes.md"
    if notes.exists() and notes.read_text(encoding="utf-8").strip():
        S.append(Paragraph("<b>Additional observations.</b> " + esc(notes.read_text(encoding="utf-8").strip()).replace("\n", "<br/>"), st["body"]))

    # 6 ---------------------------------------------------------------------------------------
    S += [Paragraph("6. Limitations and next steps", st["h1"]),
          Paragraph("<b>Small test set.</b> 30 questions (25 answerable, 5 unanswerable) written by one person: intervals are wide and the min_score estimate is rough. "
                    "<b>Key-fact matching</b> is deterministic and strict about wording variants I listed; a correct answer phrased differently can be marked wrong, and a wrong answer that happens to contain the numbers can be marked right "
                    "(no LLM judge was used, because a 4B model judging itself is not trustworthy). <b>Corpus freshness.</b> The articles are a 5-12 Oct 2026 snapshot. Several already announce Partner Program changes effective 1 Feb 2027 (for example higher entry thresholds for new creators), so the gold answers of the threshold questions are valid only until 31 Jan 2027 and the model can be confused by the future-dated text. "
                    "<b>Model.</b> gemma-4-E4B-it is small: it can ignore tools or format; the validate-repair-abstain loop contains the damage but costs extra calls. "
                    "<b>Security.</b> Prompt injection from retrieved text is mitigated by framing and by tool allow-listing, not eliminated; there is no authentication or rate limiting on the API.", st["body"]),
          Paragraph("<b>Next steps:</b> grow the set to 100 questions with a second annotator; add a cross-encoder reranker and measure it against hybrid; add an LLM-judge faithfulness score validated against human labels; "
                    "expose the three tools through an MCP server; add Thai-language questions.", st["body"]),
          Paragraph("Reproducibility", st["h2"]),
          Paragraph("All numbers come from <font face='Courier'>eval/results/</font>, regenerated by <font face='Courier'>make eval-offline</font> and <font face='Courier'>make eval-answers</font>. "
                    f"Question-set sha256 {esc((ret or ans or {}).get('meta', {}).get('questions_sha256', 'n/a')[:16])}..., corpus sha256 {esc((ret or ans or {}).get('meta', {}).get('corpus_sha256', 'n/a')[:16])}.... "
                    "Dense retrieval uses the pinned all-MiniLM-L6-v2 revision stored in data/index/manifest.json.", st["small"])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="CreatorGuide Agent - capstone report", author=author or "")

    def footer(canvas, d):
        canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(2 * cm, 1.0 * cm, "CreatorGuide Agent - capstone report"); canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Page {d.page}")

    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="")
    build(ap.parse_args().author)
