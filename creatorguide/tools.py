"""Tools the agent may call. Allow-listed by name, arguments validated, results treated as untrusted data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import safe_math
from .corpus import Corpus, doc_id_of
from .retrieval import Retriever, passes_gate
from .schema import ValidationError, validate_args

MAX_RESULT_CHARS = 5000
SNIPPET_CHARS = 900
READ_CHARS = 3500

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_docs": {
        "description": "Search the YouTube Help articles. Returns the best-matching passages with chunk_id, title and text.",
        "args": {"query": {"type": "string", "required": True, "max_len": 300},
                 "top_k": {"type": "integer", "min": 1, "max": 8}},
    },
    "read_doc": {
        "description": "Read more of one article by doc_id (e.g. 'yt-72851'). Optional heading selects a section.",
        "args": {"doc_id": {"type": "string", "required": True, "max_len": 40},
                 "heading": {"type": "string", "max_len": 120}},
    },
    "calculator": {
        "description": "Evaluate an arithmetic expression exactly, e.g. '4000 - 3200'. Use it for ALL arithmetic.",
        "args": {"expression": {"type": "string", "required": True, "max_len": 200}},
    },
}
TOOL_NAMES = frozenset(TOOL_SPECS)


def openai_tool_specs() -> list[dict[str, Any]]:
    out = []
    for name, spec in TOOL_SPECS.items():
        props, required = {}, []
        for arg, rule in spec["args"].items():
            props[arg] = {"type": rule["type"]}
            if rule.get("required"):
                required.append(arg)
        out.append({"type": "function", "function": {
            "name": name, "description": spec["description"],
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return out


def describe_tools() -> str:
    lines = []
    for name, spec in TOOL_SPECS.items():
        sig = ", ".join(f"{a}{'' if r.get('required') else '?'}: {r['type']}" for a, r in spec["args"].items())
        lines.append(f"- {name}({sig}): {spec['description']}")
    return "\n".join(lines)


@dataclass
class ToolResult:
    ok: bool
    content: str  # JSON string handed back to the model

    @property
    def error(self) -> str | None:
        return None if self.ok else json.loads(self.content).get("error")


@dataclass
class ToolBox:
    """Per-question tool state. `seen_ids` is the allow-list for citations."""

    corpus: Corpus
    retriever: Retriever
    top_k: int = 5
    min_score: float = 0.0
    seen_ids: set[str] = field(default_factory=set)

    def execute(self, name: str, raw_args: str | dict[str, Any] | None) -> ToolResult:
        if name not in TOOL_NAMES:  # allow-list: anything else is refused, never dispatched
            return self._err(f"unknown tool '{name}'. Allowed: {sorted(TOOL_NAMES)}")
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            args = validate_args(TOOL_SPECS[name]["args"], args)
            payload = getattr(self, f"_{name}")(**args)
        except json.JSONDecodeError:
            return self._err("arguments are not valid JSON")
        except (ValidationError, safe_math.MathError, KeyError, ValueError) as exc:
            return self._err(str(exc))
        return ToolResult(True, self._fit(payload))

    @staticmethod
    def _fit(payload: dict[str, Any]) -> str:
        """Serialise within MAX_RESULT_CHARS and always return valid JSON (drop results / clip text, never slice JSON)."""
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) <= MAX_RESULT_CHARS:
            return text
        payload = dict(payload, truncated=True)
        results = list(payload.get("results", []))
        while results and len(json.dumps({**payload, "results": results}, ensure_ascii=False)) > MAX_RESULT_CHARS:
            results.pop()
        if "results" in payload:
            payload["results"] = results
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) > MAX_RESULT_CHARS and isinstance(payload.get("text"), str):
            overflow = len(text) - MAX_RESULT_CHARS
            payload["text"] = payload["text"][: max(0, len(payload["text"]) - overflow - 20)]
            text = json.dumps(payload, ensure_ascii=False)
        return text

    @staticmethod
    def _err(message: str) -> ToolResult:
        return ToolResult(False, json.dumps({"error": message}, ensure_ascii=False))

    # ---- tool implementations -------------------------------------------------------------
    def _search_docs(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        hits = self.retriever.search(query, top_k or self.top_k)
        if not passes_gate(hits, self.retriever.gate_key, self.min_score):
            return {"results": [], "note": "No passage reached the relevance floor; the articles probably do not cover this."}
        results = []
        for h in hits:
            self.seen_ids.update({h["chunk_id"], h["doc_id"]})
            body = h["text"].split("\n\n", 1)[-1]
            results.append({"chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "title": h["title"],
                            "heading": h["heading"], "score": h["score"], "text": body[:SNIPPET_CHARS]})
        return {"results": results}

    def _read_doc(self, doc_id: str, heading: str | None = None) -> dict[str, Any]:
        doc_id = doc_id_of(doc_id)
        if doc_id not in self.corpus:
            raise ValidationError(f"unknown doc_id '{doc_id}'")
        sections = self.corpus.sections(doc_id)
        headings = [h for h, _ in sections]
        if heading:
            wanted = [(h, b) for h, b in sections if heading.lower() in h.lower()]
            if not wanted:
                raise ValidationError(f"no section matches '{heading}'. Headings: {headings}")
            sections = wanted
        text, used = "", []
        for h, b in sections:
            block = f"## {h}\n{b}\n"
            if len(text) + len(block) > READ_CHARS:
                text += block[: max(0, READ_CHARS - len(text))]
                used.append(h)
                break
            text += block
            used.append(h)
        self.seen_ids.add(doc_id)
        return {"doc_id": doc_id, "title": self.corpus.title(doc_id), "url": self.corpus.url(doc_id),
                "headings": headings, "sections_returned": used, "text": text}

    def _calculator(self, expression: str) -> dict[str, Any]:
        return {"expression": expression, "result": safe_math.format_number(safe_math.evaluate(expression))}
