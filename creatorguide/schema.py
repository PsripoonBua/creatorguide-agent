"""Validation for everything that crosses a trust boundary: model JSON, tool arguments, citations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

ABSTAIN_TEXT = "I could not find this in the YouTube Help articles I can search."
MAX_ANSWER_CHARS = 1500
MAX_CITATIONS = 6


class ValidationError(ValueError):
    """Model output or tool arguments failed validation (message is safe to show back to the model)."""


@dataclass
class FinalAnswer:
    answer: str
    answerable: bool
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "answerable": self.answerable, "citations": list(self.citations)}


def abstain() -> FinalAnswer:
    return FinalAnswer(answer=ABSTAIN_TEXT, answerable=False, citations=[])


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str | None) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (handles ``` fences and chatter around it)."""
    if not text or not text.strip():
        raise ValidationError("empty reply; expected a JSON object")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValidationError("no valid JSON object found in the reply")


def validate_final(obj: dict[str, Any], allowed_ids: set[str] | None = None,
                   require_citations: bool = True) -> FinalAnswer:
    """Schema: {"answer": str, "answerable": bool, "citations": [str, ...]}.

    allowed_ids: chunk ids and document ids the model has actually been shown. A citation outside
    this set is a hallucinated source and is rejected, not silently dropped.
    """
    if not isinstance(obj, dict):
        raise ValidationError("final answer must be a JSON object")
    unknown = set(obj) - {"answer", "answerable", "citations", "action"}
    if unknown:
        raise ValidationError(f"unexpected keys: {sorted(unknown)}")
    answer, answerable, citations = obj.get("answer"), obj.get("answerable"), obj.get("citations", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ValidationError('"answer" must be a non-empty string')
    if len(answer) > MAX_ANSWER_CHARS:
        raise ValidationError(f'"answer" is longer than {MAX_ANSWER_CHARS} characters')
    if not isinstance(answerable, bool):
        raise ValidationError('"answerable" must be true or false')
    if not isinstance(citations, list) or not all(isinstance(c, str) and c.strip() for c in citations):
        raise ValidationError('"citations" must be a list of non-empty strings')
    citations = list(dict.fromkeys(c.strip() for c in citations))
    if len(citations) > MAX_CITATIONS:
        raise ValidationError(f'"citations" has more than {MAX_CITATIONS} entries')
    if allowed_ids is not None:
        bad = [c for c in citations if c not in allowed_ids]
        if bad:
            raise ValidationError(f"citations not found in the evidence you were shown: {bad}")
    if answerable and require_citations and not citations:
        raise ValidationError('answerable=true needs at least one citation (use answerable=false if unsure)')
    if not answerable:
        citations = []
    return FinalAnswer(answer=answer.strip(), answerable=answerable, citations=citations)


def validate_args(spec: dict[str, dict[str, Any]], args: Any) -> dict[str, Any]:
    """Check tool arguments against a tiny schema: {name: {type, required, min, max, max_len}}."""
    if not isinstance(args, dict):
        raise ValidationError("arguments must be a JSON object")
    extra = set(args) - set(spec)
    if extra:
        raise ValidationError(f"unexpected arguments: {sorted(extra)}")
    clean: dict[str, Any] = {}
    for name, rule in spec.items():
        if name not in args or args[name] is None:
            if rule.get("required"):
                raise ValidationError(f"missing required argument: {name}")
            continue
        value, kind = args[name], rule["type"]
        if kind == "string":
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{name} must be a non-empty string")
            if len(value) > rule.get("max_len", 300):
                raise ValidationError(f"{name} is longer than {rule.get('max_len', 300)} characters")
            value = value.strip()
        elif kind == "integer":
            if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"{name} must be an integer")
            if not rule.get("min", -10**9) <= value <= rule.get("max", 10**9):
                raise ValidationError(f"{name} must be between {rule.get('min')} and {rule.get('max')}")
        else:  # pragma: no cover
            raise ValueError(f"unsupported type {kind}")
        clean[name] = value
    return clean
