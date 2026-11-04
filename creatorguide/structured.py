"""Shared result type and the validate-and-repair loop for structured (JSON) model output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLM
from .prompts import REPAIR_TEMPLATE
from .schema import FinalAnswer, ValidationError, abstain, extract_json, validate_final


@dataclass
class Result:
    final: FinalAnswer
    stop_reason: str                       # final | below_min_score | invalid_output | max_steps | error
    usage: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    system: str = ""
    retrieved_docs: list[str] = field(default_factory=list)   # doc ids shown to the model, in order
    tool_calls: int = 0
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**self.final.to_dict(), "stop_reason": self.stop_reason, "usage": self.usage,
                "system": self.system, "retrieved_docs": self.retrieved_docs,
                "tool_calls": self.tool_calls, "mode": self.mode, "events": self.events}


def add_usage(total: dict[str, int], more: dict[str, int]) -> None:
    for key, value in more.items():
        total[key] = total.get(key, 0) + value
    total["llm_calls"] = total.get("llm_calls", 0) + 1


def generate_final(llm: LLM, messages: list[dict[str, Any]], allowed_ids: set[str] | None,
                   usage: dict[str, int], require_citations: bool = True,
                   max_repairs: int = 2) -> tuple[FinalAnswer | None, str, int]:
    """Ask for the final JSON, validate it, and feed the error back up to `max_repairs` times.

    Returns (answer or None, last_error, repairs_used). `messages` is extended in place.
    """
    error = ""
    for attempt in range(max_repairs + 1):
        reply = llm.chat(messages)
        add_usage(usage, reply.usage)
        try:
            return validate_final(extract_json(reply.content), allowed_ids, require_citations), "", attempt
        except ValidationError as exc:
            error = str(exc)
            messages.append({"role": "assistant", "content": reply.content or ""})
            messages.append({"role": "user", "content": REPAIR_TEMPLATE.format(error=error)})
    return None, error, max_repairs


def fallback(system: str, stop_reason: str, usage: dict[str, int], **kw: Any) -> Result:
    return Result(final=abstain(), stop_reason=stop_reason, usage=usage, system=system, **kw)
