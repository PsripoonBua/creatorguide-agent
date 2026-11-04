"""Thin wrapper over the OpenAI SDK (pointed at the class proxy through env vars).

Everything else in the project talks to the `LLM` protocol, so tests and offline
evaluation can swap in a scripted fake without touching the network.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolsUnsupportedError(RuntimeError):
    """The endpoint rejected the `tools=` parameter (so use the JSON action protocol)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model produced it (untrusted)


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply: ...


_TRANSIENT = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}


class OpenAILLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700, client: Any = None,
                 retries: int = 3) -> None:
        if client is None:
            from openai import OpenAI  # reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment

            client = OpenAI()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

    def chat(self, messages, tools=None) -> LLMReply:
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages,
                                      temperature=self.temperature, max_tokens=self.max_tokens)
        if tools:
            kwargs["tools"] = tools
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - classify by name to avoid importing SDK internals
                name = type(exc).__name__
                if tools and name == "BadRequestError" and "tool" in str(exc).lower():
                    raise ToolsUnsupportedError(str(exc)) from exc
                if name in _TRANSIENT and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        message = response.choices[0].message
        calls = [
            ToolCall(id=c.id or f"call_{i}", name=c.function.name, arguments=c.function.arguments or "{}")
            for i, c in enumerate(getattr(message, "tool_calls", None) or [])
        ]
        usage = getattr(response, "usage", None)
        return LLMReply(
            content=message.content,
            tool_calls=calls,
            usage={"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                   "completion_tokens": getattr(usage, "completion_tokens", 0) or 0},
        )


def assistant_message_with_calls(reply: LLMReply) -> dict[str, Any]:
    """Re-serialise an assistant turn that contained tool calls, in OpenAI message format."""
    return {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in reply.tool_calls
        ],
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
