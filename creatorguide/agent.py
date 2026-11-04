"""Tool-calling agent loop with explicit stop conditions.

Stops when: (1) a valid final answer is produced, (2) `max_steps` tool rounds are used (then the
model is forced to answer from what it has), (3) the model keeps producing invalid output (abstain),
or (4) the same tool call repeats too often (loop guard).

Two ways to express tool calls, selected by AGENT_MODE:
  native - OpenAI `tools=` function calling
  json   - the model replies with {"action": "tool", ...}; works on endpoints without tool support
  auto   - try native, fall back to json if the endpoint rejects `tools=`
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterator

from .config import Settings
from .corpus import Corpus, doc_id_of
from .llm import LLM, ToolsUnsupportedError, assistant_message_with_calls
from .prompts import FORCE_FINAL, REPAIR_TEMPLATE, SEARCH_FIRST, agent_system, tool_result_message
from .rag import _check_question
from .retrieval import Retriever
from .schema import FinalAnswer, ValidationError, abstain, extract_json, validate_final
from .structured import Result, add_usage, generate_final
from .tools import ToolBox, openai_tool_specs

MAX_CALLS_PER_TURN = 3
MAX_REPEATS = 2
MAX_REPAIRS = 2


class Agent:
    def __init__(self, llm: LLM, corpus: Corpus, retriever: Retriever, settings: Settings) -> None:
        self.llm, self.corpus, self.retriever, self.settings = llm, corpus, retriever, settings

    # ------------------------------------------------------------------------------------
    def run(self, question: str) -> Result:
        final_event: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        for event in self.stream(question):
            events.append(event)
            if event["type"] == "final":
                final_event = event
        answer = FinalAnswer(final_event["answer"], final_event["answerable"], final_event["citations"])
        return Result(final=answer, stop_reason=final_event["stop_reason"], usage=final_event["usage"],
                      events=[e for e in events if e["type"] != "final"], system="agent",
                      retrieved_docs=final_event["retrieved_docs"], tool_calls=final_event["tool_calls"],
                      mode=final_event["mode"])

    # ------------------------------------------------------------------------------------
    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        question = _check_question(question)
        s = self.settings
        toolbox = ToolBox(self.corpus, self.retriever, top_k=s.top_k, min_score=s.min_score)
        usage: dict[str, int] = {}
        mode = "json" if s.agent_mode == "json" else "native"
        messages = self._start(question, native=(mode == "native"))
        rounds = repairs = 0
        repeats: Counter[str] = Counter()
        docs_shown: list[str] = []

        def finish(answer: FinalAnswer, reason: str) -> dict[str, Any]:
            return {"type": "final", **answer.to_dict(), "stop_reason": reason, "usage": usage, "mode": mode,
                    "tool_calls": rounds_calls[0], "retrieved_docs": docs_shown,
                    "sources": self._sources(answer.citations)}

        rounds_calls = [0]

        def run_tool(name: str, args: Any) -> str:
            rounds_calls[0] += 1
            yield_events.append({"type": "tool_call", "name": name, "args": args if isinstance(args, dict) else str(args)[:200]})
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            repeats[key] += 1
            if repeats[key] > MAX_REPEATS:
                content = json.dumps({"error": "You already made this exact call. Use what you have and answer."})
                ok = False
            else:
                result = toolbox.execute(name, args)
                content, ok = result.content, result.ok
                if ok and name == "search_docs":
                    for r in json.loads(content).get("results", []):
                        if r["doc_id"] not in docs_shown:
                            docs_shown.append(r["doc_id"])
            yield_events.append({"type": "tool_result", "name": name, "ok": ok, "preview": content[:300]})
            return content

        yield_events: list[dict[str, Any]] = []

        while True:
            if rounds >= s.max_steps:
                yield {"type": "note", "message": "tool budget used; forcing a final answer"}
                messages.append({"role": "user", "content": FORCE_FINAL})
                final, err, _ = generate_final(self.llm, messages, toolbox.seen_ids, usage)
                yield finish(final or abstain(), "max_steps" if final else "invalid_output")
                return

            try:
                reply = self.llm.chat(messages, tools=openai_tool_specs() if mode == "native" else None)
            except ToolsUnsupportedError:
                if s.agent_mode != "auto" or rounds:
                    raise
                mode = "json"
                messages = self._start(question, native=False)
                yield {"type": "note", "message": "endpoint rejected tools=; switched to JSON action protocol"}
                continue
            add_usage(usage, reply.usage)

            # --- native function calls ------------------------------------------------------
            if reply.tool_calls:
                messages.append(assistant_message_with_calls(reply))
                for call in reply.tool_calls[:MAX_CALLS_PER_TURN]:
                    try:
                        args: Any = json.loads(call.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    content = run_tool(call.name, args)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                for call in reply.tool_calls[MAX_CALLS_PER_TURN:]:  # every call id needs an answer
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": json.dumps({"error": "too many calls in one turn"})})
                rounds += 1
                yield from yield_events
                yield_events.clear()
                continue

            # --- text reply: either a JSON tool action or the final answer -------------------
            try:
                obj = extract_json(reply.content)
                if obj.get("action") == "tool" or "tool" in obj:
                    name = obj.get("tool")
                    if not isinstance(name, str):
                        raise ValidationError('"tool" must be a tool name string')
                    content = run_tool(name, obj.get("args", {}))
                    messages.append({"role": "assistant", "content": reply.content or ""})
                    messages.append({"role": "user", "content": tool_result_message(name, content)})
                    rounds += 1
                    yield from yield_events
                    yield_events.clear()
                    continue
                if rounds == 0:
                    raise ValidationError(SEARCH_FIRST)
                answer = validate_final(obj, toolbox.seen_ids)
            except ValidationError as exc:
                if repairs >= MAX_REPAIRS:
                    yield {"type": "note", "message": f"giving up after repeated invalid output: {exc}"}
                    yield finish(abstain(), "invalid_output")
                    return
                repairs += 1
                yield {"type": "note", "message": f"repair {repairs}: {exc}"}
                messages.append({"role": "assistant", "content": reply.content or ""})
                messages.append({"role": "user", "content": REPAIR_TEMPLATE.format(error=exc)})
                continue

            yield finish(answer, "final")
            return

    # ------------------------------------------------------------------------------------
    def _start(self, question: str, native: bool) -> list[dict[str, Any]]:
        return [{"role": "system", "content": agent_system(native)}, {"role": "user", "content": question}]

    def _sources(self, citations: list[str]) -> list[dict[str, str]]:
        out, seen = [], set()
        for c in citations:
            d = doc_id_of(c)
            if d in self.corpus and d not in seen:
                seen.add(d)
                out.append({"id": c, "doc_id": d, "title": self.corpus.title(d), "url": self.corpus.url(d)})
        return out
