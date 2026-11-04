"""Test doubles: a scripted LLM and a fake embedder. No network anywhere in the test-suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creatorguide.config import ROOT, Settings  # noqa: E402
from creatorguide.corpus import Corpus  # noqa: E402
from creatorguide.llm import LLMReply, ToolCall, ToolsUnsupportedError  # noqa: E402
from creatorguide.retrieval import BM25Retriever, load_index  # noqa: E402


class ScriptedLLM:
    """Returns pre-written replies in order and records every request it received."""

    def __init__(self, replies, reject_tools=False):
        self.replies, self.calls, self.reject_tools = list(replies), [], reject_tools

    def chat(self, messages, tools=None):
        if tools and self.reject_tools:
            raise ToolsUnsupportedError("tools not supported")
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        r = self.replies.pop(0)
        return r if isinstance(r, LLMReply) else LLMReply(content=r, usage={"prompt_tokens": 10, "completion_tokens": 5})


def tool_call(name, args, call_id="c1"):
    return LLMReply(content=None, tool_calls=[ToolCall(call_id, name, json.dumps(args))],
                    usage={"prompt_tokens": 10, "completion_tokens": 5})


def final_json(answer="ok", answerable=True, citations=()):
    return json.dumps({"answer": answer, "answerable": answerable, "citations": list(citations)})


class OneHotEmbedder:
    """Maps a query to the embedding of the chunk whose text shares the most words (deterministic)."""

    def __init__(self, chunks, embeddings):
        self.chunks, self.embeddings = chunks, embeddings

    def embed_query(self, text):
        words = set(text.lower().split())
        best = max(range(len(self.chunks)), key=lambda i: len(words & set(self.chunks[i]["text"].lower().split())))
        return self.embeddings[best]


_cache = {}


def corpus_and_bm25():
    if "c" not in _cache:
        chunks, _ = load_index(ROOT / "data" / "index")
        _cache["c"] = (Corpus.load(ROOT / "data" / "docs.jsonl"), BM25Retriever(chunks))
    return _cache["c"]


def settings(**kw):
    return Settings(**{"agent_mode": "native", "max_steps": 4, **kw}).validate()
