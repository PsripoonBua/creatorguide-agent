"""End-to-end smoke test of eval/run_answer_eval.py + score_answers.py with a fake model (no network)."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from creatorguide.llm import LLMReply, ToolCall  # noqa: E402


class FakeProxyLLM:
    """Stands in for OpenAILLM: cites the first chunk id it was shown, and uses the calculator once if allowed."""

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, tools=None):
        text = "\n".join(str(m.get("content") or "") for m in messages)
        ids = re.findall(r'"chunk_id": "(yt-\d+:\d{4})"|\[(yt-\d+:\d{4})\]', text)
        cid = next((a or b for a, b in ids), None)
        usage = {"prompt_tokens": 5, "completion_tokens": 5}
        if tools and not any(m["role"] == "tool" for m in messages):
            return LLMReply(None, [ToolCall("c1", "search_docs", json.dumps({"query": messages[1]["content"][:200]}))], usage)
        if cid is None:
            return LLMReply(json.dumps({"answer": "unknown", "answerable": False, "citations": []}), usage=usage)
        return LLMReply(json.dumps({"answer": "800 hours, 350 subscribers, 90 days", "answerable": True,
                                    "citations": [cid]}), usage=usage)


class Smoke(unittest.TestCase):
    def test_runs_all_systems_and_resumes(self):
        import _common
        import run_answer_eval
        import score_answers

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            patches = [mock.patch.object(m, "RESULTS", results) for m in (_common, run_answer_eval, score_answers)]
            patches.append(mock.patch.object(run_answer_eval, "OpenAILLM", FakeProxyLLM))
            patches.append(mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}))
            for p in patches:
                p.start()
            try:
                argv = ["run_answer_eval.py", "--retrieval", "bm25", "--limit", "12"]
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_answer_eval.main(), 0)
                for system in run_answer_eval.ALL:
                    lines = (results / f"answers_{system}.jsonl").read_text().splitlines()
                    self.assertEqual(len(lines), 12)
                    self.assertTrue(all(json.loads(l)["error"] is None for l in lines))
                summary = json.loads((results / "answers_summary.json").read_text())
                self.assertEqual(set(summary["summary"]), set(run_answer_eval.ALL))
                self.assertIn("agent_vs_rag_ours_accuracy_all", summary["paired_bootstrap"])
                self.assertTrue((results / "answers_per_question.csv").exists())
                # second run reuses saved answers: the fake model must not be called again
                with mock.patch.object(sys, "argv", argv), mock.patch.object(FakeProxyLLM, "chat", side_effect=AssertionError("called")):
                    self.assertEqual(run_answer_eval.main(), 0)
            finally:
                for p in reversed(patches):
                    p.stop()


if __name__ == "__main__":
    unittest.main()
