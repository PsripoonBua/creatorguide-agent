import hashlib
import json
import unittest

from tests.fakes import ROOT, corpus_and_bm25

from creatorguide import safe_math
from creatorguide.retrieval import BM25Retriever, doc_ranking, passes_gate, rrf_fuse, tokenize
from creatorguide.schema import (ValidationError, extract_json, validate_args, validate_final)
from creatorguide.tools import TOOL_SPECS, ToolBox, openai_tool_specs


class SafeMath(unittest.TestCase):
    def test_values(self):
        self.assertEqual(safe_math.evaluate("4,000 - 3,200"), 800)
        self.assertEqual(safe_math.evaluate("(1000-650)*2"), 700)
        self.assertEqual(safe_math.evaluate("max(1, 2) + abs(-3)"), 5)

    def test_rejects_unsafe(self):
        for bad in ["__import__('os')", "a+1", "2**9999", "1/0", "().__class__", "'x'*3", "[1]", "x=1", "1e999*1e999"]:
            with self.assertRaises(safe_math.MathError, msg=bad):
                safe_math.evaluate(bad)


class Schema(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('x {"a": 1} y'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 2}\n```'), {"a": 2})
        with self.assertRaises(ValidationError):
            extract_json("no json here")
        with self.assertRaises(ValidationError):
            extract_json(None)

    def test_final_rules(self):
        ok = validate_final({"answer": "x", "answerable": True, "citations": ["a", "a"]}, {"a"})
        self.assertEqual(ok.citations, ["a"])
        bads = [
            {"answer": "x", "answerable": True, "citations": []},            # no citation
            {"answer": "x", "answerable": True, "citations": ["zz"]},        # not shown to the model
            {"answer": "", "answerable": False},                              # empty answer
            {"answer": "x", "answerable": "yes", "citations": []},           # wrong type
            {"answer": "x", "answerable": False, "citations": [], "extra": 1},  # unknown key
        ]
        for bad in bads:
            with self.assertRaises(ValidationError, msg=bad):
                validate_final(bad, {"a"})
        self.assertEqual(validate_final({"answer": "n/a", "answerable": False, "citations": ["a"]}, {"a"}).citations, [])

    def test_validate_args(self):
        spec = TOOL_SPECS["search_docs"]["args"]
        self.assertEqual(validate_args(spec, {"query": " hi ", "top_k": "3"}), {"query": "hi", "top_k": 3})
        for bad in [{}, {"query": "x", "top_k": 99}, {"query": "x", "extra": 1}, {"query": 5}, "str"]:
            with self.assertRaises(ValidationError, msg=bad):
                validate_args(spec, bad)


class Retrieval(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Super Chat in 90 days!"), ["super", "chat", "in", "90", "days"])

    def test_bm25_finds_obvious_docs(self):
        _, bm25 = corpus_and_bm25()
        self.assertEqual(bm25.search("what is a copyright counter notification", 3)[0]["doc_id"], "yt-2807684")
        self.assertIn("yt-185111", doc_ranking(bm25.search("appeal a Community Guidelines strike", 5)))

    def test_stopword_filter_fixes_chatty_question(self):
        # Without filtering, "how long do I have to" outweighs the content words (see report, failure analysis).
        corpus, bm25 = corpus_and_bm25()
        better = BM25Retriever(bm25.chunks, remove_stopwords=True)
        q = "how long do I have to appeal a strike"
        self.assertNotIn("yt-185111", doc_ranking(bm25.search(q, 5)))
        self.assertIn("yt-185111", doc_ranking(better.search(q, 5)))

    def test_bm25_no_overlap_returns_nothing(self):
        _, bm25 = corpus_and_bm25()
        self.assertEqual(bm25.search("zzzxqv qqqwww", 5), [])
        with self.assertRaises(ValueError):
            bm25.search("   ")

    def test_rrf_hand_computed(self):
        fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62 + 1 / 61)
        self.assertEqual([c for c, _ in rrf_fuse([["a", "b"], ["b", "c"]], k=60)][0], "b")

    def test_gate(self):
        hits = [{"bm25_score": 3.0}, {"bm25_score": 7.5}]
        self.assertTrue(passes_gate(hits, "bm25_score", 7.5))
        self.assertFalse(passes_gate(hits, "bm25_score", 7.6))
        self.assertFalse(passes_gate([], "bm25_score", 0.0))


class Tools(unittest.TestCase):
    def box(self, **kw):
        corpus, bm25 = corpus_and_bm25()
        return ToolBox(corpus, bm25, **kw)

    def test_allow_list_and_validation(self):
        box = self.box()
        self.assertFalse(box.execute("shell", {"cmd": "ls"}).ok)
        self.assertFalse(box.execute("search_docs", {"query": ""}).ok)
        self.assertFalse(box.execute("search_docs", "{not json").ok)
        self.assertFalse(box.execute("read_doc", {"doc_id": "../../etc/passwd"}).ok)
        self.assertFalse(box.execute("calculator", {"expression": "__import__('os')"}).ok)

    def test_search_registers_citable_ids(self):
        box = self.box()
        res = box.execute("search_docs", {"query": "counter notification", "top_k": 2})
        self.assertTrue(res.ok)
        first = json.loads(res.content)["results"][0]
        self.assertIn(first["chunk_id"], box.seen_ids)
        self.assertIn(first["doc_id"], box.seen_ids)

    def test_min_score_floor_returns_nothing(self):
        box = self.box(min_score=10_000)
        self.assertEqual(json.loads(box.execute("search_docs", {"query": "counter notification"}).content)["results"], [])
        self.assertEqual(box.seen_ids, set())

    def test_read_doc_and_calculator(self):
        box = self.box()
        out = json.loads(box.execute("read_doc", {"doc_id": "yt-185111", "heading": "Appeal a strike"}).content)
        self.assertIn("APPEAL", out["text"])
        self.assertLessEqual(len(out["text"]), 3600)
        self.assertEqual(json.loads(box.execute("calculator", {"expression": "4000-3200"}).content)["result"], "800")

    def test_oversized_results_stay_valid_json(self):
        box = self.box()
        res = box.execute("search_docs", {"query": "strike appeal copyright channel", "top_k": 8})
        data = json.loads(res.content)              # must parse even when results had to be dropped
        self.assertLessEqual(len(res.content), 5000)
        self.assertTrue(data.get("truncated"))
        for chunk in data["results"]:               # every returned chunk is citable
            self.assertIn(chunk["chunk_id"], box.seen_ids)
        long_doc = json.loads(box.execute("read_doc", {"doc_id": "yt-10008196"}).content)
        self.assertLessEqual(len(box.execute("read_doc", {"doc_id": "yt-10008196"}).content), 5000)
        self.assertIn("text", long_doc)

    def test_openai_specs_cover_every_tool(self):
        self.assertEqual({t["function"]["name"] for t in openai_tool_specs()}, set(TOOL_SPECS))


class DataIntegrity(unittest.TestCase):
    def test_index_matches_corpus(self):
        manifest = json.loads((ROOT / "data/index/manifest.json").read_text())
        sha = hashlib.sha256((ROOT / "data/docs.jsonl").read_bytes()).hexdigest()
        self.assertEqual(sha, manifest["source_sha256"])
        corpus, bm25 = corpus_and_bm25()
        self.assertEqual(len(bm25.chunks), manifest["chunks"])
        self.assertEqual({c["doc_id"] for c in bm25.chunks}, set(corpus.docs))


if __name__ == "__main__":
    unittest.main()
