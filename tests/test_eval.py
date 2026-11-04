import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import check_questions  # noqa: E402
import metrics as M  # noqa: E402
from run_threshold_sweep import sweep  # noqa: E402


class EvalSet(unittest.TestCase):
    def test_questions_are_valid_and_evidence_verbatim(self):
        self.assertEqual(check_questions.check(), [])

    def test_composition(self):
        qs = check_questions.load_questions()
        self.assertGreaterEqual(len(qs), 30)
        types = {t: sum(q["type"] == t for q in qs) for t in ("single", "multi", "unanswerable")}
        self.assertEqual(types, {"single": 20, "multi": 5, "unanswerable": 5})

    def test_checker_catches_a_fabricated_quote(self):
        qs = check_questions.load_questions()
        qs[0]["evidence"][0]["quote"] = "this sentence is not in the article"
        tmp = ROOT / "eval" / "results" / "_tmp_bad.jsonl"
        tmp.write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
        try:
            self.assertTrue(any("not found verbatim" in p for p in check_questions.check(tmp)))
        finally:
            tmp.unlink()


class Metrics(unittest.TestCase):
    def test_retrieval_by_hand(self):
        ranked, gold = ["a", "b", "c", "d"], {"c", "z"}
        self.assertEqual(M.hit_at_k(ranked, gold, 2), 0.0)
        self.assertEqual(M.hit_at_k(ranked, gold, 3), 1.0)
        self.assertEqual(M.recall_at_k(ranked, gold, 5), 0.5)
        self.assertAlmostEqual(M.reciprocal_rank(ranked, gold), 1 / 3)
        self.assertEqual(M.reciprocal_rank(["x"], gold), 0.0)

    def test_key_fact_matching_is_token_aware(self):
        self.assertEqual(M.key_fact_groups("You need 3,000 hours", [["3000"]]), [True])        # comma-normalised
        self.assertEqual(M.key_fact_groups("a valid email", [["id"]]), [False])                 # 'id' is not inside 'valid'
        self.assertEqual(M.key_fact_groups("Use a valid ID or video", [["id", "video"]]), [True])
        self.assertEqual(M.key_fact_groups("two weeks", [["2 weeks", "two weeks"], ["x"]]), [True, False])

    def test_score_answer_cases(self):
        q = {"qid": "q", "type": "single", "answerable": True, "gold_doc_ids": ["d1"], "key_facts": [["800"]]}
        good = {"answer": "800 more hours", "answerable": True, "citations": ["d1:0001"], "usage": {}}
        self.assertEqual(M.score_answer(q, good)["grounded"], 1.0)
        wrong_source = {**good, "citations": ["d2:0001"]}
        self.assertEqual(M.score_answer(q, wrong_source)["correct"], 1.0)
        self.assertEqual(M.score_answer(q, wrong_source)["grounded"], 0.0)
        abstained = {"answer": "800", "answerable": False, "citations": [], "usage": {}}
        self.assertEqual(M.score_answer(q, abstained)["correct"], 0.0)   # saying "can't answer" earns nothing
        u = {"qid": "u", "type": "unanswerable", "answerable": False, "gold_doc_ids": [], "key_facts": []}
        self.assertEqual(M.score_answer(u, abstained)["correct"], 1.0)
        self.assertEqual(M.score_answer(u, good)["correct"], 0.0)

    def test_summary_and_bootstrap(self):
        rows = [{"type": "single", "correct": 1.0, "kf_recall": 1.0, "grounded": 1.0, "citation_precision": 1.0,
                 "pred_answerable": True, "llm_calls": 1, "tool_calls": 0, "tokens": 10},
                {"type": "unanswerable", "correct": 0.0, "kf_recall": None, "grounded": None, "citation_precision": None,
                 "pred_answerable": True, "llm_calls": 1, "tool_calls": 0, "tokens": 10}]
        s = M.summarize_answers(rows)
        self.assertEqual(s["accuracy_all"], 0.5)
        self.assertEqual(s["abstain_correct_unanswerable"], 0.0)
        b = M.paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1])
        self.assertEqual((b["mean_diff"], b["ci_low"], b["ci_high"]), (1.0, 1.0, 1.0))
        self.assertEqual(M.paired_bootstrap([1, 0, 1], [0, 1, 1], n=500), M.paired_bootstrap([1, 0, 1], [0, 1, 1], n=500))

    def test_threshold_sweep_operating_point(self):
        r = sweep([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 20])
        self.assertGreaterEqual(r["best"]["tpr_answerable_kept"], 0.9)
        self.assertEqual(r["best"]["tnr_unanswerable_rejected"], round(2 / 3, 4))
        self.assertEqual(r["no_gate"]["tnr_unanswerable_rejected"], 0.0)


if __name__ == "__main__":
    unittest.main()
