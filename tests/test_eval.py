"""Regression tests for the eval harness (evals/run_eval.py, no network)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evals"))

from run_eval import evaluate, load_cases  # noqa: E402


class EvalHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.report = evaluate(cls.cases)

    def test_dataset_shape(self):
        # Every case is labeled, and both EN and CJK are covered.
        self.assertGreaterEqual(len(self.cases), 40)
        langs = {c.get("lang") for c in self.cases}
        self.assertIn("en", langs)
        self.assertIn("zh", langs)

    def test_dataset_covers_all_tasks(self):
        expected_tasks = {c["expected"] for c in self.cases}
        self.assertEqual(
            expected_tasks,
            {"simple", "coding", "reasoning", "long_context", "translation", "creative"},
        )

    def test_overall_accuracy_floor(self):
        # Regression guard: keyword classifier must stay above 90% on the
        # labeled set. If a classify() change trips this, either fix the
        # regression or consciously relabel/extend the dataset.
        self.assertGreaterEqual(self.report["accuracy"], 0.9)

    def test_cjk_accuracy_floor(self):
        # The router's reason for existing: Chinese prompts must classify well.
        self.assertGreaterEqual(self.report["per_lang"]["zh"]["accuracy"], 0.9)

    def test_cheap_mode_is_cheapest(self):
        modes = self.report["modes"]
        cheapest = min(m["mean_cost_score"] for m in modes.values())
        self.assertEqual(modes["cheap"]["mean_cost_score"], cheapest)

    def test_quality_mode_is_highest_quality(self):
        modes = self.report["modes"]
        best = max(m["mean_quality_score"] for m in modes.values())
        self.assertEqual(modes["quality"]["mean_quality_score"], best)


if __name__ == "__main__":
    unittest.main()
