import unittest

from tools.analyze_ter71_experiments import (
    CaseScore,
    paired_bootstrap,
    summarize,
)


class Ter71AnalysisTests(unittest.TestCase):
    def test_micro_summary_uses_feature_counts(self) -> None:
        rows = [
            CaseScore("easy_1", "easy", predicted=2, expected=1, matched=1),
            CaseScore("hard_1", "hard", predicted=1, expected=3, matched=1),
        ]
        result = summarize(rows)
        self.assertEqual(result.samples, 2)
        self.assertEqual(result.matched, 2)
        self.assertAlmostEqual(result.precision, 2 / 3)
        self.assertAlmostEqual(result.recall, 1 / 2)
        self.assertAlmostEqual(result.f1, 4 / 7)

    def test_paired_bootstrap_is_deterministic_and_detects_dominance(self) -> None:
        baseline = {
            "easy_1": CaseScore("easy_1", "easy", 1, 1, 0),
            "hard_1": CaseScore("hard_1", "hard", 2, 2, 1),
        }
        treatment = {
            "easy_1": CaseScore("easy_1", "easy", 1, 1, 1),
            "hard_1": CaseScore("hard_1", "hard", 2, 2, 2),
        }
        first = paired_bootstrap(baseline, treatment, replicates=500, seed=7)
        second = paired_bootstrap(baseline, treatment, replicates=500, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first["ci95_low"], 0)
        self.assertEqual(first["matched_wins"], 2)
        self.assertEqual(first["matched_losses"], 0)


if __name__ == "__main__":
    unittest.main()
