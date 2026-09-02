import unittest

from tools.evaluate_dynamic_results import evaluate_prediction, summarize


class DynamicEvaluationTests(unittest.TestCase):
    def test_shared_matcher_scores_one_to_one_features(self):
        ground_truth = {
            "尺寸X": 10,
            "尺寸Y": 10,
            "尺寸Z": 10,
            "局部特征列表": [{
                "特征类型": "孔",
                "坐标X": 5,
                "坐标Y": 5,
                "坐标Z": 5,
                "尺寸数据": 2,
            }],
        }
        prediction = {
            "尺寸X": 10,
            "尺寸Y": 10,
            "尺寸Z": 10,
            "局部特征列表": [{
                "特征类型": "孔",
                "坐标X": 5.1,
                "坐标Y": 5,
                "坐标Z": 5,
                "尺寸数据": 2.1,
                "作用": "装配特征",
            }],
        }

        row = evaluate_prediction("easy_1", prediction, ground_truth)
        total = summarize([row])

        self.assertEqual(row["matched"], 1)
        self.assertEqual(total["precision"], 1.0)
        self.assertEqual(total["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
