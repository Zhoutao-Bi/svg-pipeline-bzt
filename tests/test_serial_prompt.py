import json
import unittest

from run_visual_json_serial import compact_refine_features_for_prompt


class RefinePromptCompactionTests(unittest.TestCase):
    def test_small_feature_json_is_unchanged(self):
        features = {
            "Recognized_Features": [{"ID": "F001"}],
            "Feature_Relationships": [{"Type": "cutting", "Feature_IDs": ["F001", "F002"]}],
        }

        compact, metadata = compact_refine_features_for_prompt(features, max_chars=1000)

        self.assertIs(compact, features)
        self.assertFalse(metadata["applied"])

    def test_oversized_relationships_are_summarized_deterministically(self):
        relationships = [
            {
                "Type": "orthogonal_intersection" if index % 2 else "same_axis_overlap",
                "Feature_IDs": [f"F{index:03d}", f"F{index + 1:03d}"],
                "Intersection_BBox": [index] * 6,
            }
            for index in range(100)
        ]
        features = {
            "Recognized_Features": [{"ID": "F001", "Center_3D": [1, 2, 3]}],
            "Feature_Relationships": relationships,
        }

        compact, metadata = compact_refine_features_for_prompt(
            features,
            max_chars=2200,
            examples_per_type=2,
        )

        self.assertTrue(metadata["applied"])
        self.assertEqual(len(compact["Feature_Relationships"]), 4)
        self.assertEqual(compact["Feature_Relationship_Summary"]["Total_Count"], 100)
        self.assertEqual(
            compact["Feature_Relationship_Summary"]["Counts_By_Type"],
            {"orthogonal_intersection": 50, "same_axis_overlap": 50},
        )
        self.assertLessEqual(
            len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))),
            2200,
        )
        self.assertEqual(len(features["Feature_Relationships"]), 100)


if __name__ == "__main__":
    unittest.main()
