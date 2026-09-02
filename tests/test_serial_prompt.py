import json
import unittest

from run_visual_json_serial import (
    build_dynamic_refine_user_prompt,
    compact_refine_features_for_prompt,
    ensure_cross_scale_hole_patterns,
)


class RefinePromptCompactionTests(unittest.TestCase):
    def test_cross_scale_hole_pattern_restores_missing_members(self):
        def evidence(offset):
            holes = []
            for x in (2, 5, 8):
                holes.append({
                    "Axis": "Z",
                    "Center_XY": [x + offset, 5],
                    "Main_Diameter": 2,
                    "Shape": "Circle",
                    "Shape_Params": {"Diameter": 2},
                    "Steps": [{"Diameter": 2, "Z_Start": 0, "Z_End": 2}],
                })
            return {
                "Part_Overview": {"Bounding_Box_LWH": [20, 20, 2]},
                "Positive_Pillars": [],
                "Negative_Holes": holes,
            }

        prediction = {
            "尺寸X": 20,
            "尺寸Y": 20,
            "尺寸Z": 2,
            "局部特征列表": [{
                "特征类型": "孔", "特征形状": "圆形",
                "坐标X": 2, "坐标Y": 5, "坐标Z": 1,
                "尺寸类型": "直径", "尺寸数据": 2, "作用": "装配特征",
            }],
            "整体特征": "test",
        }

        validated, metadata = ensure_cross_scale_hole_patterns(
            prediction,
            evidence(0),
            evidence(0.2),
        )

        self.assertEqual(metadata["matched_cross_scale_patterns"], 1)
        self.assertEqual(metadata["appended_feature_count"], 2)
        self.assertEqual(len(validated["局部特征列表"]), 3)

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

    def test_dynamic_prompt_reenriches_and_includes_coarse_evidence(self):
        duplicate = {
            "Axis": "Z",
            "Center_XY": [5, 5],
            "Main_Diameter": 2,
            "Shape": "Circle",
            "Shape_Params": {"Diameter": 2},
            "Steps": [{"Diameter": 2, "Z_Start": 0, "Z_End": 10}],
        }
        geometry = {
            "Part_Overview": {"Bounding_Box_LWH": [10, 10, 10]},
            "Slice_Metadata": {"Axis_Layer_Spacing": {"Z": 0.01}},
            "Positive_Pillars": [],
            "Negative_Holes": [duplicate, {**duplicate, "Center_XY": [5.2, 5.1]}],
            "Recognized_Features": [{"ID": "stale-1"}, {"ID": "stale-2"}],
        }

        prompt, metadata = build_dynamic_refine_user_prompt(
            {"局部特征列表": []},
            {"ranges": []},
            geometry,
            geometry,
        )
        payload = json.loads(prompt)

        self.assertIn("0.1mm粗切特征摘要", payload)
        self.assertEqual(
            len(payload["0.01mm细切特征摘要"]["Recognized_Features"]),
            1,
        )
        self.assertNotIn("Negative_Holes", payload["0.01mm细切特征摘要"])
        self.assertEqual(metadata["fine"]["recognized_features_before"], 2)
        self.assertEqual(metadata["fine"]["recognized_features_after"], 1)


if __name__ == "__main__":
    unittest.main()
