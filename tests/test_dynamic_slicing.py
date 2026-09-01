import unittest

from dynamic_slicing import build_fine_slice_plan, ranges_by_axis
from stl_to_svg import _normalize_slice_ranges


COARSE = {
    "Part_Overview": {"Bounding_Box_LWH": [50.0, 150.0, 10.0]},
    "Positive_Pillars": [],
    "Negative_Holes": [
        {
            "Axis": "Z",
            "Center_XY": [25.0, 125.0],
            # Coarse max_slices=30 leaves the last sampled plane at 9.67 mm.
            # The adaptive margin must still recover the full 0-10 mm range.
            "Steps": [{"Z_Start": 0.0, "Z_End": 9.67}],
        }
    ],
}


class DynamicSlicePlanTests(unittest.TestCase):
    def test_matches_assembly_hole_to_full_depth_interval(self):
        first_agent = {
            "局部特征列表": [{
                "特征类型": "孔",
                "坐标X": 25.2,
                "坐标Y": 124.8,
                "坐标Z": 5.0,
                "作用": "装配特征",
            }]
        }
        plan = build_fine_slice_plan(first_agent, COARSE, range_margin=0.2)

        self.assertEqual(plan["decision_basis"], "first_agent_assembly_features")
        self.assertEqual(plan["ranges"], [{
            "axis": "Z",
            "start": 0.0,
            "end": 10.0,
            "reasons": ["Agent装配特征#1匹配Negative_Holes[0]"],
        }])
        self.assertEqual(ranges_by_axis(plan)["Z"], [[0.0, 10.0]])
        self.assertEqual(plan["estimated_slices"], 1001)

    def test_unmatched_visual_feature_uses_narrow_ranges_on_each_axis(self):
        first_agent = {
            "局部特征列表": [{
                "特征类型": "槽",
                "坐标X": 10.0,
                "坐标Y": 20.0,
                "坐标Z": 5.0,
                "作用": "装配特征",
            }]
        }
        plan = build_fine_slice_plan(first_agent, COARSE, fallback_half_width=1.0)

        self.assertEqual([item["axis"] for item in plan["ranges"]], ["X", "Y", "Z"])
        self.assertEqual(ranges_by_axis(plan), {
            "X": [[9.0, 11.0]],
            "Y": [[19.0, 21.0]],
            "Z": [[4.0, 6.0]],
        })

    def test_rejects_plan_over_safety_limit(self):
        first_agent = {"局部特征列表": []}
        with self.assertRaisesRegex(ValueError, "超过上限"):
            build_fine_slice_plan(first_agent, COARSE, max_total_slices=100)

    def test_matches_slot_to_richer_recognized_feature(self):
        coarse = {
            "Part_Overview": {"Bounding_Box_LWH": [50.0, 150.0, 10.0]},
            "Recognized_Features": [{
                "Semantic_Type": "Through_Slot",
                "Axis": "Z",
                "Center_3D": [10.0, 20.0, 5.0],
                "Depth_Range": [4.0, 6.0],
            }],
        }
        first_agent = {
            "局部特征列表": [{
                "特征类型": "槽",
                "坐标X": 10.0,
                "坐标Y": 20.0,
                "坐标Z": 5.0,
                "作用": "装配特征",
            }]
        }

        plan = build_fine_slice_plan(first_agent, coarse, range_margin=0.2)

        self.assertEqual(plan["feature_matches"][0]["coarse_json_key"], "Recognized_Features")
        self.assertEqual(plan["ranges"][0]["axis"], "Z")
        self.assertEqual(plan["ranges"][0]["reasons"], ["Agent装配特征#1匹配Recognized_Features[0]"])

    def test_ignores_projection_evidence_when_matching_richer_features(self):
        coarse = {
            "Part_Overview": {"Bounding_Box_LWH": [20.0, 20.0, 20.0]},
            "Recognized_Features": [
                {
                    "Semantic_Type": "Pad",
                    "Role": "Projection_Evidence",
                    "Axis": "X",
                    "Center_3D": [10.0, 10.0, 10.0],
                    "Depth_Range": [8.0, 12.0],
                },
                {
                    "Semantic_Type": "Boss",
                    "Role": "Canonical_Candidate",
                    "Axis": "Z",
                    "Center_3D": [10.2, 10.0, 10.0],
                    "Depth_Range": [7.0, 13.0],
                },
            ],
        }
        first_agent = {
            "局部特征列表": [{
                "特征类型": "凸台",
                "坐标X": 10.0,
                "坐标Y": 10.0,
                "坐标Z": 10.0,
                "作用": "装配特征",
            }]
        }

        plan = build_fine_slice_plan(first_agent, coarse)

        self.assertEqual(plan["feature_matches"][0]["coarse_json_index"], 1)
        self.assertEqual(plan["feature_matches"][0]["axis"], "Z")


class SliceRangeTests(unittest.TestCase):
    def test_clamps_sorts_and_merges_ranges(self):
        self.assertEqual(
            _normalize_slice_ranges([[8, 12], [-1, 2], [1.5, 3]], 10),
            [[0.0, 3.0], [8.0, 10]],
        )


if __name__ == "__main__":
    unittest.main()
