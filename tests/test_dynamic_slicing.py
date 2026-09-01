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


class SliceRangeTests(unittest.TestCase):
    def test_clamps_sorts_and_merges_ranges(self):
        self.assertEqual(
            _normalize_slice_ranges([[8, 12], [-1, 2], [1.5, 3]], 10),
            [[0.0, 3.0], [8.0, 10]],
        )


if __name__ == "__main__":
    unittest.main()
