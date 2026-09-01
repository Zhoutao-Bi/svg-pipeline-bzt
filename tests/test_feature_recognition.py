import math
import unittest

from extract_features import ShapeFeatureAnalyzer
from feature_recognition import enrich_feature_data, summarize_feature_data


def feature(axis, center, start, end, diameter, *, shape="Circle", params=None):
    center_key = {"X": "Center_YZ", "Y": "Center_XZ", "Z": "Center_XY"}[axis]
    return {
        "Axis": axis,
        center_key: center,
        "Main_Diameter": diameter,
        "Shape": shape,
        "Shape_Params": params or ({"Diameter": diameter} if shape == "Circle" else {}),
        "Steps": [{
            "Diameter": diameter,
            "Shape": shape,
            "Shape_Params": params or {},
            f"{axis}_Start": start,
            f"{axis}_End": end,
        }],
    }


class ShapeClassificationTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = ShapeFeatureAnalyzer()

    def test_four_sided_profiles_are_not_unknown(self):
        square = self.analyzer.classify_shape([(0, 0), (4, 0), (4, 4), (0, 4)])
        rectangle = self.analyzer.classify_shape([(0, 0), (8, 0), (8, 2), (0, 2)])

        self.assertEqual(square["shape_type"], "Square")
        self.assertEqual(rectangle["shape_type"], "Rectangle")
        self.assertEqual(rectangle["shape_params"]["Length"], 8.0)

    def test_ellipse_is_distinguished_from_capsule(self):
        points = [
            (6 * math.cos(index * math.pi / 24), 2 * math.sin(index * math.pi / 24))
            for index in range(48)
        ]
        ellipse = self.analyzer.classify_shape(points)

        self.assertEqual(ellipse["shape_type"], "Ellipse")
        self.assertAlmostEqual(ellipse["shape_params"]["Major_Diameter"], 12.0, places=1)


class TopologyRecognitionTests(unittest.TestCase):
    def test_overlapping_orthogonal_bores_are_preserved_and_linked(self):
        data = {
            "Part_Overview": {"Bounding_Box_LWH": [4, 4, 4]},
            "Solid_Base_Layers": [],
            "Positive_Pillars": [],
            "Negative_Holes": [
                feature("X", [2, 2], 0, 4, 4),
                feature("Y", [2, 2], 0, 4, 4),
            ],
        }

        enriched = enrich_feature_data(data)

        self.assertEqual(len(enriched["Negative_Holes"]), 2)
        self.assertEqual(len(enriched["Recognized_Features"]), 2)
        relation = enriched["Feature_Relationships"][0]
        self.assertEqual(relation["Type"], "orthogonal_intersection")
        self.assertEqual(relation["Axes"], ["X", "Y"])

    def test_semantic_types_cover_steps_slots_ribs_and_shoulders(self):
        counterbore = feature("Z", [5, 5], 0, 10, 6)
        counterbore["Steps"] = [
            {"Diameter": 6, "Z_Start": 0, "Z_End": 2},
            {"Diameter": 3, "Z_Start": 2, "Z_End": 10},
        ]
        groove = feature(
            "Z", [5, 5], 0, 1, 8,
            shape="Capsule", params={"Length": 8, "Width": 2, "Angle": 0},
        )
        rib = feature(
            "Z", [5, 5], 0, 2, 8,
            shape="Rectangle", params={"Length": 8, "Width": 1, "Angle": 0},
        )
        data = {
            "Part_Overview": {"Bounding_Box_LWH": [10, 10, 10]},
            "Solid_Base_Layers": [
                {"ID": "Base", "Z_Range": [0, 4], "Size_XY": [10, 10]},
                {"ID": "Tier", "Z_Range": [4, 8], "Size_XY": [8, 8]},
            ],
            "Positive_Pillars": [rib],
            "Negative_Holes": [counterbore, groove],
        }

        enriched = enrich_feature_data(data)
        semantic_types = {item["Semantic_Type"] for item in enriched["Recognized_Features"]}

        self.assertEqual(semantic_types, {"Counterbore", "Groove", "Rib"})
        self.assertEqual(enriched["Profile_Transitions"][0]["Semantic_Type"], "Outer_Shoulder")

    def test_slice_spacing_recovers_a_through_hole_at_sampled_boundaries(self):
        data = {
            "Part_Overview": {"Bounding_Box_LWH": [10, 10, 6]},
            "Slice_Metadata": {"Axis_Layer_Spacing": {"Z": 0.25}},
            "Solid_Base_Layers": [],
            "Positive_Pillars": [],
            "Negative_Holes": [feature("Z", [5, 5], 0, 5.75, 4)],
        }

        enriched = enrich_feature_data(data)

        self.assertEqual(enriched["Recognized_Features"][0]["Semantic_Type"], "Through_Hole")

    def test_positive_side_projection_is_retained_but_not_double_counted(self):
        circle = feature("Z", [2, 2], 0, 4, 4)
        side = feature(
            "X", [2, 2], 0, 4, 4,
            shape="Square", params={"Length": 4, "Width": 4, "Angle": 0},
        )
        data = {
            "Part_Overview": {"Bounding_Box_LWH": [4, 4, 4]},
            "Solid_Base_Layers": [],
            "Positive_Pillars": [circle, side],
            "Negative_Holes": [],
        }

        enriched = enrich_feature_data(data)
        summary = summarize_feature_data(enriched)

        self.assertEqual(len(enriched["Recognized_Features"]), 2)
        self.assertEqual(summary["recognized_features"], 1)
        self.assertEqual(summary["projection_evidence"], 1)
        projected = next(item for item in enriched["Recognized_Features"] if item["Role"] == "Projection_Evidence")
        self.assertEqual(projected["Canonical_Feature_ID"], "F001")

    def test_repeated_collinear_holes_form_a_linear_pattern(self):
        holes = [feature("Z", [x, 5], 0, 2, 1) for x in (2, 5, 8)]
        data = {
            "Part_Overview": {"Bounding_Box_LWH": [10, 10, 2]},
            "Solid_Base_Layers": [],
            "Positive_Pillars": [],
            "Negative_Holes": holes,
        }

        enriched = enrich_feature_data(data)
        pattern = enriched["Feature_Patterns"][0]

        self.assertEqual(pattern["Type"], "Linear_Pattern")
        self.assertEqual(pattern["Count"], 3)
        self.assertEqual(pattern["Pitch"], 3.0)
        self.assertTrue(all(item["Pattern_ID"] == pattern["ID"] for item in enriched["Recognized_Features"]))

    def test_summary_is_compact_and_deterministic(self):
        data = {
            "Part_Overview": {"Bounding_Box_LWH": [5, 5, 5]},
            "Solid_Base_Layers": [],
            "Positive_Pillars": [],
            "Negative_Holes": [feature("Z", [2.5, 2.5], 0, 5, 1)],
        }
        summary = summarize_feature_data(enrich_feature_data(data))

        self.assertEqual(summary["semantic_counts"], {"Through_Hole": 1})
        self.assertEqual(summary["recognized_features"], 1)


if __name__ == "__main__":
    unittest.main()
