import tempfile
import unittest
from pathlib import Path

from consensus_fusion import (
    WORKFLOW_KEYS,
    discover_samples,
    fuse_predictions,
    run_consensus,
)


def feature(x, role="装配特征", size=10.0):
    return {
        "特征类型": "孔",
        "特征形状": "圆形",
        "坐标X": x,
        "坐标Y": 20.0,
        "坐标Z": 5.0,
        "尺寸类型": "直径",
        "尺寸数据": size,
        "作用": role,
    }


def prediction(features, description="零件"):
    return {
        "名字": "sample",
        "整体特征": description,
        "尺寸X": 100.0,
        "尺寸Y": 40.0,
        "尺寸Z": 10.0,
        "局部特征列表": features,
    }


class ConsensusFusionTests(unittest.TestCase):
    def setUp(self):
        self.geometry = {"Part_Overview": {"Bounding_Box_LWH": [100.0, 40.0, 10.0]}}

    def test_balanced_keeps_two_workflow_candidate_with_one_assembly_vote(self):
        predictions = {
            "visual_only": prediction([feature(10.0, "轻量化特征")]),
            "visual_json_parallel": prediction([feature(11.0)]),
            "visual_json_serial": prediction([]),
        }
        output, diagnostics = fuse_predictions("part", predictions, self.geometry)
        self.assertEqual(1, len(output["局部特征列表"]))
        self.assertEqual(10.5, output["局部特征列表"][0]["坐标X"])
        self.assertEqual("装配特征", output["局部特征列表"][0]["作用"])
        self.assertEqual(1, diagnostics["kept_clusters"])

    def test_single_workflow_candidate_is_rejected(self):
        predictions = {
            "visual_only": prediction([feature(10.0)]),
            "visual_json_parallel": prediction([]),
            "visual_json_serial": prediction([]),
        }
        output, _ = fuse_predictions("part", predictions, self.geometry)
        self.assertEqual([], output["局部特征列表"])

    def test_cluster_without_assembly_vote_is_rejected(self):
        predictions = {
            "visual_only": prediction([feature(10.0, "轻量化特征")]),
            "visual_json_parallel": prediction([feature(11.0, "其他")]),
            "visual_json_serial": prediction([feature(10.5, "轻量化特征")]),
        }
        output, _ = fuse_predictions("part", predictions, self.geometry)
        self.assertEqual([], output["局部特征列表"])

    def test_precision_requires_visual_and_parallel_support(self):
        predictions = {
            "visual_only": prediction([feature(10.0)]),
            "visual_json_parallel": prediction([]),
            "visual_json_serial": prediction([feature(10.5)]),
        }
        output, _ = fuse_predictions("part", predictions, self.geometry, profile="precision")
        self.assertEqual([], output["局部特征列表"])

        predictions["visual_json_parallel"] = prediction([feature(10.2, "其他")])
        output, _ = fuse_predictions("part", predictions, self.geometry, profile="precision")
        self.assertEqual(1, len(output["局部特征列表"]))

    def test_geometry_bbox_overrides_model_dimensions(self):
        predictions = {workflow: prediction([]) for workflow in WORKFLOW_KEYS}
        output, _ = fuse_predictions("part", predictions, self.geometry)
        self.assertEqual((100.0, 40.0, 10.0), (output["尺寸X"], output["尺寸Y"], output["尺寸Z"]))

    def test_discover_samples_ignores_incomplete_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "easy_1_refined_terra_visual_only.txt").touch()
            (root / "easy_2_refined_luna_visual_only.txt").touch()
            (root / "noise.txt").touch()
            self.assertEqual(["easy_1"], discover_samples(root, "terra"))

    def test_run_consensus_rejects_empty_source(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            with self.assertRaises(FileNotFoundError):
                run_consensus(Path(source), Path(output), "terra", "balanced")


if __name__ == "__main__":
    unittest.main()
