"""Fuse the three ablation outputs into one evidence-backed prediction.

The three model workflows make partially independent errors.  This module
clusters compatible features across workflows and keeps only features with
cross-workflow support.  It does not call a model, so it can be run after the
existing ablation modes at negligible cost.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from codex_client import CODEX_MODEL, CODEX_MODEL_TAG
from pipeline import DEFAULT_RESULTS_DIR


WORKFLOW_KEYS = (
    "visual_only",
    "visual_json_parallel",
    "visual_json_serial",
)
TARGET_TYPES = {"孔", "柱", "槽", "倒角"}
ASSEMBLY_ROLE = "装配特征"
TYPE_ORDER = {"孔": 0, "柱": 1, "槽": 2, "倒角": 3}

DEFAULT_COORDINATE_TOLERANCE = float(
    os.getenv("CONSENSUS_COORDINATE_TOLERANCE", "0.05")
)
DEFAULT_SIZE_TOLERANCE = float(os.getenv("CONSENSUS_SIZE_TOLERANCE", "0.5"))


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _coordinates(feature: dict) -> tuple[float, float, float] | None:
    values = tuple(_number(feature.get(key)) for key in ("坐标X", "坐标Y", "坐标Z"))
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _coordinate_distance(left: dict, right: dict, diagonal: float) -> float:
    left_coordinates = _coordinates(left)
    right_coordinates = _coordinates(right)
    if left_coordinates is None or right_coordinates is None:
        return math.inf
    distance = math.sqrt(
        sum((left_value - right_value) ** 2 for left_value, right_value in zip(left_coordinates, right_coordinates))
    )
    return distance / max(diagonal, 1.0)


def _relative_size_distance(left: dict, right: dict) -> float:
    left_size = _number(left.get("尺寸数据"))
    right_size = _number(right.get("尺寸数据"))
    if left_size is None or right_size is None:
        return math.inf
    return abs(left_size - right_size) / max(min(abs(left_size), abs(right_size)), 1.0)


def _median(values: Iterable[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return statistics.median(available) if available else None


@dataclass
class FeatureCluster:
    members: dict[str, dict] = field(default_factory=dict)

    @property
    def support(self) -> int:
        return len(self.members)

    @property
    def assembly_votes(self) -> int:
        return sum(feature.get("作用") == ASSEMBLY_ROLE for feature in self.members.values())

    def representative(self) -> dict:
        """Return a stable median representative without inventing categories."""
        preferred = next(
            (
                self.members[key]
                for key in ("visual_json_serial", "visual_json_parallel", "visual_only")
                if key in self.members
            ),
            next(iter(self.members.values())),
        )
        output = dict(preferred)
        for key in ("坐标X", "坐标Y", "坐标Z", "尺寸数据"):
            value = _median(_number(feature.get(key)) for feature in self.members.values())
            if value is not None:
                output[key] = value
        output["作用"] = ASSEMBLY_ROLE
        return output


def _part_dimensions(local_geometry: dict, predictions: dict[str, dict]) -> tuple[float, float, float]:
    bounding_box = local_geometry.get("Part_Overview", {}).get("Bounding_Box_LWH")
    if isinstance(bounding_box, list) and len(bounding_box) == 3:
        values = tuple(_number(value) for value in bounding_box)
        if all(value is not None for value in values):
            return values  # type: ignore[return-value]

    dimensions = []
    for key in ("尺寸X", "尺寸Y", "尺寸Z"):
        value = _median(_number(prediction.get(key)) for prediction in predictions.values())
        if value is None:
            raise ValueError(f"无法确定整体尺寸字段 {key}")
        dimensions.append(value)
    return tuple(dimensions)  # type: ignore[return-value]


def cluster_features(
    predictions: dict[str, dict],
    dimensions: tuple[float, float, float],
    coordinate_tolerance: float = DEFAULT_COORDINATE_TOLERANCE,
    size_tolerance: float = DEFAULT_SIZE_TOLERANCE,
) -> list[FeatureCluster]:
    """Greedily form one-feature-per-workflow consensus clusters."""
    diagonal = math.sqrt(sum(value * value for value in dimensions))
    clusters: list[FeatureCluster] = []

    for workflow in WORKFLOW_KEYS:
        prediction = predictions.get(workflow, {})
        for feature in prediction.get("局部特征列表", []):
            if feature.get("特征类型") not in TARGET_TYPES:
                continue

            matches = []
            for cluster in clusters:
                if workflow in cluster.members:
                    continue
                representative = cluster.representative()
                if representative.get("特征类型") != feature.get("特征类型"):
                    continue
                coordinate_distance = _coordinate_distance(feature, representative, diagonal)
                size_distance = _relative_size_distance(feature, representative)
                if coordinate_distance <= coordinate_tolerance and size_distance <= size_tolerance:
                    matches.append((coordinate_distance + 0.1 * size_distance, cluster))

            if matches:
                min(matches, key=lambda item: item[0])[1].members[workflow] = feature
            else:
                clusters.append(FeatureCluster({workflow: feature}))
    return clusters


def _keep_cluster(cluster: FeatureCluster, profile: str) -> bool:
    if cluster.assembly_votes < 1:
        return False
    if profile == "balanced":
        return cluster.support >= 2
    if profile == "precision":
        return {"visual_only", "visual_json_parallel"}.issubset(cluster.members)
    raise ValueError(f"未知共识配置: {profile!r}；可选 balanced 或 precision")


def fuse_predictions(
    sample: str,
    predictions: dict[str, dict],
    local_geometry: dict,
    profile: str = "balanced",
    coordinate_tolerance: float = DEFAULT_COORDINATE_TOLERANCE,
    size_tolerance: float = DEFAULT_SIZE_TOLERANCE,
) -> tuple[dict, dict]:
    """Fuse one sample and return ``(prediction, diagnostics)``."""
    missing = [workflow for workflow in WORKFLOW_KEYS if workflow not in predictions]
    if missing:
        raise ValueError(f"{sample} 缺少流程输出: {', '.join(missing)}")

    dimensions = _part_dimensions(local_geometry, predictions)
    clusters = cluster_features(
        predictions,
        dimensions,
        coordinate_tolerance=coordinate_tolerance,
        size_tolerance=size_tolerance,
    )
    kept = [cluster for cluster in clusters if _keep_cluster(cluster, profile)]
    features = [cluster.representative() for cluster in kept]
    features.sort(
        key=lambda feature: (
            TYPE_ORDER.get(feature.get("特征类型"), 99),
            *(_coordinates(feature) or (math.inf, math.inf, math.inf)),
        )
    )

    description = next(
        (
            predictions[workflow].get("整体特征")
            for workflow in ("visual_json_serial", "visual_json_parallel", "visual_only")
            if predictions[workflow].get("整体特征")
        ),
        "",
    )
    output = {
        "名字": sample,
        "整体特征": description,
        "尺寸X": dimensions[0],
        "尺寸Y": dimensions[1],
        "尺寸Z": dimensions[2],
        "局部特征列表": features,
    }
    diagnostics = {
        "profile": profile,
        "clusters": len(clusters),
        "kept_clusters": len(kept),
        "discarded_clusters": len(clusters) - len(kept),
        "workflow_feature_counts": {
            workflow: len(predictions[workflow].get("局部特征列表", []))
            for workflow in WORKFLOW_KEYS
        },
    }
    return output, diagnostics


def _load_sample(source_dir: Path, sample: str, model_tag: str) -> tuple[dict[str, dict], dict]:
    predictions = {}
    for workflow in WORKFLOW_KEYS:
        path = source_dir / f"{sample}_refined_{model_tag}_{workflow}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        predictions[workflow] = json.loads(path.read_text(encoding="utf-8"))

    geometry_path = source_dir / f"{sample}_features.json"
    if not geometry_path.is_file():
        raise FileNotFoundError(geometry_path)
    return predictions, json.loads(geometry_path.read_text(encoding="utf-8"))


def discover_samples(source_dir: Path, model_tag: str) -> list[str]:
    suffix = f"_refined_{model_tag}_visual_only.txt"
    return sorted(path.name[: -len(suffix)] for path in source_dir.glob(f"*{suffix}"))


def run_consensus(
    source_dir: Path,
    output_dir: Path,
    model_tag: str,
    profile: str,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(source_dir, model_tag)
    if not samples:
        raise FileNotFoundError(
            f"在 {source_dir} 未找到 *_refined_{model_tag}_visual_only.txt"
        )
    rows = []
    for sample in samples:
        started_at = time.perf_counter()
        try:
            predictions, geometry = _load_sample(source_dir, sample, model_tag)
            output, diagnostics = fuse_predictions(sample, predictions, geometry, profile=profile)
            destination = output_dir / f"{sample}_refined_{model_tag}_consensus_{profile}.txt"
            destination.write_text(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            rows.append(
                {
                    "base_name": sample,
                    "fusion_time_s": round(time.perf_counter() - started_at, 6),
                    "status": "OK",
                    "error": "",
                    **diagnostics,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "base_name": sample,
                    "fusion_time_s": round(time.perf_counter() - started_at, 6),
                    "status": "FAIL",
                    "error": str(exc),
                    "profile": profile,
                    "clusters": 0,
                    "kept_clusters": 0,
                    "discarded_clusters": 0,
                    "workflow_feature_counts": {},
                }
            )

    metrics_path = output_dir / f"metrics_consensus_{profile}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "base_name",
            "fusion_time_s",
            "status",
            "error",
            "profile",
            "clusters",
            "kept_clusters",
            "discarded_clusters",
            "workflow_feature_counts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["workflow_feature_counts"] = json.dumps(
                serialized["workflow_feature_counts"], ensure_ascii=False, separators=(",", ":")
            )
            writer.writerow(serialized)
    return rows


def main() -> None:
    source_dir = Path(os.getenv("CONSENSUS_SOURCE_DIR", DEFAULT_RESULTS_DIR)).resolve()
    output_dir = Path(os.getenv("RESULTS_DIR", DEFAULT_RESULTS_DIR)).resolve()
    profile = os.getenv("CONSENSUS_PROFILE", "balanced").strip().lower()
    rows = run_consensus(source_dir, output_dir, CODEX_MODEL_TAG, profile)
    success = sum(row["status"] == "OK" for row in rows)
    failed = len(rows) - success
    print(f"共识融合: 模型={CODEX_MODEL} profile={profile} 样本={len(rows)} 成功={success} 失败={failed}")
    print(f"来源: {source_dir}")
    print(f"输出: {output_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
