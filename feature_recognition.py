"""Topology-aware semantic recognition for slice-derived CAD features.

The legacy extractor intentionally exposes ``Positive_Pillars`` and
``Negative_Holes``.  They are useful low-level observations, but their names do
not describe slots, pockets, bosses, ribs, stepped holes, or relationships
between observations from different slicing axes.  This module enriches that
compatible representation without requiring a boundary-representation CAD
kernel.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


AXES = ("X", "Y", "Z")
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
CENTER_KEYS = {"X": "Center_YZ", "Y": "Center_XZ", "Z": "Center_XY"}
PROFILE_SHAPES = {"Circle", "Ellipse", "Triangle", "Pentagon", "Hexagon"}
ELONGATED_SHAPES = {"Capsule", "Rectangle"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _part_bbox(data: dict) -> list[float]:
    raw = data.get("Part_Overview", {}).get("Bounding_Box_LWH", [])
    values = [_number(value) for value in raw[:3]]
    return values if len(values) == 3 else [0.0, 0.0, 0.0]


def _sampling_by_axis(data: dict) -> dict[str, float]:
    raw = data.get("Slice_Metadata", {}).get("Axis_Layer_Spacing", {})
    return {
        axis: _number(raw.get(axis))
        for axis in AXES
        if _number(raw.get(axis)) > 0
    }


def _depth_range(feature: dict) -> tuple[float, float]:
    axis = feature.get("Axis")
    if axis not in AXES:
        return 0.0, 0.0
    keys = (f"{axis}_Start", f"{axis}_End")
    values = [
        _number(step.get(key), math.nan)
        for step in feature.get("Steps") or []
        for key in keys
    ]
    values = [value for value in values if math.isfinite(value)]
    return (min(values), max(values)) if values else (0.0, 0.0)


def _plane_center(feature: dict) -> tuple[float, float]:
    values = feature.get(CENTER_KEYS.get(feature.get("Axis"), "")) or []
    return (_number(values[0]), _number(values[1])) if len(values) >= 2 else (0.0, 0.0)


def feature_center_3d(feature: dict) -> list[float]:
    """Return a zero-based XYZ center for a legacy feature."""
    axis = feature.get("Axis")
    plane_a, plane_b = _plane_center(feature)
    depth_start, depth_end = _depth_range(feature)
    depth = (depth_start + depth_end) / 2.0
    if axis == "X":
        return [depth, plane_a, plane_b]
    if axis == "Y":
        return [plane_a, depth, plane_b]
    return [plane_a, plane_b, depth]


def _rotated_extents(length: float, width: float, angle_degrees: float) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    return (
        length * cos_a + width * sin_a,
        length * sin_a + width * cos_a,
    )


def feature_plane_size(feature: dict) -> tuple[float, float]:
    """Return the feature width/height in the slicing plane."""
    shape = feature.get("Shape", "Unknown")
    params = feature.get("Shape_Params") or {}
    fallback = max(_number(feature.get("Main_Diameter")), 0.01)
    if shape == "Circle":
        diameter = max(_number(params.get("Diameter"), fallback), 0.01)
        return diameter, diameter
    if shape == "Ellipse":
        major = max(_number(params.get("Major_Diameter"), fallback), 0.01)
        minor = max(_number(params.get("Minor_Diameter"), fallback), 0.01)
        return _rotated_extents(major, minor, _number(params.get("Angle")))
    if shape in {"Capsule", "Rectangle", "Square"}:
        length = max(_number(params.get("Length"), fallback), 0.01)
        width = max(_number(params.get("Width"), fallback), 0.01)
        return _rotated_extents(length, width, _number(params.get("Angle")))
    diameter = max(
        _number(params.get("Circumcircle_Diameter"), fallback),
        fallback,
        0.01,
    )
    return diameter, diameter


def feature_bbox(feature: dict, *, inflate_zero_depth: bool = True) -> list[float]:
    """Build a shape-aware XYZ AABB for a legacy feature."""
    axis = feature.get("Axis")
    center = feature_center_3d(feature)
    plane_w, plane_h = feature_plane_size(feature)
    start, end = _depth_range(feature)
    if inflate_zero_depth and end - start <= 1e-9:
        half = max(0.005, min(plane_w, plane_h) * 0.01)
        start -= half
        end += half
    if axis == "X":
        return [start, end, center[1] - plane_w / 2, center[1] + plane_w / 2,
                center[2] - plane_h / 2, center[2] + plane_h / 2]
    if axis == "Y":
        return [center[0] - plane_w / 2, center[0] + plane_w / 2, start, end,
                center[2] - plane_h / 2, center[2] + plane_h / 2]
    return [center[0] - plane_w / 2, center[0] + plane_w / 2,
            center[1] - plane_h / 2, center[1] + plane_h / 2, start, end]


def _bbox_intersection(a: list[float], b: list[float]) -> tuple[float, list[float]]:
    box = [
        max(a[0], b[0]), min(a[1], b[1]),
        max(a[2], b[2]), min(a[3], b[3]),
        max(a[4], b[4]), min(a[5], b[5]),
    ]
    lengths = [box[1] - box[0], box[3] - box[2], box[5] - box[4]]
    if any(length <= 0 for length in lengths):
        return 0.0, box
    return lengths[0] * lengths[1] * lengths[2], box


def _bbox_volume(box: list[float]) -> float:
    return (
        max(0.0, box[1] - box[0])
        * max(0.0, box[3] - box[2])
        * max(0.0, box[5] - box[4])
    )


def _diameter_profile(feature: dict) -> list[float]:
    return [
        _number(step.get("Diameter"))
        for step in feature.get("Steps") or []
        if _number(step.get("Diameter")) > 0
    ]


def _diameter_stages(feature: dict, relative_tolerance: float = 0.08) -> int:
    stages: list[float] = []
    for diameter in _diameter_profile(feature):
        if not stages or abs(diameter - stages[-1]) / max(stages[-1], diameter, 1e-9) > relative_tolerance:
            stages.append(diameter)
    return len(stages)


def _boundary_state(
    feature: dict,
    bbox: list[float],
    sampling_by_axis: dict[str, float],
) -> tuple[bool, bool, float]:
    axis = feature.get("Axis")
    extent = bbox[AXIS_INDEX.get(axis, 0)] if axis in AXES else 0.0
    start, end = _depth_range(feature)
    # Slice locations are plane centers, so the first/last observation cannot
    # coincide exactly with a model boundary.  Prefer measured slice spacing;
    # the 30-slice fallback matches the pipeline's coarse cap for old JSON.
    spacing = sampling_by_axis.get(axis, extent / 30.0 if extent > 0 else 0.0)
    tolerance = max(0.05, spacing * 1.25)
    return start <= tolerance, end >= extent - tolerance, extent


def _semantic_type(
    feature: dict,
    polarity: str,
    bbox: list[float],
    sampling_by_axis: dict[str, float],
) -> tuple[str, list[str], float]:
    shape = feature.get("Shape", "Unknown")
    plane_w, plane_h = feature_plane_size(feature)
    aspect = max(plane_w, plane_h) / max(min(plane_w, plane_h), 1e-9)
    start, end = _depth_range(feature)
    touches_min, touches_max, axis_extent = _boundary_state(feature, bbox, sampling_by_axis)
    depth_ratio = (end - start) / max(axis_extent, 1e-9)
    stages = _diameter_stages(feature)
    evidence = [f"{polarity.lower()}_topology", f"{shape.lower()}_profile"]
    if touches_min:
        evidence.append("touches_axis_min")
    if touches_max:
        evidence.append("touches_axis_max")
    if stages > 1:
        evidence.append(f"{stages}_diameter_stages")
    support = len(feature.get("Steps") or [])
    evidence.append(f"supported_by_{support}_step{'s' if support != 1 else ''}")

    if polarity == "Negative":
        if shape in {"Circle", "Ellipse"}:
            if stages >= 2:
                semantic = "Counterbore" if stages == 2 else "Stepped_Hole"
            elif touches_min and touches_max:
                semantic = "Through_Hole"
            elif touches_min or touches_max:
                semantic = "Blind_Hole"
            else:
                semantic = "Internal_Bore"
        elif shape in {"Triangle", "Pentagon", "Hexagon"}:
            semantic = "Polygonal_Through_Hole" if touches_min and touches_max else "Polygonal_Pocket"
        elif shape in ELONGATED_SHAPES and aspect >= 1.35:
            if touches_min and touches_max:
                semantic = "Through_Slot"
            elif depth_ratio <= 0.25 and (touches_min or touches_max):
                semantic = "Groove"
            else:
                semantic = "Blind_Slot"
        elif touches_min and touches_max:
            semantic = "Through_Pocket"
        elif touches_min or touches_max:
            semantic = "Blind_Pocket"
        else:
            semantic = "Internal_Cavity"
    else:
        if shape in {"Circle", "Ellipse"}:
            if stages >= 2:
                semantic = "Stepped_Boss"
            elif depth_ratio >= 0.8:
                semantic = "Cylindrical_Body"
            else:
                semantic = "Boss"
        elif shape in ELONGATED_SHAPES and aspect >= 2.0 and depth_ratio <= 0.5:
            semantic = "Rib"
        elif shape in {"Triangle", "Pentagon", "Hexagon"}:
            semantic = "Prismatic_Boss"
        elif shape in {"Rectangle", "Square"}:
            semantic = "Pad"
        else:
            semantic = "Protrusion"

    confidence = 0.52
    if shape != "Unknown":
        confidence += 0.18
    confidence += min(support, 5) * 0.025
    if touches_min or touches_max:
        confidence += 0.05
    if stages > 1:
        confidence += 0.05
    return semantic, evidence, round(min(confidence, 0.95), 3)


def _same_axis_duplicate(a: dict, b: dict) -> bool:
    if a.get("Axis") != b.get("Axis") or a.get("Shape") != b.get("Shape"):
        return False
    center_distance = math.dist(feature_center_3d(a), feature_center_3d(b))
    size_a, size_b = feature_plane_size(a), feature_plane_size(b)
    size_error = max(
        abs(size_a[i] - size_b[i]) / max(size_a[i], size_b[i], 1e-9)
        for i in range(2)
    )
    box_a, box_b = feature_bbox(a), feature_bbox(b)
    intersection, _ = _bbox_intersection(box_a, box_b)
    overlap = intersection / max(min(_bbox_volume(box_a), _bbox_volume(box_b)), 1e-9)
    return center_distance <= 0.1 and size_error <= 0.03 and overlap >= 0.9


def deduplicate_axis_features(features: Iterable[dict]) -> list[dict]:
    """Remove only near-identical same-axis records.

    Cross-axis overlap is evidence of topology, not evidence that either record
    is a ghost.  This deliberately replaces the old 60%-AABB deletion rule.
    """
    kept: list[dict] = []
    for feature in features:
        duplicate_index = next(
            (index for index, existing in enumerate(kept) if _same_axis_duplicate(feature, existing)),
            None,
        )
        if duplicate_index is None:
            kept.append(feature)
            continue
        old_support = len(kept[duplicate_index].get("Steps") or [])
        new_support = len(feature.get("Steps") or [])
        if new_support > old_support:
            kept[duplicate_index] = feature
    return kept


def _recognized_record(
    feature: dict,
    polarity: str,
    index: int,
    bbox: list[float],
    sampling_by_axis: dict[str, float],
) -> dict:
    semantic, evidence, confidence = _semantic_type(feature, polarity, bbox, sampling_by_axis)
    plane_w, plane_h = feature_plane_size(feature)
    start, end = _depth_range(feature)
    return {
        "ID": f"F{index:03d}",
        "Semantic_Type": semantic,
        "Polarity": polarity,
        "Axis": feature.get("Axis"),
        "Shape": feature.get("Shape", "Unknown"),
        "Center_3D": [round(value, 3) for value in feature_center_3d(feature)],
        "Depth_Range": [round(start, 3), round(end, 3)],
        "Cross_Section_Size": [round(plane_w, 3), round(plane_h, 3)],
        "Main_Diameter": round(_number(feature.get("Main_Diameter")), 3),
        "Shape_Params": copy.deepcopy(feature.get("Shape_Params") or {}),
        "Role": "Canonical_Candidate",
        "Confidence": confidence,
        "Evidence": evidence,
        "Source": feature.pop("_source"),
        "_legacy": feature,
        "_bbox": feature_bbox(feature),
    }


def _relationship_type(a: dict, b: dict) -> str:
    if a["Axis"] == b["Axis"]:
        plane_indices = {"X": (1, 2), "Y": (0, 2), "Z": (0, 1)}[a["Axis"]]
        center_distance = math.dist(
            [a["Center_3D"][index] for index in plane_indices],
            [b["Center_3D"][index] for index in plane_indices],
        )
        scale = max(min(*a["Cross_Section_Size"], *b["Cross_Section_Size"]), 1e-9)
        if center_distance <= scale * 0.1:
            return "coaxial_cut" if a["Polarity"] != b["Polarity"] else "coaxial_overlap"
        return "same_axis_overlap"
    if a["Polarity"] != b["Polarity"]:
        return "cuts" if "Negative" in {a["Polarity"], b["Polarity"]} else "cross_axis_overlap"
    if a["Shape"] in PROFILE_SHAPES and b["Shape"] in PROFILE_SHAPES:
        return "orthogonal_intersection"
    if a["Shape"] in PROFILE_SHAPES or b["Shape"] in PROFILE_SHAPES:
        return "projection_overlap"
    return "cross_axis_overlap"


def _build_relationships(features: list[dict]) -> list[dict]:
    relationships = []
    for index, first in enumerate(features):
        for second in features[index + 1:]:
            intersection, box = _bbox_intersection(first["_bbox"], second["_bbox"])
            if intersection <= 0:
                continue
            volume_a = max(_bbox_volume(first["_bbox"]), 1e-9)
            volume_b = max(_bbox_volume(second["_bbox"]), 1e-9)
            overlap = intersection / min(volume_a, volume_b)
            if overlap < 0.02:
                continue
            relation = _relationship_type(first, second)
            confidence = min(0.98, 0.55 + min(overlap, 1.0) * 0.35)
            relationships.append({
                "Type": relation,
                "Feature_IDs": [first["ID"], second["ID"]],
                "Axes": [first["Axis"], second["Axis"]],
                "Overlap_Ratio": round(overlap, 4),
                "Intersection_BBox": [round(value, 3) for value in box],
                "Confidence": round(confidence, 3),
            })
    return relationships


def _mark_projection_evidence(features: list[dict], relationships: list[dict]) -> None:
    """Mark only high-confidence positive rectangular side projections.

    The records remain in ``Recognized_Features`` for auditability.  Negative
    features are never demoted here because an overlapping hole and slot may be
    two real intersecting voids.
    """
    by_id = {feature["ID"]: feature for feature in features}
    for relationship in relationships:
        if relationship["Type"] != "projection_overlap" or relationship["Overlap_Ratio"] < 0.6:
            continue
        first, second = (by_id[item] for item in relationship["Feature_IDs"])
        if first["Polarity"] != "Positive" or second["Polarity"] != "Positive":
            continue
        primary = first if first["Shape"] in PROFILE_SHAPES else second
        projection = second if primary is first else first
        if primary["Shape"] not in PROFILE_SHAPES or projection["Shape"] not in {"Rectangle", "Square", "Unknown"}:
            continue
        projection["Role"] = "Projection_Evidence"
        projection["Canonical_Feature_ID"] = primary["ID"]
        relationship["Canonical_Feature_ID"] = primary["ID"]
        relationship["Projection_Feature_ID"] = projection["ID"]


def _size_compatible(first: dict, second: dict, tolerance: float = 0.1) -> bool:
    a = sorted(first["Cross_Section_Size"])
    b = sorted(second["Cross_Section_Size"])
    return all(abs(x - y) / max(x, y, 1e-9) <= tolerance for x, y in zip(a, b))


def _cluster_pattern_candidates(features: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for feature in features:
        placed = False
        for group in groups:
            reference = group[0]
            if (
                feature["Semantic_Type"] == reference["Semantic_Type"]
                and feature["Axis"] == reference["Axis"]
                and feature["Shape"] == reference["Shape"]
                and _size_compatible(feature, reference)
            ):
                group.append(feature)
                placed = True
                break
        if not placed:
            groups.append([feature])
    return [group for group in groups if len(group) >= 3]


def _plane_positions(group: list[dict]) -> np.ndarray:
    axis = group[0]["Axis"]
    indices = {"X": (1, 2), "Y": (0, 2), "Z": (0, 1)}[axis]
    return np.array([[feature["Center_3D"][i] for i in indices] for feature in group], dtype=float)


def _classify_pattern(points: np.ndarray) -> tuple[str, dict]:
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] <= 1e-9 or singular[1] / singular[0] <= 0.08:
        order = np.argsort(points[:, np.argmax(np.ptp(points, axis=0))])
        ordered = points[order]
        distances = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
        return "Linear_Pattern", {"Pitch": round(float(np.median(distances)), 3)}
    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius_mean = float(np.mean(radii))
    radius_cv = float(np.std(radii) / max(radius_mean, 1e-9))
    if radius_cv <= 0.08:
        return "Circular_Pattern", {
            "Center": [round(float(value), 3) for value in center],
            "Radius": round(radius_mean, 3),
        }
    return "Repeated_Features", {}


def _build_patterns(features: list[dict]) -> list[dict]:
    patterns = []
    for group in _cluster_pattern_candidates(features):
        pattern_type, params = _classify_pattern(_plane_positions(group))
        pattern_id = f"P{len(patterns) + 1:03d}"
        for feature in group:
            feature["Pattern_ID"] = pattern_id
        patterns.append({
            "ID": pattern_id,
            "Type": pattern_type,
            "Feature_IDs": [feature["ID"] for feature in group],
            "Axis": group[0]["Axis"],
            "Count": len(group),
            **params,
        })
    return patterns


def _solid_transitions(layers: list[dict]) -> list[dict]:
    transitions = []
    ordered = sorted(layers, key=lambda item: _number((item.get("Z_Range") or [0])[0]))
    for first, second in zip(ordered, ordered[1:]):
        size_a, size_b = first.get("Size_XY") or [], second.get("Size_XY") or []
        if len(size_a) < 2 or len(size_b) < 2:
            continue
        delta = [_number(size_b[i]) - _number(size_a[i]) for i in range(2)]
        scale = max(*map(abs, map(_number, size_a)), *map(abs, map(_number, size_b)), 1.0)
        if max(map(abs, delta)) / scale < 0.02:
            continue
        z_range = second.get("Z_Range") or [0.0]
        transitions.append({
            "ID": f"T{len(transitions) + 1:03d}",
            "Semantic_Type": "Outer_Shoulder",
            "Axis": "Z",
            "Position": round(_number(z_range[0]), 3),
            "Size_Before": [round(_number(value), 3) for value in size_a[:2]],
            "Size_After": [round(_number(value), 3) for value in size_b[:2]],
            "Delta_XY": [round(value, 3) for value in delta],
            "Direction": "expands" if sum(delta) > 0 else "contracts",
            "Confidence": 0.75,
            "Source_Layers": [first.get("ID"), second.get("ID")],
        })
    return transitions


def enrich_feature_data(data: dict) -> dict:
    """Return a copy of legacy feature JSON enriched with semantic topology."""
    enriched = copy.deepcopy(data)
    bbox = _part_bbox(enriched)
    sampling_by_axis = _sampling_by_axis(enriched)
    legacy: list[tuple[str, dict, str]] = []
    for key, polarity in (("Positive_Pillars", "Positive"), ("Negative_Holes", "Negative")):
        deduplicated = deduplicate_axis_features(enriched.get(key) or [])
        enriched[key] = deduplicated
        for index, feature in enumerate(deduplicated):
            tagged = copy.deepcopy(feature)
            tagged["_source"] = f"{key}[{index}]"
            legacy.append((polarity, tagged, key))

    recognized = [
        _recognized_record(feature, polarity, index, bbox, sampling_by_axis)
        for index, (polarity, feature, _) in enumerate(legacy, start=1)
    ]
    relationships = _build_relationships(recognized)
    _mark_projection_evidence(recognized, relationships)
    patterns = _build_patterns([
        feature for feature in recognized
        if feature["Role"] != "Projection_Evidence"
    ])
    transitions = _solid_transitions(enriched.get("Solid_Base_Layers") or [])

    for feature in recognized:
        feature.pop("_legacy", None)
        feature.pop("_bbox", None)
    enriched["Feature_Schema_Version"] = "2.0"
    enriched["Recognized_Features"] = recognized
    enriched["Feature_Relationships"] = relationships
    enriched["Feature_Patterns"] = patterns
    enriched["Profile_Transitions"] = transitions
    return enriched


def summarize_feature_data(data: dict) -> dict:
    """Produce a compact deterministic summary for evaluation tooling."""
    semantic_counts: dict[str, int] = defaultdict(int)
    relationship_counts: dict[str, int] = defaultdict(int)
    pattern_counts: dict[str, int] = defaultdict(int)
    observations = data.get("Recognized_Features") or []
    canonical = [feature for feature in observations if feature.get("Role") != "Projection_Evidence"]
    for feature in canonical:
        semantic_counts[feature.get("Semantic_Type", "Unknown")] += 1
    for relationship in data.get("Feature_Relationships") or []:
        relationship_counts[relationship.get("Type", "unknown")] += 1
    for pattern in data.get("Feature_Patterns") or []:
        pattern_counts[pattern.get("Type", "unknown")] += 1
    return {
        "recognized_features": sum(semantic_counts.values()),
        "slice_observations": len(observations),
        "projection_evidence": len(observations) - len(canonical),
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "relationships": sum(relationship_counts.values()),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "patterns": sum(pattern_counts.values()),
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "profile_transitions": len(data.get("Profile_Transitions") or []),
    }
