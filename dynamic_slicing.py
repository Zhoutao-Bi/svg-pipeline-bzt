"""Build an auditable fine-slice plan from the first agent and coarse JSON."""

from __future__ import annotations

import math
from typing import Any


AXES = ("X", "Y", "Z")
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
TYPE_TO_LOCAL_KEY = {"孔": "Negative_Holes", "柱": "Positive_Pillars"}

AGENT_TYPE_ALIASES = {
    "孔": {"hole", "bore", "counterbore", "stepped_hole", "polygonal_hole"},
    "通孔": {"through_hole", "polygonal_through_hole"},
    "盲孔": {"blind_hole"},
    "沉孔": {"counterbore", "stepped_hole"},
    "台阶孔": {"counterbore", "stepped_hole"},
    "柱": {"boss", "cylindrical_body", "stepped_boss", "prismatic_boss"},
    "凸台": {"boss", "stepped_boss", "prismatic_boss", "pad"},
    "槽": {"slot", "groove"},
    "狭槽": {"slot", "groove"},
    "凹槽": {"groove", "pocket", "cavity"},
    "口袋": {"pocket", "cavity"},
    "加强筋": {"rib"},
    "筋": {"rib"},
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bbox(coarse_data: dict) -> list[float]:
    raw = coarse_data.get("Part_Overview", {}).get("Bounding_Box_LWH", [])
    values = [_number(value) for value in raw[:3]]
    if len(values) != 3 or any(value is None or value <= 0 for value in values):
        raise ValueError("粗切 JSON 缺少有效的 Part_Overview.Bounding_Box_LWH")
    return [float(value) for value in values]


def _local_center(feature: dict) -> list[float] | None:
    axis = feature.get("Axis")
    steps = feature.get("Steps") or []
    if axis not in AXES or not steps:
        return None
    start_key, end_key = f"{axis}_Start", f"{axis}_End"
    starts = [_number(step.get(start_key)) for step in steps]
    ends = [_number(step.get(end_key)) for step in steps]
    depths = [value for value in starts + ends if value is not None]
    if not depths:
        return None
    depth = (min(depths) + max(depths)) / 2
    if axis == "X":
        plane = feature.get("Center_YZ") or []
        values = [depth, *plane[:2]]
    elif axis == "Y":
        plane = feature.get("Center_XZ") or []
        values = [plane[0] if len(plane) > 0 else None, depth,
                  plane[1] if len(plane) > 1 else None]
    else:
        plane = feature.get("Center_XY") or []
        values = [*(plane[:2]), depth]
    center = [_number(value) for value in values]
    return [float(value) for value in center] if all(value is not None for value in center) else None


def _agent_center(feature: dict) -> list[float] | None:
    center = [_number(feature.get(f"坐标{axis}")) for axis in AXES]
    return [float(value) for value in center] if all(value is not None for value in center) else None


def _recognized_center(feature: dict) -> list[float] | None:
    center = [_number(value) for value in (feature.get("Center_3D") or [])[:3]]
    return [float(value) for value in center] if len(center) == 3 and all(value is not None for value in center) else None


def _recognized_interval(feature: dict) -> tuple[str, float, float] | None:
    axis = feature.get("Axis")
    values = [_number(value) for value in (feature.get("Depth_Range") or [])[:2]]
    if axis not in AXES or len(values) != 2 or any(value is None for value in values):
        return None
    return axis, min(values), max(values)


def _normalized_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _candidate_matches(agent_type: Any, candidate: dict) -> bool:
    requested = _normalized_type(agent_type)
    semantic = _normalized_type(candidate.get("semantic_type"))
    if not requested:
        return True
    aliases = AGENT_TYPE_ALIASES.get(str(agent_type).strip())
    if aliases:
        return any(alias in semantic for alias in aliases)
    if semantic and (requested in semantic or semantic in requested):
        return True
    # Preserve compatibility with the original two coarse buckets.
    return candidate.get("feature_type") == agent_type


def _local_interval(feature: dict) -> tuple[str, float, float] | None:
    axis = feature.get("Axis")
    if axis not in AXES:
        return None
    start_key, end_key = f"{axis}_Start", f"{axis}_End"
    steps = feature.get("Steps") or []
    values = [
        _number(step.get(key))
        for step in steps
        for key in (start_key, end_key)
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return axis, min(values), max(values)


def _merge_ranges(ranges: list[dict], bbox: list[float]) -> list[dict]:
    merged = []
    for axis in AXES:
        axis_ranges = []
        extent = bbox[AXIS_INDEX[axis]]
        for item in ranges:
            if item["axis"] != axis:
                continue
            start = max(0.0, min(extent, float(item["start"])))
            end = max(0.0, min(extent, float(item["end"])))
            if end < start:
                start, end = end, start
            boundary_tolerance = max(0.05, min(extent * 0.02, 1.0))
            if start <= boundary_tolerance:
                start = 0.0
            if end >= extent - boundary_tolerance:
                end = extent
            if end - start <= 1e-9:
                continue
            axis_ranges.append({**item, "start": start, "end": end})
        axis_ranges.sort(key=lambda item: (item["start"], item["end"]))
        for item in axis_ranges:
            if not merged or merged[-1]["axis"] != axis or item["start"] > merged[-1]["end"] + 1e-9:
                merged.append({
                    "axis": axis,
                    "start": round(item["start"], 3),
                    "end": round(item["end"], 3),
                    "reasons": [item["reason"]],
                })
            else:
                merged[-1]["end"] = round(max(merged[-1]["end"], item["end"]), 3)
                if item["reason"] not in merged[-1]["reasons"]:
                    merged[-1]["reasons"].append(item["reason"])
    return merged


def build_fine_slice_plan(
    first_agent: dict,
    coarse_data: dict,
    *,
    layer_height: float = 0.01,
    range_margin: float = 0.2,
    fallback_half_width: float = 1.0,
    max_total_slices: int = 30000,
    coarse_max_slices: int = 30,
) -> dict:
    """Select full-plane depth ranges at 0.01 mm for the dynamic second pass.

    Geometric features from the first agent are matched to same-type coarse JSON
    features by normalized 3-D center distance.  Selection must not depend on the
    first agent's tentative assembly/lightweight role: that role is deliberately
    re-evaluated by the second agent.  A matched feature uses the full coarse
    depth interval. Unmatched visual features create narrow ranges on all three
    axes so that the plan remains driven by the first agent.
    """
    if layer_height <= 0:
        raise ValueError("layer_height 必须大于 0")
    bbox = _bbox(coarse_data)
    diagonal = max(math.sqrt(sum(value * value for value in bbox)), 1.0)
    all_agent_features = first_agent.get("局部特征列表") or []
    supported_types = set(TYPE_TO_LOCAL_KEY) | set(AGENT_TYPE_ALIASES)
    agent_features = [
        feature for feature in all_agent_features
        if str(feature.get("特征类型") or "").strip() in supported_types
    ]
    if agent_features:
        decision_basis = "first_agent_geometric_features"
    elif all_agent_features:
        agent_features = list(all_agent_features)
        decision_basis = "first_agent_features_fallback_no_supported_type"
    else:
        decision_basis = "coarse_json_fallback_no_agent_features"

    local_candidates = []
    recognized = coarse_data.get("Recognized_Features") or []
    if recognized:
        canonical = [
            (index, feature)
            for index, feature in enumerate(recognized)
            if feature.get("Role") != "Projection_Evidence"
        ]
        candidates = canonical or list(enumerate(recognized))
        for index, feature in candidates:
            center = _recognized_center(feature)
            interval = _recognized_interval(feature)
            if center and interval:
                local_candidates.append({
                    "feature_type": None,
                    "semantic_type": feature.get("Semantic_Type"),
                    "key": "Recognized_Features",
                    "index": index,
                    "feature": feature,
                    "center": center,
                    "interval": interval,
                })
    else:
        for feature_type, key in TYPE_TO_LOCAL_KEY.items():
            for index, feature in enumerate(coarse_data.get(key) or []):
                center = _local_center(feature)
                interval = _local_interval(feature)
                if center and interval:
                    local_candidates.append({
                        "feature_type": feature_type,
                        "semantic_type": "hole" if feature_type == "孔" else "boss",
                        "key": key,
                        "index": index,
                        "feature": feature,
                        "center": center,
                        "interval": interval,
                    })

    raw_ranges = []
    matches = []
    used_candidates = set()
    for agent_index, feature in enumerate(agent_features):
        center = _agent_center(feature)
        feature_type = feature.get("特征类型")
        ranked = []
        if center:
            for candidate_index, candidate in enumerate(local_candidates):
                if candidate_index in used_candidates:
                    continue
                distance = math.dist(center, candidate["center"])
                type_match = _candidate_matches(feature_type, candidate)
                ranked.append((not type_match, distance, candidate_index, candidate))
        if ranked:
            type_fallback, distance, candidate_index, candidate = min(
                ranked, key=lambda item: (item[0], item[1])
            )
            distance_limit = 0.2 if type_fallback else 0.35
            if distance / diagonal <= distance_limit:
                used_candidates.add(candidate_index)
                axis, start, end = candidate["interval"]
                adaptive_margin = max(
                    range_margin,
                    2 * bbox[AXIS_INDEX[axis]] / max(coarse_max_slices, 1),
                )
                raw_ranges.append({
                    "axis": axis,
                    "start": start - adaptive_margin,
                    "end": end + adaptive_margin,
                    "reason": (
                        f"Agent几何特征#{agent_index + 1}"
                        f"{'几何回退匹配' if type_fallback else '类型匹配'}"
                        f"{candidate['key']}[{candidate['index']}]"
                    ),
                })
                matches.append({
                    "agent_feature_index": agent_index,
                    "coarse_json_key": candidate["key"],
                    "coarse_json_index": candidate["index"],
                    "axis": axis,
                    "center_distance": round(distance, 3),
                    "type_match": not type_fallback,
                })
                continue

        if center:
            fallback_axes = []
            for axis, coordinate in zip(AXES, center):
                raw_ranges.append({
                    "axis": axis,
                    "start": coordinate - fallback_half_width,
                    "end": coordinate + fallback_half_width,
                    "reason": f"Agent特征#{agent_index + 1}无可靠粗JSON匹配",
                })
                fallback_axes.append(axis)
            matches.append({
                "agent_feature_index": agent_index,
                "coarse_json_key": None,
                "coarse_json_index": None,
                "axis": fallback_axes,
                "center_distance": None,
            })

    if not raw_ranges:
        # Last-resort behavior is explicit in the plan, rather than silently
        # reverting to a full-model 0.01 mm scan.
        for candidate in local_candidates:
            axis, start, end = candidate["interval"]
            adaptive_margin = max(
                range_margin,
                2 * bbox[AXIS_INDEX[axis]] / max(coarse_max_slices, 1),
            )
            raw_ranges.append({
                "axis": axis,
                "start": start - adaptive_margin,
                "end": end + adaptive_margin,
                "reason": f"无Agent坐标，使用{candidate['key']}[{candidate['index']}]",
            })

    ranges = _merge_ranges(raw_ranges, bbox)
    estimated_slices = sum(
        math.floor((item["end"] - item["start"]) / layer_height) + 1
        for item in ranges
    )
    if not ranges:
        raise ValueError("无法从第一 Agent 结论和粗切 JSON 生成有效细切范围")
    if estimated_slices > max_total_slices:
        raise ValueError(
            f"细切计划预计 {estimated_slices} 层，超过上限 {max_total_slices}；"
            "请缩小 DYNAMIC_RANGE_MARGIN 或提高 DYNAMIC_MAX_FINE_SLICES"
        )

    return {
        "strategy": "full_plane_depth_ranges",
        "decision_basis": decision_basis,
        "layer_height": layer_height,
        "coarse_max_slices": coarse_max_slices,
        "bounding_box_lwh": bbox,
        "ranges": ranges,
        "feature_matches": matches,
        "estimated_slices": estimated_slices,
    }


def ranges_by_axis(plan: dict) -> dict[str, list[list[float]]]:
    """Convert the persisted plan into stl_to_svg's compact range mapping."""
    result = {axis: [] for axis in AXES}
    for item in plan.get("ranges") or []:
        axis = item.get("axis")
        if axis in result:
            result[axis].append([float(item["start"]), float(item["end"])])
    return result
