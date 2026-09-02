"""
消融实验 - 有串行 JSON（视觉 → JSON 矫正）

阶段1: 批量切片 → 阶段2: 并发两轮 Codex 分析。CSV 实时追加。
"""

import os
import sys
import time
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from feature_recognition import enrich_feature_data

from pipeline import (
    get_stl_files, run_pipeline, get_local_data, save_result,
    append_csv_row, run_dynamic_refinement, DEFAULT_RESULTS_DIR, SLICE_MODE,
)
from codex_client import (
    call_codex_vision, call_codex_text, ensure_codex_oauth,
    CODEX_MODEL, CODEX_REASONING_EFFORT, CodexCallError,
)

MAX_WORKERS = int(os.getenv("CODEX_CONCURRENCY", "1"))
REFINE_MAX_FEATURE_JSON_CHARS = int(
    os.getenv("REFINE_MAX_FEATURE_JSON_CHARS", "900000")
)
REFINE_RELATIONSHIP_EXAMPLES_PER_TYPE = int(
    os.getenv("REFINE_RELATIONSHIP_EXAMPLES_PER_TYPE", "8")
)

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "名字": {"type": "string"},
        "尺寸X": {"description": "零件整体包络尺寸 Length (X)", "type": "number"},
        "尺寸Y": {"description": "零件整体包络尺寸 Width (Y)", "type": "number"},
        "尺寸Z": {"description": "零件整体包络尺寸 Height (Z)", "type": "number"},
        "局部特征列表": {
            "description": "按照孔、柱、槽、圆角的顺序依次列出",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "特征类型": {"description": "只能回答以下其中一个词语：孔、柱、槽、倒角", "type": "string"},
                    "特征形状": {"description": "只能回答以下其中一个词语：圆形、类圆形、三角形、四边形、五边形、六边形、多边形、其他。", "type": "string"},
                    "坐标X": {"type": "number"},
                    "坐标Y": {"type": "number"},
                    "坐标Z": {"type": "number"},
                    "尺寸类型": {"description": "只能回答以下其中一个词语：直径、边长、角度", "type": "string"},
                    "尺寸数据": {"type": "number"},
                    "作用": {"description": "只能回答以下其中一个词语：装配特征、轻量化特征、其他", "type": "string"},
                },
                "required": ["特征类型", "特征形状", "坐标X", "坐标Y", "坐标Z", "尺寸类型", "尺寸数据", "作用"],
                "additionalProperties": False,
            },
        },
        "整体特征": {"description": "他的整体的几何形状，各个特征，他是啥，可能是干嘛的，有啥用。", "type": "string"},
    },
    "required": ["名字", "整体特征", "尺寸X", "尺寸Y", "尺寸Z", "局部特征列表"],
    "additionalProperties": False,
}

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "尺寸X": {"description": "零件整体包络尺寸 Length (X)", "type": "number"},
        "尺寸Y": {"description": "零件整体包络尺寸 Width (Y)", "type": "number"},
        "尺寸Z": {"description": "零件整体包络尺寸 Height (Z)", "type": "number"},
        "局部特征列表": {
            "description": "按照孔、柱、槽、圆角的顺序依次列出",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "特征类型": {"description": "只能回答以下其中一个词语：孔、柱、槽、倒角", "type": "string"},
                    "特征形状": {"description": "只能回答以下其中一个词语：圆形、类圆形、三角形、四边形、五边形、六边形、多边形、其他。", "type": "string"},
                    "坐标X": {"type": "number"},
                    "坐标Y": {"type": "number"},
                    "坐标Z": {"type": "number"},
                    "尺寸类型": {"description": "只能回答以下其中一个词语：直径、边长、角度", "type": "string"},
                    "尺寸数据": {"type": "number"},
                    "作用": {"description": "只能回答以下其中一个词语：装配特征、轻量化特征、其他", "type": "string"},
                },
                "required": ["特征类型", "特征形状", "坐标X", "坐标Y", "坐标Z", "尺寸类型", "尺寸数据", "作用"],
                "additionalProperties": False,
            },
        },
        "整体特征": {"description": "他的整体的几何形状", "type": "string"},
    },
    "required": ["整体特征", "尺寸X", "尺寸Y", "尺寸Z", "局部特征列表"],
    "additionalProperties": False,
}

VISION_SYSTEM_PROMPT = r"""System Prompt: 你是一位资深机器视觉与逆向工程专家。你的任务是仅依靠提供的粗切多视图深度图（X/Y/Z轴），对该机械零件进行视觉特征提取，并直接输出可填入结构化表格的数据。
分析思路与核心准则:
1、纯视觉依赖: 你需要通过观察深度图的色阶映射（黄向紫代表浅入深）和三视图轮廓，提取零件的几何形态。
2、重复判断：由于三视图问题，注意判断是否有重复和重叠的局部特征。
3、包络估算: 观察长宽高比例，估测全局包络尺寸（请提供基于视觉比例的数值估算）。
4、视觉提示: 寻找深度图上的色阶突变区域或穿透区域。重点识别并列出：孔、柱、槽、倒角。
形态判定: 利用视觉优势，直接判定连续的颜色渐变为【平滑曲面/倒角/锥面】。准确分辨孔、柱、槽的真实形状。
特征判别：必须谨慎判断每个特征属于装配、轻量化或其他。只有实际参与定位、连接、配合、夹持的结构才标记为装配特征；坐标用于下一步动态细切选区，需尽量准确。
输出约束: 严格按照提供的 JSON Schema 输出。局部特征必须按照孔、柱、槽、圆角的顺序输出。"""

VISION_USER_PROMPT = "结构化输出。"

REFINE_SYSTEM_PROMPT = r"""Role: 你是一位资深机械设计工程师和 CAD 数据分析专家。你的任务是联合粗切/细切深度图、几何 JSON 与第一 Agent 的粗切视觉结论，对数据进行最终矫正。
分析思路与核心准则:
1、证据优先级：Part_Overview.Bounding_Box_LWH 是整体尺寸真值来源；Canonical_Candidate 的 Center_3D、Depth_Range、Cross_Section_Size 是局部数值的首选来源；细切图负责验证形状和用途；第一 Agent 的数值与用途只是待复核假设。
2、语义映射：Through/Blind/Internal/Stepped/Counterbore Hole 或 Bore 输出为“孔”；封闭且贯穿零件的长圆/胶囊/多边形开口（Through_Slot）在旧输出模式中也是“孔”；只有开边、非贯穿或去料形成的 Blind_Slot、Groove、Pocket 才输出为“槽”；Boss、Stepped_Boss、Cylindrical_Body、Pad、Rib 输出为“柱”。不要因为 JSON 使用了更细的英文语义而漏掉旧输出模式中的孔/柱/槽。
3、装配用途：装配项必须是离散、可定位的连接/配合几何，并获得至少两类相互一致的证据（粗图、细图、第一 Agent、粗/细 JSON 中任意两类）。封闭贯穿孔、盲孔、沉孔、定位孔、成组紧固孔和长圆调节孔在满足该条件且无明确反证时优先标记“装配特征”；单个封闭贯穿孔不能仅凭孔径较大就判为减重，只有重复减重阵列或明确的大型开放空腔才标记轻量化。外轮廓主体、贯穿全长且没有直径台阶的 Cylindrical_Body、法兰盘、支撑肋、底座或 Pad 默认属于“其他”，不要仅因 JSON 名称含 Boss/柱就报为装配特征；但轴对称阶梯轴上由明确肩部分隔的各个外径段属于离散配合轴颈，应保留为装配“柱”。仅在单个细切视图出现、而粗图和粗 JSON 均不支持的候选不得标记为装配。若第一 Agent 和粗切三视图确认零件是无横向开孔的轴对称/回转体，则所有偏离回转轴的细切孔/柱都视为截面伪影并删除，即使 JSON 在相邻层重复检测到它们；只有粗图或第一 Agent 明确显示横向孔时才能破除此规则。第二 Agent 必须独立复核第一 Agent 的“作用”，不能机械继承。
4、坐标与尺寸规则：整体尺寸直接复制 JSON 包络。局部特征坐标使用 JSON 的三维中心；沿切片轴的坐标是 Depth_Range 中点。只有 Depth_Range 实际覆盖该轴包络至少 80% 时才可把该轴坐标取包络中点；短盲孔、浅孔或局部槽必须保留真实 Depth_Range 中点，严禁仅因图像投影看似贯穿就移到零件中心。若选区是内部窄窗且特征贯穿整个选区，需再用粗切候选判断真实深度，不能把选区边界误当零件边界。阶梯孔/沉孔/阶梯轴必须查看 Profile_Summary.Dominant_Diameter_Stages：每个主导阶段的 Depth_Range 中点就是该段轴向坐标，Diameter 是该段尺寸；装配孔的“尺寸数据”优先填真实配合小径而非沉孔大径或外包络；同轴外柱/法兰只有满足第3条的离散配合条件才作为独立装配“柱”。
5、重复与阵列判断：同一 Canonical_Feature_ID 或同轴高重叠观测只输出一次；沉孔、扩孔、阶梯孔的多个直径阶段属于同一个孔，只输出一个装配“孔”（取真实配合小径），不得把入口扩孔与主孔拆成两个实体。Projection_Evidence 只是侧视证据，不作为独立实体。不同轴的真实相交孔/槽仍是不同实体，不得因 AABB 重叠删除。若同一 Count>=3 的圆孔阵列在粗切与细切 Feature_Patterns 中都出现，它是高置信装配阵列，必须按各 Canonical Feature 的真实中心逐个保留；只有细切单边出现的阵列仍按第3条审慎处理。
6、保留与有效性规则：第一 Agent 已确认的装配特征应保留，除非粗/细 JSON 或细图有明确反证；细切选区未覆盖不构成删除理由。输出前逐项检查类型、用途、XYZ 坐标、尺寸和重复项。所有坐标必须位于 Part_Overview.Bounding_Box_LWH 对应范围内；若细切候选中心越界或与粗切中心发生整体平移，应使用粗切 JSON/第一 Agent 中位于包络内且与图像一致的中心，禁止输出越界坐标。"""


PROMPT_FEATURE_KEYS = (
    "Part_Overview",
    "Slice_Metadata",
    "Recognized_Features",
    "Feature_Relationships",
    "Feature_Patterns",
    "Profile_Transitions",
)


def prepare_feature_evidence_for_prompt(features: dict) -> tuple[dict, dict]:
    """Re-enrich archived JSON and remove duplicated legacy payload sections."""
    enriched = enrich_feature_data(features)
    evidence = {
        key: enriched[key]
        for key in PROMPT_FEATURE_KEYS
        if key in enriched
    }
    evidence["Legacy_Feature_Counts"] = {
        "Positive_Pillars": len(enriched.get("Positive_Pillars") or []),
        "Negative_Holes": len(enriched.get("Negative_Holes") or []),
    }
    compact, metadata = compact_refine_features_for_prompt(evidence)
    metadata.update({
        "recognized_features_before": len(features.get("Recognized_Features") or []),
        "recognized_features_after": len(enriched.get("Recognized_Features") or []),
        "canonical_features_after": sum(
            feature.get("Role") != "Projection_Evidence"
            for feature in enriched.get("Recognized_Features") or []
        ),
    })
    return compact, metadata


def compact_refine_features_for_prompt(
    features: dict,
    *,
    max_chars: int = REFINE_MAX_FEATURE_JSON_CHARS,
    examples_per_type: int = REFINE_RELATIONSHIP_EXAMPLES_PER_TYPE,
) -> tuple[dict, dict]:
    """Bound the second-agent JSON without changing archived geometry data.

    Topology recognition can create O(n²) pairwise relationships.  The full list is
    useful as an artifact, but sending thousands of repetitive pairs to Codex can
    exceed the CLI's one-million-character input limit.  Preserve all recognized
    features and legacy coordinates, while replacing only an oversized relationship
    list with deterministic per-type counts and representative examples.
    """
    original_chars = len(json.dumps(features, ensure_ascii=False, separators=(",", ":")))
    relationships = features.get("Feature_Relationships") or []
    metadata = {
        "applied": False,
        "original_chars": original_chars,
        "sent_chars": original_chars,
        "relationship_count": len(relationships),
        "relationship_examples_sent": len(relationships),
    }
    if (original_chars <= max_chars and len(relationships) <= 100) or not relationships:
        return features, metadata

    counts = Counter(
        relationship.get("Type", "unknown") for relationship in relationships
    )
    examples: list[dict] = []
    example_counts: dict[str, int] = defaultdict(int)
    for relationship in relationships:
        relationship_type = relationship.get("Type", "unknown")
        if example_counts[relationship_type] >= examples_per_type:
            continue
        examples.append(relationship)
        example_counts[relationship_type] += 1

    compact = dict(features)
    compact["Feature_Relationships"] = examples
    compact["Feature_Relationship_Summary"] = {
        "Total_Count": len(relationships),
        "Counts_By_Type": dict(sorted(counts.items())),
        "Examples_Per_Type_Limit": examples_per_type,
        "Examples_Sent": len(examples),
        "Compaction_Reason": "bounded_second_agent_prompt",
    }
    sent_chars = len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
    if sent_chars > max_chars:
        raise ValueError(
            "关系摘要后的细切 JSON 仍超过模型传输上限: "
            f"{sent_chars} > {max_chars} chars"
        )
    metadata.update({
        "applied": True,
        "sent_chars": sent_chars,
        "relationship_examples_sent": len(examples),
        "relationship_counts_by_type": dict(sorted(counts.items())),
    })
    return compact, metadata


def build_dynamic_refine_user_prompt(
    first_agent: dict,
    plan: dict,
    fine_features: dict,
    coarse_features: dict | None = None,
) -> tuple[str, dict]:
    compact_features, fine_metadata = prepare_feature_evidence_for_prompt(fine_features)
    payload = {
        "第一Agent粗切视觉结论": first_agent,
        "细切选区JSON": plan,
        "0.01mm细切特征摘要": compact_features,
    }
    metadata = {"fine": fine_metadata}
    if coarse_features is not None:
        compact_coarse, coarse_metadata = prepare_feature_evidence_for_prompt(coarse_features)
        payload["0.1mm粗切特征摘要"] = compact_coarse
        metadata["coarse"] = coarse_metadata
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    metadata["total_user_prompt_chars"] = len(prompt)
    return prompt, metadata


def _confirmed_hole_patterns(features: dict) -> list[tuple[dict, list[dict]]]:
    feature_by_id = {
        feature.get("ID"): feature
        for feature in features.get("Recognized_Features") or []
    }
    confirmed = []
    for pattern in features.get("Feature_Patterns") or []:
        count = pattern.get("Count") or 0
        members = [
            feature_by_id[feature_id]
            for feature_id in pattern.get("Feature_IDs") or []
            if feature_id in feature_by_id
        ]
        if count < 3 or len(members) != count:
            continue
        if not all(
            feature.get("Shape") == "Circle"
            and any(token in str(feature.get("Semantic_Type")) for token in ("Hole", "Bore"))
            for feature in members
        ):
            continue
        confirmed.append((pattern, members))
    return confirmed


def ensure_cross_scale_hole_patterns(
    prediction: dict,
    coarse_features: dict,
    fine_features: dict,
) -> tuple[dict, dict]:
    """Deterministically retain hole arrays independently found at both scales."""
    coarse_evidence, _ = prepare_feature_evidence_for_prompt(coarse_features)
    fine_evidence, _ = prepare_feature_evidence_for_prompt(fine_features)
    coarse_patterns = _confirmed_hole_patterns(coarse_evidence)
    fine_patterns = _confirmed_hole_patterns(fine_evidence)
    output = json.loads(json.dumps(prediction, ensure_ascii=False))
    output_features = output.setdefault("局部特征列表", [])
    dimensions = [
        float(output.get(key) or 0.0)
        for key in ("尺寸X", "尺寸Y", "尺寸Z")
    ]
    diagonal = max(math.sqrt(sum(value * value for value in dimensions)), 1.0)
    appended_ids: list[str] = []
    matched_patterns = 0

    def median_diameter(features: list[dict]) -> float:
        values = sorted(float(feature["Cross_Section_Size"][0]) for feature in features)
        return values[len(values) // 2]

    for coarse_pattern, coarse_members in coarse_patterns:
        coarse_diameter = median_diameter(coarse_members)
        corroborated = any(
            (
                coarse_pattern.get("Type"),
                coarse_pattern.get("Axis"),
                coarse_pattern.get("Count"),
            ) == (
                fine_pattern.get("Type"),
                fine_pattern.get("Axis"),
                fine_pattern.get("Count"),
            )
            and abs(coarse_diameter - median_diameter(fine_members))
            / max(coarse_diameter, median_diameter(fine_members), 1.0) <= 0.15
            for fine_pattern, fine_members in fine_patterns
        )
        if not corroborated:
            continue
        matched_patterns += 1
        used_output_indices: set[int] = set()
        for member in coarse_members:
            center = [float(value) for value in member["Center_3D"]]
            diameter = float(member["Cross_Section_Size"][0])
            candidates = []
            for index, feature in enumerate(output_features):
                if index in used_output_indices:
                    continue
                if feature.get("特征类型") != "孔" or feature.get("作用") != "装配特征":
                    continue
                try:
                    candidate_center = [
                        float(feature[key]) for key in ("坐标X", "坐标Y", "坐标Z")
                    ]
                    candidate_size = float(feature["尺寸数据"])
                except (KeyError, TypeError, ValueError):
                    continue
                normalized_distance = math.dist(center, candidate_center) / diagonal
                relative_size_error = abs(diameter - candidate_size) / max(
                    diameter, abs(candidate_size), 1.0
                )
                if normalized_distance <= 0.08 and relative_size_error <= 0.2:
                    candidates.append((normalized_distance + 0.25 * relative_size_error, index))
            if candidates:
                used_output_indices.add(min(candidates)[1])
                continue
            if not all(0.0 <= value <= dimensions[index] for index, value in enumerate(center)):
                continue
            output_features.append({
                "特征类型": "孔",
                "特征形状": "圆形",
                "坐标X": round(center[0], 3),
                "坐标Y": round(center[1], 3),
                "坐标Z": round(center[2], 3),
                "尺寸类型": "直径",
                "尺寸数据": round(diameter, 3),
                "作用": "装配特征",
            })
            used_output_indices.add(len(output_features) - 1)
            appended_ids.append(str(member.get("ID")))

    feature_order = {"孔": 0, "柱": 1, "槽": 2, "倒角": 3}
    output_features.sort(key=lambda feature: feature_order.get(feature.get("特征类型"), 4))
    return output, {
        "matched_cross_scale_patterns": matched_patterns,
        "appended_feature_count": len(appended_ids),
        "appended_coarse_feature_ids": appended_ids,
    }


def process_one_codex(stl_path: Path, base_name: str, pipeline_time: float, csv_path: Path):
    """两阶段 Codex：Vision → Text 矫正。完成后立即写 CSV。"""
    data = get_local_data(base_name)
    if not data["image_path"]:
        append_csv_row(csv_path, {
            "base_name": base_name, "pipeline_time_s": round(pipeline_time, 1),
            "fine_pipeline_time_s": 0,
            "codex_time_s": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "status": "SKIP", "error": "未找到拼合图像",
        })
        return base_name, False, "未找到拼合图像"

    total_prompt = total_completion = 0
    fine_pipeline_time = 0.0
    model_time = 0.0
    try:
        # -- 阶段 1: Vision --
        model_t0 = time.time()
        llm2_output, usage1 = call_codex_vision(
            system_prompt=VISION_SYSTEM_PROMPT,
            user_prompt=VISION_USER_PROMPT,
            image_path=data["image_path"],
            json_schema=FEATURE_SCHEMA,
        )
        model_time += time.time() - model_t0
        total_prompt += usage1["prompt_tokens"]
        total_completion += usage1["completion_tokens"]
        first_agent_path = DEFAULT_RESULTS_DIR / f"{base_name}_first_agent.json"
        first_agent_path.write_text(llm2_output, encoding="utf-8")

        # -- 阶段 2: Text 矫正 --
        if SLICE_MODE == "dynamic":
            fine_t0 = time.time()
            fine = run_dynamic_refinement(stl_path, base_name, llm2_output)
            fine_pipeline_time = round(time.time() - fine_t0, 1)
            fine_features_text = fine["features_txt"].read_text(encoding="utf-8")
            fine_features_data = json.loads(fine_features_text)
            coarse_features_data = json.loads(data["features_text"])
            second_user_prompt, prompt_metadata = build_dynamic_refine_user_prompt(
                json.loads(llm2_output),
                fine["plan"],
                fine_features_data,
                coarse_features_data,
            )
            (DEFAULT_RESULTS_DIR / f"{base_name}_second_agent_payload_meta.json").write_text(
                json.dumps(prompt_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            model_t0 = time.time()
            result, usage2 = call_codex_vision(
                system_prompt=REFINE_SYSTEM_PROMPT,
                user_prompt=second_user_prompt,
                image_path=fine["combined_png"],
                json_schema=REFINE_SCHEMA,
            )
            model_time += time.time() - model_t0
            total_prompt += usage2["prompt_tokens"]
            total_completion += usage2["completion_tokens"]
            validated, validation_metadata = ensure_cross_scale_hole_patterns(
                json.loads(result),
                coarse_features_data,
                fine_features_data,
            )
            result = json.dumps(validated, ensure_ascii=False, separators=(",", ":"))
            prompt_metadata["deterministic_validation"] = validation_metadata
            (DEFAULT_RESULTS_DIR / f"{base_name}_second_agent_payload_meta.json").write_text(
                json.dumps(prompt_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif data["features_text"]:
            llm3_user_prompt = f"原始数据{llm2_output}   提供的新的json信息{data['features_text']}"
            model_t0 = time.time()
            result, usage2 = call_codex_text(
                system_prompt=REFINE_SYSTEM_PROMPT,
                user_prompt=llm3_user_prompt,
                json_schema=REFINE_SCHEMA,
            )
            model_time += time.time() - model_t0
            total_prompt += usage2["prompt_tokens"]
            total_completion += usage2["completion_tokens"]
        else:
            result = llm2_output

        total_tokens = total_prompt + total_completion
        codex_time = round(model_time, 1)
        save_result(base_name, "luna_visual_json_serial", result)
        append_csv_row(csv_path, {
            "base_name": base_name, "pipeline_time_s": round(pipeline_time, 1),
            "fine_pipeline_time_s": fine_pipeline_time,
            "codex_time_s": codex_time,
            "prompt_tokens": total_prompt, "completion_tokens": total_completion,
            "total_tokens": total_tokens, "status": "OK", "error": "",
        })
        return base_name, True, f"OK ({len(result)} 字符, {total_tokens} tokens, {codex_time}s)"

    except Exception as e:
        codex_time = round(model_time, 1)
        append_csv_row(csv_path, {
            "base_name": base_name, "pipeline_time_s": round(pipeline_time, 1),
            "fine_pipeline_time_s": fine_pipeline_time,
            "codex_time_s": codex_time, "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "status": "FAIL", "error": str(e),
        })
        return base_name, False, str(e)


def main():
    try:
        ensure_codex_oauth()
    except CodexCallError as exc:
        print(exc)
        sys.exit(1)

    stl_files = get_stl_files()
    if not stl_files:
        print("在 STL 目录未找到 STL 文件")
        sys.exit(1)

    csv_path = DEFAULT_RESULTS_DIR / "metrics_visual_json_serial.csv"
    print(f"文件数: {len(stl_files)}")
    print(f"模型: {CODEX_MODEL} | 思考: {CODEX_REASONING_EFFORT} | 并发: {MAX_WORKERS}")
    print(f"CSV:  {csv_path}")
    print("=" * 50)

    # ==================== 阶段 1 ====================
    print("\n[阶段1] 批量切片\n" + "-" * 30)
    t0 = time.time()
    pipe_times = {}

    for i, stl_path in enumerate(stl_files, 1):
        print(f"  [{i}/{len(stl_files)}] {stl_path.name}", end=" ... ", flush=True)
        t1 = time.time()
        bn = stl_path.stem
        try:
            run_pipeline(stl_path)
            elapsed = round(time.time() - t1, 1)
            pipe_times[bn] = elapsed
            print(f"{elapsed}s")
        except Exception as e:
            elapsed = round(time.time() - t1, 1)
            pipe_times[bn] = elapsed
            print(f"FAIL ({elapsed}s): {e}")
            append_csv_row(csv_path, {
                "base_name": bn, "pipeline_time_s": elapsed,
                "fine_pipeline_time_s": 0,
                "codex_time_s": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "status": "PIPE_FAIL", "error": str(e)[:200],
            })

    print(f"\n[阶段1] 完成 ({round(time.time() - t0, 1)}s)")

    # ==================== 阶段 2 ====================
    print(f"\n[阶段2] 并发 Codex (workers={MAX_WORKERS}, 每个2轮)\n" + "-" * 30)
    t0 = time.time()
    success, fail = 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_one_codex, stl_path, stl_path.stem, pipe_times[stl_path.stem], csv_path): stl_path.stem
            for stl_path in stl_files if stl_path.stem in pipe_times
        }
        for future in as_completed(futures):
            bn, ok, msg = future.result()
            idx = list(pipe_times.keys()).index(bn) + 1
            status = "OK" if ok else "FAIL"
            print(f"  [{idx}/{len(pipe_times)}] {bn} [{status}] {msg}")
            if ok:
                success += 1
            else:
                fail += 1

    print(f"\n[阶段2] 完成 ({round(time.time() - t0, 1)}s) — 成功 {success}, 失败 {fail}")
    print(f"CSV 已保存: {csv_path}")
    print("\n完成。")


if __name__ == "__main__":
    main()
