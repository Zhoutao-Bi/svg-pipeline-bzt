"""
消融实验 - 有串行 JSON（视觉 → JSON 矫正）

阶段1: 批量切片 → 阶段2: 并发两轮 Codex 分析。CSV 实时追加。
"""

import os
import sys
import time
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

REFINE_SYSTEM_PROMPT = r"""Role: 你是一位资深机械设计工程师和 CAD 数据分析专家。你的任务是联合细切深度图、细切 JSON 与第一 Agent 的粗切视觉结论，对数据进行最终矫正。
分析思路与核心准则:
信息依赖: 附图是第一 Agent 选区后以 0.01 mm 重新切片生成的渲染图；用户文本同时给出细切选区 JSON、细切特征 JSON 和第一 Agent 结论，三者都必须使用。
特征判别：该部分属于装配、轻量化、其他类型的特征或者是无用特征。
2、重复判断：注意判断是否有重复和重叠的局部特征。
数据矫正: 视觉负责判断真实形状与作用，细切 JSON 优先提供坐标和尺寸；第一 Agent 结论用于保留粗切阶段已确认但细切选区未覆盖的特征。"""


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
    if original_chars <= max_chars or not relationships:
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
) -> tuple[str, dict]:
    compact_features, metadata = compact_refine_features_for_prompt(fine_features)
    prompt = json.dumps({
        "第一Agent粗切视觉结论": first_agent,
        "细切选区JSON": plan,
        "0.01mm细切特征JSON": compact_features,
    }, ensure_ascii=False, separators=(",", ":"))
    metadata = {**metadata, "total_user_prompt_chars": len(prompt)}
    return prompt, metadata


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
            second_user_prompt, prompt_metadata = build_dynamic_refine_user_prompt(
                json.loads(llm2_output),
                fine["plan"],
                json.loads(fine_features_text),
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
