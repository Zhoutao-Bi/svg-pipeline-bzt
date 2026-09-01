"""
消融实验 - 有并行 JSON（视觉 + JSON 融合）

阶段1: 批量切片 → 阶段2: 并发 LLM。CSV 实时追加。
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline_utils import (
    get_stl_files, run_pipeline, get_local_data, save_result,
    call_openai_vision, append_csv_row,
    OPENAI_API_KEY, OPENAI_MODEL, DEFAULT_RESULTS_DIR,
)

MAX_WORKERS = int(os.getenv("LLM_CONCURRENCY", "1"))

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
        "整体特征": {"description": "对零件主基体形状的描述（他的整体的几何形状，各个特征，他是啥，可能是干嘛的，有啥用。不需要写数字，描述即可）", "type": "string"},
    },
    "required": ["名字", "整体特征", "尺寸X", "尺寸Y", "尺寸Z", "局部特征列表"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = r"""System Prompt: 你是一位资深机器视觉与逆向工程专家。你的任务是依靠提供的多视图深度图（X/Y/Z轴）， JSON 几何特征描述文件，对该机械零件进行特征提取，并直接输出可填入结构化表格的数据。
分析思路与核心准则:
信息依赖: 你需要通过观察深度图的色阶映射（黄向紫代表浅入深）和三视图轮廓，提取零件的几何形态。一切识别特征以视觉为主，视觉占比70%，一切几何定位特征以json文本位置，json文本占比70%。若两者冲突需要指出提示。
包络统计: 通过json完成长宽高的统计。
语义识别为主: 寻找深度图上的色阶突变区域或穿透区域（如明显的白色/深色空洞）。重点识别并列出：孔、柱、槽、倒角。
形态判定: 利用视觉优势，直接判定连续的颜色渐变为【平滑曲面/倒角/锥面】。准确分辨孔、柱、槽的真实形状（正圆、胶囊形、半圆槽、扇形、方形、三角形等。
特征判别：该部分属于装配、轻量化、其他类型的特征或者是无用特征。
重复判断：由于三视图问题，注意判断是否有重复的局部特征。尤其是当XYZ的坐标在很接近的位置时，如果有则保留一个最可能的即可。
输出约束: 严格按照提供的 JSON Schema 输出。坐标 和 尺寸 字段请通过json完成读取。局部特征必须按照孔、柱、槽、圆角的顺序输出。"""


def process_one_llm(base_name: str, pipeline_time: float, csv_path: Path):
    """单模型 LLM（视觉 + JSON 融合）。完成后立即写 CSV。"""
    data = get_local_data(base_name)
    if not data["img_base64"]:
        append_csv_row(csv_path, {
            "base_name": base_name, "pipeline_time_s": round(pipeline_time, 1),
            "llm_time_s": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "status": "SKIP", "error": "未找到拼合图像",
        })
        return base_name, False, "未找到拼合图像"

    user_prompt = f"结构化输出。json的内容：{data['features_text']}"
    t0 = time.time()
    try:
        content, usage = call_openai_vision(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            img_base64=data["img_base64"],
            json_schema=FEATURE_SCHEMA,
        )
        llm_time = round(time.time() - t0, 1)
        save_result(base_name, "gpt5mini_paralleljson", content)
        append_csv_row(csv_path, {
            "base_name": base_name, "pipeline_time_s": round(pipeline_time, 1),
            "llm_time_s": llm_time,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "status": "OK", "error": "",
        })
        return base_name, True, f"OK ({len(content)} 字符, {usage['total_tokens']} tokens, {llm_time}s)"
    except Exception as e:
        llm_time = round(time.time() - t0, 1)
        append_csv_row(csv_path, {
            "base_name": base_name, "pipeline_time_s": round(pipeline_time, 1),
            "llm_time_s": llm_time, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "status": "FAIL", "error": str(e),
        })
        return base_name, False, str(e)


def main():
    if not OPENAI_API_KEY:
        print("请先设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    stl_files = get_stl_files()
    if not stl_files:
        print("在 STL 目录未找到 STL 文件")
        sys.exit(1)

    csv_path = DEFAULT_RESULTS_DIR / "metrics_paralleljson.csv"
    print(f"文件数: {len(stl_files)}")
    print(f"模型: {OPENAI_MODEL} | 并发: {MAX_WORKERS}")
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
                "llm_time_s": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "status": "PIPE_FAIL", "error": str(e)[:200],
            })

    print(f"\n[阶段1] 完成 ({round(time.time() - t0, 1)}s)")

    # ==================== 阶段 2 ====================
    print(f"\n[阶段2] 并发 LLM (workers={MAX_WORKERS})\n" + "-" * 30)
    t0 = time.time()
    success, fail = 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_one_llm, bn, pipe_times[bn], csv_path): bn
            for bn in pipe_times
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
