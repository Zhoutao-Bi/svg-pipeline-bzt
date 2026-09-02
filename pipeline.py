"""
共享工具：STL 几何流水线执行、本地数据检索和结果保存。

用法:
    from pipeline import run_pipeline, get_local_data, save_result
"""

import os
import sys
import gc
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from dynamic_slicing import build_fine_slice_plan, ranges_by_axis

BASE_DIR = Path(__file__).resolve().parent

# ---- 配置 ----
INPUT_STL_DIR = Path(os.getenv("INPUT_STL_DIR", BASE_DIR / "input_stl")).resolve()
INPUT_STL_PATTERN = os.getenv("INPUT_STL_PATTERN", "*.[sS][tT][lL]")
DEFAULT_RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "result")).resolve()
GRIPPER_CONFIG_FILE = BASE_DIR / "gripper_config.json"

# 切片模式: "coarse" | "fine" | "dynamic"
#   coarse  - 粗切片: layer_height=0.1, max_slices=30
#   fine    - 细切片: layer_height=0.01, 不限张数
#   dynamic - 动态切片: 粗切 → 第一 Agent → 选区 0.01 细切 → 第二 Agent
SLICE_MODE = os.getenv("SLICE_MODE", "coarse").lower()
if SLICE_MODE not in {"coarse", "fine", "dynamic"}:
    raise ValueError(f"未知 SLICE_MODE={SLICE_MODE!r}，可选 coarse/fine/dynamic")

DYNAMIC_COARSE_LAYER_HEIGHT = float(os.getenv("DYNAMIC_COARSE_LAYER_HEIGHT", "0.1"))
DYNAMIC_COARSE_MAX_SLICES = int(os.getenv("DYNAMIC_COARSE_MAX_SLICES", "30"))
DYNAMIC_FINE_LAYER_HEIGHT = float(os.getenv("DYNAMIC_FINE_LAYER_HEIGHT", "0.01"))
DYNAMIC_RANGE_MARGIN = float(os.getenv("DYNAMIC_RANGE_MARGIN", "0.2"))
DYNAMIC_FALLBACK_HALF_WIDTH = float(os.getenv("DYNAMIC_FALLBACK_HALF_WIDTH", "1.0"))
DYNAMIC_MAX_FINE_SLICES = int(os.getenv("DYNAMIC_MAX_FINE_SLICES", "30000"))

# 流水线脚本（按顺序执行）
PIPELINE_SCRIPTS = [
    "stl_to_svg.py",
    "optimize_svg.py",
    "merge_svg.py",
    "extract_features.py",
    "minify_features.py",
]

# 流水线生成的临时文件/目录（运行后清理）
PIPELINE_TEMP = [
    "slices_x", "slices_y", "slices_z",
    "optimized_slices_x", "optimized_slices_y", "optimized_slices_z",
    "merged_slices_x.svg", "merged_slices_y.svg", "merged_slices_z.svg",
    "features_raw.json", "features_minified.json", "features_refined.json",
    "slice_metadata.json",
    "depth_view_x.png", "depth_view_y.png", "depth_view_z.png",
    "feature_overview.png",
    "current_task.stl",
]

# All geometry scripts use fixed filenames in BASE_DIR. Dynamic model calls may
# be concurrent, so only one fine geometry pass can own those files at a time.
PIPELINE_LOCK = threading.Lock()


def clean_pipeline_temp():
    for name in PIPELINE_TEMP:
        p = BASE_DIR / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)


def run_pipeline(stl_path: Path, results_dir: Optional[Path] = None) -> dict:
    """
    对单个 STL 文件执行完整流水线，返回 {
        "base_name": str,
        "combined_png": Path,
        "features_json": Path,
        "features_txt": Path,
    }
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    base_name = stl_path.stem
    png_dest = results_dir / f"{base_name}_combined.png"
    json_dest = results_dir / f"{base_name}_features.json"
    txt_dest = results_dir / f"{base_name}_features.txt"

    # 如果结果已存在，直接返回
    if png_dest.exists() and json_dest.exists():
        return {
            "base_name": base_name,
            "combined_png": png_dest,
            "features_json": json_dest,
            "features_txt": txt_dest if txt_dest.exists() else json_dest,
        }

    clean_pipeline_temp()

    # ── 根据 SLICE_MODE 设置切片参数 ──
    env = os.environ.copy()
    if SLICE_MODE == "fine":
        env["SLICE_LAYER_HEIGHT"] = "0.01"
        env["SLICE_MAX_SLICES"] = "99999"
        print(f"[*] 切片模式: 细切片 (layer=0.01, unlimited)")
    elif SLICE_MODE == "dynamic":
        env["SLICE_LAYER_HEIGHT"] = str(DYNAMIC_COARSE_LAYER_HEIGHT)
        env["SLICE_MAX_SLICES"] = str(DYNAMIC_COARSE_MAX_SLICES)
        print(
            "[*] 切片模式: 动态第一阶段粗切 "
            f"(layer={DYNAMIC_COARSE_LAYER_HEIGHT}, max={DYNAMIC_COARSE_MAX_SLICES})"
        )
    else:  # coarse (default)
        env["SLICE_LAYER_HEIGHT"] = "0.1"
        env["SLICE_MAX_SLICES"] = "30"
        print(f"[*] 切片模式: 粗切片 (layer=0.1, max=30)")

    # 复制 STL 到 current_task.stl
    target = BASE_DIR / "current_task.stl"
    shutil.copy(stl_path, target)

    script_timeout = int(os.getenv("PIPELINE_TIMEOUT", "300"))
    for script in PIPELINE_SCRIPTS:
        script_path = BASE_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"缺少脚本: {script}")
        try:
            subprocess.run([sys.executable, str(script_path)],
                           check=True, cwd=str(BASE_DIR), env=env,
                           capture_output=True, timeout=script_timeout)
        except subprocess.TimeoutExpired:
            print(f"    [!] {script} 超时({script_timeout}s)，跳过")
            raise

    # 归档结果（移除 Solid_Base_Layers）
    json_src = BASE_DIR / "features_minified.json"
    if json_src.exists():
        with open(json_src, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        output_data.pop("Solid_Base_Layers", None)
        with open(json_dest, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, separators=(",", ":"))
        with open(txt_dest, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, separators=(",", ":"))
        # 同时更新中间文件，确保后续读取的数据一致
        with open(json_src, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, separators=(",", ":"))

    # 拼合三视图
    combined = stitch_images(base_name)
    if combined:
        shutil.move(str(combined), str(png_dest))

    clean_pipeline_temp()
    gc.collect()  # 释放 trimesh 占用的原生内存

    # 内存监控（诊断用）
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"    [mem] 可用: {mem.available//1024//1024}MB / 总量: {mem.total//1024//1024}MB")
    except ImportError:
        pass

    return {
        "base_name": base_name,
        "combined_png": png_dest,
        "features_json": json_dest,
        "features_txt": txt_dest if txt_dest.exists() else json_dest,
    }


def run_dynamic_refinement(
    stl_path: Path,
    base_name: str,
    first_agent_output: str | dict,
    results_dir: Optional[Path] = None,
) -> dict:
    """Run a first-agent-guided 0.01 mm full-plane refinement pass."""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    coarse_json_path = results_dir / f"{base_name}_features.json"
    if not coarse_json_path.is_file():
        raise FileNotFoundError(f"找不到粗切 JSON: {coarse_json_path}")

    first_agent = (
        json.loads(first_agent_output)
        if isinstance(first_agent_output, str)
        else first_agent_output
    )
    with open(coarse_json_path, "r", encoding="utf-8") as coarse_file:
        coarse_data = json.load(coarse_file)
    plan = build_fine_slice_plan(
        first_agent,
        coarse_data,
        layer_height=DYNAMIC_FINE_LAYER_HEIGHT,
        range_margin=DYNAMIC_RANGE_MARGIN,
        fallback_half_width=DYNAMIC_FALLBACK_HALF_WIDTH,
        max_total_slices=DYNAMIC_MAX_FINE_SLICES,
        coarse_max_slices=DYNAMIC_COARSE_MAX_SLICES,
    )

    plan_path = results_dir / f"{base_name}_fine_slice_plan.json"
    with open(plan_path, "w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, ensure_ascii=False, indent=2)

    png_dest = results_dir / f"{base_name}_fine_combined.png"
    json_dest = results_dir / f"{base_name}_fine_features.json"
    txt_dest = results_dir / f"{base_name}_fine_features.txt"
    env = os.environ.copy()
    env["SLICE_LAYER_HEIGHT"] = str(DYNAMIC_FINE_LAYER_HEIGHT)
    env["SLICE_MAX_SLICES"] = str(DYNAMIC_MAX_FINE_SLICES)
    env["SLICE_RANGES_JSON"] = json.dumps(ranges_by_axis(plan), separators=(",", ":"))
    env["PRESERVE_GLOBAL_SLICE_COORDINATES"] = "1"

    print(
        f"[*] {base_name} 动态细切: {len(plan['ranges'])} 个合并区间, "
        f"预计 {plan['estimated_slices']} 层"
    )
    script_timeout = int(os.getenv("PIPELINE_TIMEOUT", "300"))
    with PIPELINE_LOCK:
        clean_pipeline_temp()
        try:
            shutil.copy(stl_path, BASE_DIR / "current_task.stl")
            for script in PIPELINE_SCRIPTS:
                script_path = BASE_DIR / script
                subprocess.run(
                    [sys.executable, str(script_path)],
                    check=True,
                    cwd=str(BASE_DIR),
                    env=env,
                    capture_output=True,
                    timeout=script_timeout,
                )

            json_src = BASE_DIR / "features_minified.json"
            if not json_src.is_file():
                raise FileNotFoundError("动态细切未生成 features_minified.json")
            with open(json_src, "r", encoding="utf-8") as source_file:
                fine_data = json.load(source_file)
            fine_data.pop("Solid_Base_Layers", None)
            compact = json.dumps(fine_data, ensure_ascii=False, separators=(",", ":"))
            json_dest.write_text(compact, encoding="utf-8")
            txt_dest.write_text(compact, encoding="utf-8")

            combined = stitch_images(f"{base_name}_fine")
            if not combined:
                raise FileNotFoundError("动态细切未生成可用渲染图")
            shutil.move(str(combined), str(png_dest))
        finally:
            clean_pipeline_temp()
            gc.collect()

    return {
        "base_name": base_name,
        "combined_png": png_dest,
        "features_json": json_dest,
        "features_txt": txt_dest,
        "plan_json": plan_path,
        "plan": plan,
    }


def stitch_images(clean_fn: str) -> Optional[Path]:
    """把 X/Y/Z 三视图拼合成一张图"""
    from PIL import Image, ImageDraw, ImageFont

    img_names = ["depth_view_x.png", "depth_view_y.png", "depth_view_z.png"]
    images, valid_names = [], []

    for name in img_names:
        p = BASE_DIR / name
        if p.exists():
            images.append(Image.open(p))
            valid_names.append(name)

    if not images:
        return None

    header_height = 150
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images) + header_height

    combined = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined)

    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except (IOError, OSError):
        font = ImageFont.load_default()

    x_offset = 0
    for img, name in zip(images, valid_names):
        combined.paste(img, (x_offset, header_height))
        draw.text((x_offset + 50, 40), name, fill=(0, 0, 0), font=font)
        x_offset += img.width

    for img in images:
        img.close()

    out_path = BASE_DIR / f"combined_views_{clean_fn}.png"
    combined.save(str(out_path))
    combined.close()
    return out_path


def get_local_data(base_name: str, results_dir: Optional[Path] = None) -> dict:
    """
    本地版 /get_local_data：读取流水线结果，返回 {
        "image_path": Path | None, # 拼合图像路径
        "features_text": str,     # JSON 几何特征文本
        "grasp_text": str,        # 抓手信息文本
    }
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    # 定位拼合图像
    png_path = results_dir / f"{base_name}_combined.png"

    # 读取特征文本（优先 .txt，其次 .json）
    txt_path = results_dir / f"{base_name}_features.txt"
    json_path = results_dir / f"{base_name}_features.json"
    features_text = ""
    for p in (txt_path, json_path):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                features_text = f.read()
            break

    # 读取全局 grasp 文本
    grasp_text = ""
    if GRIPPER_CONFIG_FILE.exists():
        with open(GRIPPER_CONFIG_FILE, "r", encoding="utf-8") as f:
            grasp_text = f.read()

    return {
        "image_path": png_path if png_path.is_file() else None,
        "features_text": features_text,
        "grasp_text": grasp_text,
    }


def save_result(base_name: str, model_name: str, text_content: str,
                results_dir: Optional[Path] = None) -> Path:
    """本地版 /save_result_refined：将 Codex 输出保存到文件。"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    save_path = results_dir / f"{base_name}_refined_{model_name}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    return save_path


def append_csv_row(csv_path: Path, row: dict):
    """追加一行到 CSV，如果文件不存在则先写表头。"""
    import csv
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_stl_files(stl_dir: Optional[Path] = None) -> list:
    """获取待处理的 STL 文件列表"""
    d = stl_dir or INPUT_STL_DIR
    if not d.exists():
        return []
    return sorted(path for path in d.rglob(INPUT_STL_PATTERN) if path.is_file())
