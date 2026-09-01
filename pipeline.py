"""
共享工具：STL 流水线执行、本地数据检索、OpenAI API 调用。

用法:
    from pipeline import run_pipeline, get_local_data, call_openai_vision, save_result

环境变量:
    OPENAI_API_KEY  - OpenAI API 密钥（必需）
    OPENAI_BASE_URL - API 基础 URL（可选，默认 https://api.openai.com/v1）
    OPENAI_MODEL    - 模型名（可选，默认 gpt-5-mini-2025-08-07）
"""

import os
import sys
import gc
import json
import base64
import shutil
import subprocess
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

# ---- 配置 ----
INPUT_STL_DIR = Path(os.getenv("INPUT_STL_DIR", BASE_DIR / "input_stl")).resolve()
DEFAULT_RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results")).resolve()
GRIPPER_CONFIG_FILE = BASE_DIR / "gripper_config.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")

# 切片模式: "coarse" | "fine" | "dynamic"
#   coarse  - 粗切片: layer_height=0.1, max_slices=30
#   fine    - 细切片: layer_height=0.01, 不限张数
#   dynamic - 动态切片: 先粗切片(0.1/30张) → 再用 refine_features 高精度精炼(0.01)
SLICE_MODE = os.getenv("SLICE_MODE", "coarse")

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
    "depth_view_x.png", "depth_view_y.png", "depth_view_z.png",
    "feature_overview.png",
    "current_task.stl",
]


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
        env["SLICE_LAYER_HEIGHT"] = "0.1"
        env["SLICE_MAX_SLICES"] = "30"
        print(f"[*] 切片模式: 动态 (先粗后精)")
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

    # ── 动态模式：调用 refine_features 进行高精度精炼 ──
    if SLICE_MODE == "dynamic":
        refiner_path = BASE_DIR / "refine_features.py"
        if refiner_path.exists():
            print("[*] 动态模式: 运行 refine_features 高精度精炼...")
            subprocess.run([sys.executable, str(refiner_path)],
                           check=True, cwd=str(BASE_DIR), env=env, capture_output=True)
            # 用精炼结果覆盖精简特征（后续步骤以此为准）
            refined_json = BASE_DIR / "features_refined.json"
            if refined_json.exists():
                import json as _json
                with open(refined_json, "r", encoding="utf-8") as f:
                    refined_data = _json.load(f)
                refined_data.pop("Solid_Base_Layers", None)
                minified_path = BASE_DIR / "features_minified.json"
                with open(minified_path, "w", encoding="utf-8") as f:
                    _json.dump(refined_data, f, ensure_ascii=False, separators=(",", ":"))
                print(f"[+] 精炼完成: {refined_json.name} → {minified_path.name}")
        else:
            print("[!] 找不到 refine_features.py，跳过精炼")

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
        "img_base64": str,        # 拼合图像的 base64 编码
        "features_text": str,     # JSON 几何特征文本
        "grasp_text": str,        # 抓手信息文本
    }
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    # 读取拼合图像
    png_path = results_dir / f"{base_name}_combined.png"
    img_base64 = ""
    if png_path.exists():
        with open(png_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

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
        "img_base64": img_base64,
        "features_text": features_text,
        "grasp_text": grasp_text,
    }


def save_result(base_name: str, model_name: str, text_content: str,
                results_dir: Optional[Path] = None) -> Path:
    """本地版 /save_result_refined：将 LLM 输出保存到文件"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    save_path = results_dir / f"{base_name}_refined_{model_name}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    return save_path


def call_openai_vision(system_prompt: str, user_prompt: str,
                       img_base64: str, json_schema: dict,
                       api_key: str = "", model: str = "",
                       base_url: str = "") -> tuple:
    """
    调用 OpenAI Vision API，传入图像 + 结构化输出 schema。
    返回 (content: str, usage: dict)
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or OPENAI_API_KEY,
        base_url=base_url or OPENAI_BASE_URL,
    )
    model_name = model or OPENAI_MODEL

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "part_features",
                "strict": True,
                "schema": json_schema,
            },
        },
        # temperature 已移除，此模型不支持自定义值
    )

    content = resp.choices[0].message.content
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
    }
    return content, usage


def call_openai_text(system_prompt: str, user_prompt: str,
                     json_schema: dict,
                     api_key: str = "", model: str = "",
                     base_url: str = "") -> tuple:
    """调用 OpenAI Text API（无图像）。返回 (content: str, usage: dict)。"""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or OPENAI_API_KEY,
        base_url=base_url or OPENAI_BASE_URL,
    )
    model_name = model or OPENAI_MODEL

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "part_features",
                "strict": True,
                "schema": json_schema,
            },
        },
        # temperature 已移除，此模型不支持自定义值
    )

    content = resp.choices[0].message.content
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
    }
    return content, usage


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
    return sorted(d.rglob("*.[sS][tT][lL]"))
