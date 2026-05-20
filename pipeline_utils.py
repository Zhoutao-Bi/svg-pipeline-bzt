"""
共享工具：STL 流水线执行、本地数据检索、OpenAI API 调用。

用法:
    from pipeline_utils import run_pipeline, get_local_data, call_openai_vision, save_result

环境变量:
    OPENAI_API_KEY  - OpenAI API 密钥（必需）
    OPENAI_BASE_URL - API 基础 URL（可选，默认 https://api.openai.com/v1）
    OPENAI_MODEL    - 模型名（可选，默认 gpt-5-mini-2025-08-07）
"""

import os
import sys
import json
import base64
import shutil
import subprocess
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

# ---- 配置 ----
INPUT_STL_DIR = BASE_DIR / "dtqp"
DEFAULT_RESULTS_DIR = BASE_DIR / "dtqp_results"
GRASP_FILE = BASE_DIR / "bsp_grasp.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")

# 流水线脚本（按顺序执行）
PIPELINE_SCRIPTS = [
    "stl2vsg11.py",
    "svg2svg.py",
    "vsg_merge.py",
    "svg_json_v6.py",
    "json_token.py",
]

# 流水线生成的临时文件/目录（运行后清理）
PIPELINE_TEMP = [
    "Out_X", "Out_Y", "Out_Z",
    "Out_X_new", "Out_Y_new", "Out_Z_new",
    "Out_X.txt", "Out_Y.txt", "Out_Z.txt",
    "Full_Features_v33_minified2.json", "Full_Features_v33.json",
    "Full_Features_v34.json", "Full_Features_v34_minified.json",
    "Full_Features_v33_minified.json",
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

    # 复制 STL 到 current_task.stl
    target = BASE_DIR / "current_task.stl"
    shutil.copy(stl_path, target)

    for script in PIPELINE_SCRIPTS:
        script_path = BASE_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"缺少脚本: {script}")
        subprocess.run([sys.executable, str(script_path)],
                       check=True, cwd=str(BASE_DIR), capture_output=True)

    # 归档结果
    json_src = BASE_DIR / "Full_Features_v33_minified.json"
    if json_src.exists():
        shutil.copy(json_src, json_dest)
        # 同时存一份 .txt（兼容 Dify 的文件名习惯）
        shutil.copy(json_src, txt_dest)

    # 拼合三视图
    combined = stitch_images(base_name)
    if combined:
        shutil.move(str(combined), str(png_dest))

    clean_pipeline_temp()

    return {
        "base_name": base_name,
        "combined_png": png_dest,
        "features_json": json_dest,
        "features_txt": txt_dest if txt_dest.exists() else json_dest,
    }


def stitch_images(clean_fn: str) -> Optional[Path]:
    """把 X/Y/Z 三视图拼合成一张图"""
    from PIL import Image, ImageDraw, ImageFont

    img_names = ["View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png"]
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

    out_path = BASE_DIR / f"Combined_Views_{clean_fn}.png"
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
    if GRASP_FILE.exists():
        with open(GRASP_FILE, "r", encoding="utf-8") as f:
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
                       base_url: str = "") -> str:
    """
    调用 OpenAI Vision API，传入图像 + 结构化输出 schema。
    返回 LLM 的 JSON 字符串。
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

    return resp.choices[0].message.content


def call_openai_text(system_prompt: str, user_prompt: str,
                     json_schema: dict,
                     api_key: str = "", model: str = "",
                     base_url: str = "") -> str:
    """调用 OpenAI Text API（无图像），返回 JSON 字符串。"""
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

    return resp.choices[0].message.content


def get_stl_files(stl_dir: Optional[Path] = None) -> list:
    """获取待处理的 STL 文件列表"""
    d = stl_dir or INPUT_STL_DIR
    if not d.exists():
        return []
    return sorted(d.rglob("*.[sS][tT][lL]"))
