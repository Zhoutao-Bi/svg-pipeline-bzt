"""
消融实验 - 无 JSON（纯视觉）

仅依赖多视图深度图（不提供 JSON 几何特征），由 Vision LLM 直接推断
零件的几何特征（包络尺寸、孔/柱/槽/倒角）。

用法:
    export OPENAI_API_KEY="sk-..."
    python workflow_no_json.py
"""

import sys

from pipeline_utils import (
    get_stl_files, run_pipeline, get_local_data, save_result,
    call_openai_vision, OPENAI_API_KEY, OPENAI_MODEL,
)

# ── JSON Schema（与 Dify workflow 一致）──────────────────────
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
                    "特征类型": {
                        "description": "只能回答以下其中一个词语：孔、柱、槽、倒角",
                        "type": "string",
                    },
                    "特征形状": {
                        "description": "只能回答以下其中一个词语：圆形、类圆形、三角形、四边形、五边形、六边形、多边形、其他。",
                        "type": "string",
                    },
                    "坐标X": {"type": "number"},
                    "坐标Y": {"type": "number"},
                    "坐标Z": {"type": "number"},
                    "尺寸类型": {
                        "description": "只能回答以下其中一个词语：直径、边长、角度",
                        "type": "string",
                    },
                    "尺寸数据": {"type": "number"},
                    "作用": {
                        "description": "只能回答以下其中一个词语：装配特征、轻量化特征、其他",
                        "type": "string",
                    },
                },
                "required": ["特征类型", "特征形状", "坐标X", "坐标Y", "坐标Z", "尺寸类型", "尺寸数据", "作用"],
                "additionalProperties": False,
            },
        },
        "整体特征": {
            "description": "对零件主基体形状的描述（如：他是啥，可能是干嘛的，有啥用。）",
            "type": "string",
        },
    },
    "required": ["名字", "整体特征", "尺寸X", "尺寸Y", "尺寸Z", "局部特征列表"],
    "additionalProperties": False,
}

# ── System Prompt（来自消融实验无json.yml）────────────────────
SYSTEM_PROMPT = """System Prompt: 你是一位资深机器视觉与逆向工程专家。你的任务是仅依靠提供的多视图深度图（X/Y/Z轴），对该机械零件进行视觉特征提取，并直接输出可填入结构化表格的数据。
分析思路与核心准则:
纯视觉依赖: 你需要通过观察深度图的色阶映射（黄向紫代表浅入深）和三视图轮廓，提取零件的几何形态。
包络估算: 观察长宽高比例，估测全局包络尺寸（请提供基于视觉比例的数值估算）。
语义识别为主: 寻找深度图上的色阶突变区域或穿透区域（如明显的白色/深色空洞）。重点识别并列出：孔、柱、槽、倒角。
形态判定: 利用视觉优势，直接判定连续的颜色渐变为【平滑曲面/倒角/锥面】。准确分辨孔、柱、槽的真实形状（正圆、胶囊形、半圆槽、扇形、方形、三角形等。
特征判别：该部分属于装配、轻量化、其他类型的特征或者是无用特征。
重复判断：由于三视图问题，注意判断是否有重复的局部特征。尤其是当XYZ的坐标在很接近的位置时，如果有则保留一个最可能的即可。
输出约束: 严格按照提供的 JSON Schema 输出。因为没有原始数据，坐标 和 尺寸 字段请给出基于视觉逻辑和比例尺的估算数值。局部特征必须按照孔、柱、槽、圆角的顺序输出。输出格式的例子如下：
{
    "尺寸X": 50,
    "尺寸Y": 150,
    "尺寸Z": 10,
    "局部特征列表": [
        {
           "特征类型":"孔",
           "特征形状":"圆形",
            "坐标X": 25,
            "坐标Y": 130,
            "坐标Z": 5,
            "尺寸类型": "直径",
             "尺寸数据":40,
            "作用": "装配特征"
        },
        {
           "特征类型":"孔",
           "特征形状":"圆形",
            "坐标X": 40,
            "坐标Y": 120,
            "坐标Z": 6,
             "尺寸类型": "直径",
             "尺寸数据":20
             "作用": "轻量化特征"
        },
{
           "特征类型":"柱",
           "特征形状":"方形",
            "坐标X": 40,
            "坐标Y": 120,
            "坐标Z": 6,
             "尺寸类型": "边长",
             "尺寸数据":"40",
             "尺寸数据":"20"
             "作用": "装配特征"
        },
    ]
    "整体特征": "长方形薄板状工件，整体为长条矩形带孔和柱；在靠近一端偏上的位置有一个大型贯穿圆孔。",
}"""

USER_PROMPT = "结构化输出。"


def main():
    if not OPENAI_API_KEY:
        print("请先设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    stl_files = get_stl_files()
    if not stl_files:
        print(f"在 111/ 目录未找到 STL 文件")
        sys.exit(1)

    print(f"找到 {len(stl_files)} 个 STL 文件")
    print(f"模型: {OPENAI_MODEL}")
    print(f"{'=' * 50}")

    for i, stl_path in enumerate(stl_files, 1):
        print(f"\n[{i}/{len(stl_files)}] {stl_path.name}")

        # 1. 有数据跳过，无数据则切片
        info = run_pipeline(stl_path)
        base_name = info["base_name"]

        # 2. 获取本地数据
        data = get_local_data(base_name)
        if not data["img_base64"]:
            print(f"  SKIP: 未找到拼合图像")
            continue

        # 3. 调用 Vision LLM（无 JSON 特征）
        print(f"  调用 LLM (纯视觉)...")
        try:
            result = call_openai_vision(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
                img_base64=data["img_base64"],
                json_schema=FEATURE_SCHEMA,
            )
        except Exception as e:
            print(f"  LLM 调用失败: {e}")
            continue

        # 4. 保存结果
        save_path = save_result(base_name, "gpt5mini_nojson", result)
        print(f"  结果已保存: {save_path.name}")

    print(f"\n完成。")


if __name__ == "__main__":
    main()
