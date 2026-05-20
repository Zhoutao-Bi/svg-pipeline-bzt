"""
消融实验 - 有串行 JSON（视觉 → JSON 矫正）

第一阶段：Vision LLM 仅看深度图，输出初步特征。
第二阶段：Text LLM 读取 JSON 几何特征，对第一阶段的结果进行坐标/尺寸矫正。

用法:
    export OPENAI_API_KEY="sk-..."
    python workflow_serial_json.py
"""

import sys

from pipeline_utils import (
    get_stl_files, run_pipeline, get_local_data, save_result,
    call_openai_vision, call_openai_text, OPENAI_API_KEY, OPENAI_MODEL,
)

# ── 第一阶段：Vision LLM 的 Schema（同无json）───────────────
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
                },
                "required": ["特征类型", "特征形状", "坐标X", "坐标Y", "坐标Z", "尺寸类型", "尺寸数据"],
                "additionalProperties": False,
            },
        },
        "整体特征": {
            "description": "他的整体的几何形状，各个特征，他是啥，可能是干嘛的，有啥用。不需要写数字，描述即可",
            "type": "string",
        },
    },
    "required": ["名字", "整体特征", "尺寸X", "尺寸Y", "尺寸Z", "局部特征列表"],
    "additionalProperties": False,
}

# ── 第二阶段：Text LLM 的 Schema（无"名字"字段）────────────────
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
            "description": "他的整体的几何形状，各个特征，他是啥，可能是干嘛的，有啥用。不需要写数字，描述即可",
            "type": "string",
        },
    },
    "required": ["整体特征", "尺寸X", "尺寸Y", "尺寸Z", "局部特征列表"],
    "additionalProperties": False,
}

# ── 第一阶段 System Prompt（纯视觉，同无json）────────────────
LLM2_SYSTEM_PROMPT = """System Prompt: 你是一位资深机器视觉与逆向工程专家。你的任务是仅依靠提供的多视图深度图（X/Y/Z轴），对该机械零件进行视觉特征提取，并直接输出可填入结构化表格的数据。
分析思路与核心准则:
1、纯视觉依赖: 你需要通过观察深度图的色阶映射（黄向紫代表浅入深）和三视图轮廓，提取零件的几何形态。
2、重复判断：由于三视图问题，注意判断是否有重复和重叠的局部特征。尤其是当XYZ的坐标在很接近的位置时，特别是颜色渐变的地方，此时应该用另外两个侧视图去判断。如果有则保留一个最可能的即可。
3、包络估算: 观察长宽高比例，估测全局包络尺寸（请提供基于视觉比例的数值估算）。
4、视觉提示: 寻找深度图上的色阶突变区域或穿透区域（如明显的白色/深色空洞）。重点识别并列出：孔、柱、槽、倒角。
形态判定: 利用视觉优势，直接判定连续的颜色渐变为【平滑曲面/倒角/锥面】。准确分辨孔、柱、槽的真实形状（正圆、胶囊形、半圆槽、扇形、方形、三角形等。
输出约束: 严格按照提供的 JSON Schema 输出。因为没有原始数据，坐标 和 尺寸 字段请给出基于视觉逻辑和比例尺的估算数值。局部特征必须按照孔、柱、槽、圆角的顺序输出。输出格式的例子如下：
输出格式的例子如下：
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

LLM2_USER_PROMPT = "结构化输出。"

# ── 第二阶段 System Prompt（用 JSON 矫正）───────────────────
LLM3_SYSTEM_PROMPT = """Role: 你是一位资深机械设计工程师和 CAD 数据分析专家。你的任务是根据提供的 JSON 几何特征描述文件，对数据进行矫正。
分析思路与核心准则:
信息依赖: 你需要通过json的信息和提供给你的txt数据信息进行对比。
特征判别：该部分属于装配、轻量化、其他类型的特征或者是无用特征。
2、重复判断：注意判断是否有重复和重叠的局部特征。尤其是当各个特征的XYZ的坐标在接近的位置时，如果有则保留一个最可能的即可。
数据矫正: 参考原始txt里的坐标、尺寸、整体特征的描述信息，将原始的txt信息里的坐标、尺寸数据更新成json信息里的。如果json里没有则无需改动。其他的不要做任何修改。
输出格式的例子如下：
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

        # 2. 获取本地数据（跳过流水线）
        data = get_local_data(base_name)
        if not data["img_base64"]:
            print(f"  SKIP: 未找到拼合图像")
            continue

        # 3. 第一阶段：纯视觉 LLM
        print(f"  阶段1: Vision LLM (纯视觉)...")
        try:
            llm2_output = call_openai_vision(
                system_prompt=LLM2_SYSTEM_PROMPT,
                user_prompt=LLM2_USER_PROMPT,
                img_base64=data["img_base64"],
                json_schema=FEATURE_SCHEMA,
            )
            print(f"  阶段1: 完成 ({len(llm2_output)} 字符)")
        except Exception as e:
            print(f"  阶段1 LLM 调用失败: {e}")
            continue

        # 4. 第二阶段：用 JSON 特征矫正
        if not data["features_text"]:
            print(f"  阶段2: SKIP (无 JSON 特征数据)")
            result = llm2_output
        else:
            print(f"  阶段2: Text LLM (JSON 矫正)...")
            llm3_user_prompt = (
                f"原始数据{llm2_output}   提供的新的json信息{data['features_text']}"
            )
            try:
                result = call_openai_text(
                    system_prompt=LLM3_SYSTEM_PROMPT,
                    user_prompt=llm3_user_prompt,
                    json_schema=REFINE_SCHEMA,
                )
                print(f"  阶段2: 完成 ({len(result)} 字符)")
            except Exception as e:
                print(f"  阶段2 LLM 调用失败: {e}")
                result = llm2_output  # fallback 到第一阶段结果

        # 5. 保存
        save_path = save_result(base_name, "gpt5mini_serialjson", result)
        print(f"  结果已保存: {save_path.name}")

    print(f"\n完成。")


if __name__ == "__main__":
    main()
