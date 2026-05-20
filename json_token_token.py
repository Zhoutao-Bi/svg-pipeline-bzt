import json

def process_to_compact_json(input_file, output_file):
    try:
        # 1. 读取原始 JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 模拟 Dify 逻辑：确保拿到特征字典
        features_dict = data.get("features", data)
        if isinstance(features_dict, str):
            features_dict = json.loads(features_dict)

        # 3. 核心减负：直接整块删除 Solid_Base_Layers
        # 只要存在这个键，就把它连根拔起
        if "Solid_Base_Layers" in features_dict:
            del features_dict["Solid_Base_Layers"]
            print("✂️ 已彻底删除 Solid_Base_Layers 数据块")

        # 4. 极限压缩：利用 separators 去掉所有空格，转成紧凑字符串
        features_text = json.dumps(features_dict, ensure_ascii=False, separators=(',', ':'))

        # 5. 封装成最终简洁的 JSON 结构
        result_payload = {
            "features_text": features_text
        }

        # 6. 保存为文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_payload, f, ensure_ascii=False)

        print(f"✅ 任务完成！已生成极简 JSON：{output_file}")

    except Exception as e:
        print(f"❌ 解析失败: {str(e)}")

# ================= 运行 =================
if __name__ == "__main__":
    process_to_compact_json("Full_Features_v33_minified.json", "Full_Features_v33_minified2.json")