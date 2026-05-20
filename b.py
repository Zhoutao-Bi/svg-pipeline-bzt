import os
import json

def batch_convert_json_to_txt(input_folder, output_folder):
    """
    批量将文件夹下的 JSON 文件转换为 TXT 文件
    
    :param input_folder: 包含 JSON 文件的源文件夹路径
    :param output_folder: 保存 TXT 文件的目标文件夹路径
    """
    # 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_filepath = os.path.join(input_folder, filename)
            # 构建对应的 TXT 文件名
            txt_filename = filename.replace(".json", ".txt")
            txt_filepath = os.path.join(output_folder, txt_filename)

            try:
                # 读取 JSON 文件
                with open(json_filepath, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                # 写入 TXT 文件
                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    
                    # ==========================================
                    # 模式 1：将整个 JSON 数据格式化后直接写入 TXT
                    # 适用于只需要改后缀或保存完整数据结构的情况
                    # ==========================================
                    formatted_data = json.dumps(data, indent=4, ensure_ascii=False)
                    txt_file.write(formatted_data)

                    # ==========================================
                    # 模式 2：提取 JSON 中的特定字段写入 TXT (需根据实际数据结构修改)
                    # 示例：假设 JSON 是一个字典，只写入它的键值对
                    # ==========================================
                    # if isinstance(data, dict):
                    #     for key, value in data.items():
                    #         txt_file.write(f"{key}: {value}\n")
                    # elif isinstance(data, list):
                    #     for item in data:
                    #         txt_file.write(f"{str(item)}\n")
                    
                print(f"✅ 成功转换: {filename} -> {txt_filename}")
                
            except json.JSONDecodeError:
                print(f"❌ 转换失败 {filename}: JSON 格式不正确，无法解析。")
            except Exception as e:
                print(f"❌ 转换失败 {filename}: 发生错误 {e}")

if __name__ == "__main__":
    # 配置你的文件夹路径 (相对路径或绝对路径均可)
    INPUT_DIR = "./local_results"   # 替换为存放 .json 文件的文件夹路径
    OUTPUT_DIR = "./local_results2" # 替换为准备存放 .txt 文件的文件夹路径
    
    # 运行批量转换
    batch_convert_json_to_txt(INPUT_DIR, OUTPUT_DIR)
    print("\n🎉 批量处理完成！")