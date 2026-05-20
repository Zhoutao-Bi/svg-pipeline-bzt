import os
import re
import json
import pandas as pd
from pathlib import Path

def parse_to_wide_row(file_path):
    """读取 txt，将所有特征横向展开到一行中的独立列，并使用 [ X ] 格式的序号"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 移除干扰标签，例如 [source: 1] 
        clean_content = re.sub(r'\[source:\s*\d+\]', '', content).strip()
        
        # 2. 解析 JSON
        data = json.loads(clean_content)
        
        # 3. 提取全局基础信息
        row_data = {
            "文件名": file_path.name,
            "板长_X": data.get("尺寸X"),
            "板宽_Y": data.get("尺寸Y"),
            "板厚_Z": data.get("尺寸Z"),
            "整体特征描述": data.get("整体特征")
        }
        
        # 4. 提取并横向展开局部特征
        features = data.get("局部特征列表", [])
        row_data["特征总数"] = len(features)
        
        for i, feat in enumerate(features, 1):
            # 动态生成带有 [ X ] 格式的列名
            # 例如: 特征 [ 1 ]_类型
            prefix = f"特征[{i}]"
            
            row_data[f"{prefix}_类型"] = feat.get("特征类型")
            row_data[f"{prefix}_形状"] = feat.get("特征形状")
            row_data[f"{prefix}_坐标X"] = feat.get("坐标X")
            row_data[f"{prefix}_坐标Y"] = feat.get("坐标Y")
            row_data[f"{prefix}_坐标Z"] = feat.get("坐标Z")
            
            # 尺寸信息如果有换行，替换为空格，避免 Excel 行高过大
            size_info = feat.get("尺寸类型", "")
            if isinstance(size_info, str):
                size_info = size_info.replace('\n', ' ')
            row_data[f"{prefix}_尺寸类型"] = size_info
            size_info = feat.get("尺寸数据", "")
            if isinstance(size_info, str):
                size_info = size_info.replace('\n', ' ')
            row_data[f"{prefix}_尺寸数据"] = size_info
            
        return row_data
        
    except Exception as e:
        print(f"解析文件 {file_path.name} 时出错: {e}")
        return None

def main(folder_path, output_file):
    input_dir = Path(folder_path)
    all_rows = []
    
    # 遍历处理所有 txt 文件
    for txt_file in input_dir.glob("*.txt"):
        row_data = parse_to_wide_row(txt_file)
        if row_data:
            all_rows.append(row_data)
    
    if all_rows:
        # 创建 DataFrame，pandas 会自动对齐所有的列
        df = pd.DataFrame(all_rows)
        
        # 提取基础列和特征列，调整顺序
        base_cols = ["文件名", "板长_X", "板宽_Y", "板厚_Z", "特征总数", "整体特征描述"]
        feature_cols = [col for col in df.columns if col not in base_cols]
        
        # 确保 特征 [ 1 ] 排在 特征 [ 2 ] 前面
        feature_cols.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        
        # 重新组合列顺序
        final_cols = base_cols + feature_cols
        df = df[final_cols]
        
        # 导出 Excel
        df.to_excel(output_file, index=False)
        print(f"✅ 处理完成！已将 {len(all_rows)} 个文件存入宽表。")
        print(f"输出路径: {output_file}")
    else:
        print("❌ 未发现有效数据，请检查文件夹路径或文件格式。")

# --- 配置区 ---
FOLDER = "./"  # 将这里替换为你的实际文件夹路径
OUTPUT = "特征展开宽表_新格式.xlsx"

if __name__ == "__main__":
    main(FOLDER, OUTPUT)