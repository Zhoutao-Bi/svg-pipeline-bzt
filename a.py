import os
import pandas as pd
from datetime import datetime

def calculate_json_generation_times(folder_path, output_excel_path):
    # 1. 获取文件夹中所有的 .json 文件路径
    # 这里将 .txt 修改为了 .json
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    
    if not files:
        print("未在指定文件夹中找到任何 json 文件！")
        return

    # 2. 获取每个文件的创建时间并存储
    file_data = []
    for file in files:
        # 注意：os.path.getctime 在 Windows 上是创建时间，在 Linux 上是 metadata 最后修改时间
        # 如果你使用的是 Mac，严格的创建时间可以使用 os.stat(file).st_birthtime
        ctime = os.path.getctime(file) 
        file_data.append({
            'file_path': file,
            'file_name': os.path.basename(file),
            'timestamp': ctime
        })

    # 3. 按时间先后顺序对文件进行排序（确保是按生成的先后顺序相减）
    file_data.sort(key=lambda x: x['timestamp'])

    # 4. 计算时间差并格式化数据
    results = []
    for i in range(len(file_data)):
        current = file_data[i]
        
        # 将时间戳转换为可读格式
        readable_time = datetime.fromtimestamp(current['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        # 计算与上一个文件的时间差（单位：秒）
        if i == 0:
            # 第一个文件没有上一个文件，时间差记为 0
            time_diff = 0.0  
        else:
            previous = file_data[i-1]
            time_diff = current['timestamp'] - previous['timestamp']
            
        results.append({
            '文件名': current['file_name'],
            '创建时间': readable_time,
            '生成时间(距上一个文件/秒)': round(time_diff, 3) # 保留3位小数
        })

    # 5. 使用 pandas 写入 Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"处理完成！共处理了 {len(files)} 个 JSON 文件。")
    print(f"结果已成功保存至: {output_excel_path}")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 请将这里的路径替换为你实际的文件夹路径和想要保存的Excel文件名
    TARGET_FOLDER = r"local_results"  # 路径前面加 r 防止转义字符报错
    OUTPUT_EXCEL = r"a.xlsx"
    
    calculate_json_generation_times(TARGET_FOLDER, OUTPUT_EXCEL)