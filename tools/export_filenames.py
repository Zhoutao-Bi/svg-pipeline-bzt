import os
import csv

def export_filenames_to_csv(folder_path, output_csv_name):
    """
    读取指定文件夹下的所有文件名，并保存到 CSV 文件中。
    
    :param folder_path: 要读取的文件夹路径
    :param output_csv_name: 输出的 CSV 文件名
    """
    # 确保存放路径存在
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹路径 '{folder_path}'")
        return

    # 获取文件夹下的所有内容，并过滤出纯文件（排除子文件夹）
    all_items = os.listdir(folder_path)
    file_names = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]

    # 如果没有找到文件，提前退出
    if not file_names:
        print(f"⚠️ 文件夹 '{folder_path}' 中没有找到任何文件。")
        return

    # 将文件名写入 CSV 文件
    try:
        # newline='' 防止在 Windows 上出现多余的空行，encoding='utf-8-sig' 确保中文在 Excel 中不乱码
        with open(output_csv_name, mode='w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            
            # 写入表头（第一行）
            writer.writerow(['文件名称'])
            
            # 逐行写入文件名
            for name in file_names:
                writer.writerow([name])
                
        print(f"✅ 成功！共读取了 {len(file_names)} 个文件，已保存至 '{output_csv_name}'")
        
    except Exception as e:
        print(f"❌ 写入 CSV 时发生错误: {e}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 替换为你想要读取的文件夹路径 
    # (例如: 'D:/my_folder' 或 './target_folder')
    # 使用 '.' 代表当前代码所在的文件夹
    TARGET_FOLDER = './local_results' 
    
    # 2. 生成的 CSV 文件名
    OUTPUT_CSV = 'file_names_list.csv' 
    
    # ============================================
    
    print("开始处理...")
    export_filenames_to_csv(TARGET_FOLDER, OUTPUT_CSV)