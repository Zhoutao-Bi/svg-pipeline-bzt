import os
import shutil

def clone_json_to_txt(root_dir):
    """
    遍历目录，将 .json 文件内容原封不动地复制到 .txt 文件中
    """
    print(f"[*] 正在搜索目录: {os.path.abspath(root_dir)}")
    count = 0

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                # 获取原文件完整路径
                json_path = os.path.join(subdir, file)
                # 构造目标文件路径 (只改后缀)
                txt_path = os.path.splitext(json_path)[0] + ".txt"
                
                try:
                    # 使用 shutil.copy2 可以保留原始文件的元数据（如创建时间）和内容
                    shutil.copy2(json_path, txt_path)
                    print(f"[OK] 已生成: {txt_path}")
                    count += 1
                except Exception as e:
                    print(f"[!] 无法处理 {file}: {e}")

    print(f"\n[+] 处理完毕，共生成了 {count} 个 TXT 文件。")

if __name__ == "__main__":
    # 默认处理当前脚本所在目录及其子目录
    clone_json_to_txt('./')