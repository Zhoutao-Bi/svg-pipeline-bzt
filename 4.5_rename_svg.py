import os
import shutil
import sys

def copy_and_rename_specific_svg(src_dir, dest_dir):
    print(f"  [Debug] 检查源文件夹: {src_dir}")
    if not os.path.exists(src_dir):
        print(f"  [警告] 源文件夹不存在，已跳过: {src_dir}")
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    files_in_dir = os.listdir(src_dir)
    print(f"  [Debug] 源文件夹中共有 {len(files_in_dir)} 个文件/文件夹")

    processed_count = 0
    for filename in files_in_dir:
        src_path = os.path.join(src_dir, filename)
        
        if os.path.isfile(src_path) and filename.lower().endswith('.svg'):
            new_filename = filename[:-4] + '.txt'
            dest_path = os.path.join(dest_dir, new_filename)
            
            shutil.copy(src_path, dest_path)
            print(f"  > 成功导出: {filename} -> {new_filename}")
            processed_count += 1

    return processed_count

if __name__ == "__main__":
    base_out = "data/03_output"
    print(f"========== 开始运行 4.5 导出脚本 ==========")
    
    if os.path.exists(base_out):
        projects = [d for d in os.listdir(base_out) if os.path.isdir(os.path.join(base_out, d))]
        
        if len(sys.argv) > 1:
            projects = [p for p in projects if p == sys.argv[1]]
        
        if not projects:
            print("  [警告] 输出目录下为空或未匹配到指定项目！")
        
        for project_name in projects:
            print(f"\n>>> [4.5] 正在处理项目 [{project_name}] ...")
            
            target_folders = ["merged_svg", "optimized_svg"]
            export_folder = f"{base_out}/{project_name}/txt_exports"
            
            total_count = 0
            for folder_name in target_folders:
                source_folder = f"{base_out}/{project_name}/{folder_name}"
                print(f"\n  --- 扫描子目录: {folder_name} ---")
                
                count = copy_and_rename_specific_svg(source_folder, export_folder)
                total_count += count
            
            if total_count == 0:
                print(f"  [提示] 项目 [{project_name}] 没有处理任何文件！")
            else:
                print(f"\n  [完成] 项目 [{project_name}] 共成功导出了 {total_count} 个 TXT 文件。")
    else:
        print(f"⚠️ [错误] 找不到目录: {base_out}")
        
    print(f"\n========== 4.5 脚本运行结束 ==========")