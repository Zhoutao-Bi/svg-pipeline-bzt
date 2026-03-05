import os
import sys
import glob
import subprocess

def main():
    input_dir = "data/01_input"
    if not os.path.exists(input_dir):
        print("⚠️ 未找到 data/01_input 目录。请先将你的 STP/STEP/STL 文件放进去。")
        return
        
    # 支持 3 种格式的扫描
    files = glob.glob(os.path.join(input_dir, "*.stp")) + \
            glob.glob(os.path.join(input_dir, "*.step")) + \
            glob.glob(os.path.join(input_dir, "*.stl"))
            
    if not files:
        print(f"⚠️ 在 {input_dir} 中未找到任何模型文件。")
        return
        
    projects = [os.path.splitext(os.path.basename(f))[0] for f in files]
    
    print("\n========== 🎛️ 项目流水线主控台 (0~4.5步) ==========")
    for i, proj in enumerate(projects):
        # 判断完成的标准变成了：有没有 txt_exports 文件夹
        status = "⏳ 未处理"
        if os.path.exists(f"data/03_output/{proj}/txt_exports"):
            status = "✅ 已全部完成"
        elif os.path.exists(f"data/02_temp/{proj}/stl"):
            status = "🔄 处理到一半"
            
        print(f"[{i+1}] {proj.ljust(15)} \t({status})")
        
    print("-" * 40)
    print(f"[{len(projects)+1}] 批量处理 (自动跳过已完成项目)")
    print(f"[0] 退出控制台")
    
    choice = input("\n👉 请输入要执行的操作序号: ")
    try:
        choice = int(choice)
    except:
        print("输入无效，已退出。")
        return
        
    # 移除了第 5 步脚本
    scripts = [
        "0_step2stl.py", 
        "1_stl2svg.py", 
        "2_svg2svg.py", 
        "3_vsg_merge.py", 
        "4_svg_slow.py", 
        "4.5_rename_svg.py"
    ]
    
    if choice == 0:
        print("退出成功。")
        return
        
    # 单选指定项目（强制重做）
    elif 1 <= choice <= len(projects):
        target = projects[choice-1]
        print(f"\n🚀 开始处理指定项目: {target} (忽略当前状态，覆盖执行)")
        for s in scripts:
            if not os.path.exists(s):
                print(f"找不到脚本 {s}，请检查。")
                continue
            subprocess.run([sys.executable, s, target])
            
    # 批量补全模式
    elif choice == len(projects) + 1:
        print("\n🚀 开始批量处理未完成的项目...")
        for proj in projects:
            # 检查 txt_exports 文件夹来判断是否完成
            if not os.path.exists(f"data/03_output/{proj}/txt_exports"):
                print(f"\n>>> 自动接力处理: {proj}")
                for s in scripts:
                    subprocess.run([sys.executable, s, proj])
            else:
                print(f"  [跳过] {proj} 已有 TXT 导出成果，视为已完成。")
                
if __name__ == "__main__":
    main()