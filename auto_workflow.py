import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# 锁定绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 配置区 =================
# 存放待处理 STL 模型的文件夹
INPUT_DIR = os.path.join(BASE_DIR, "222")
# 存放所有处理结果和 Excel 报表的统一文件夹
OUTPUT_DIR = os.path.join(BASE_DIR, "local_results8")

# 基础解析需要运行的脚本
BASE_SCRIPTS = [
    "stl2vsg11.py", 
    "svg2svg.py", 
    "vsg_merge.py", 
    "svg_json_v6.py", 
    "json_token.py",
    "json_token_token.py"
]
# ==========================================

def clean_workspace():
    """清理上一轮跑完留下的临时文件夹和文件"""
    folders_to_clean = ["Out_X", "Out_Y", "Out_Z", "Out_X_new", "Out_Y_new", "Out_Z_new"]
    files_to_clean = [
        "Out_X.txt", "Out_Y.txt", "Out_Z.txt", 
        "Full_Features_v33_minified.json", "Full_Features_v33.json","Full_Features_v33_minified2.json",
        "Full_Features_v34.json", "Full_Features_v34_minified.json", 
        "View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png",
        "current_task.stl"
    ]
    
    for folder in folders_to_clean:
        f_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(f_path): shutil.rmtree(f_path)
            
    for f_name in files_to_clean:
        f_path = os.path.join(BASE_DIR, f_name)
        if os.path.exists(f_path): os.remove(f_path)

def stitch_images(clean_fn: str):
    """把三个视角的深度图拼合在一块"""
    img_names = ["View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png"]
    images, valid_names = [], []
    
    for name in img_names:
        full_path = os.path.join(BASE_DIR, name)
        if os.path.exists(full_path):
            images.append(Image.open(full_path))
            valid_names.append(name)
            
    if not images: return None

    header_height = 150 
    total_width = sum([img.width for img in images])
    max_height = max([img.height for img in images]) + header_height

    combined_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined_img)

    try: font = ImageFont.truetype("arial.ttf", 80)
    except IOError: font = ImageFont.load_default()

    x_offset = 0
    for img, name in zip(images, valid_names):
        combined_img.paste(img, (x_offset, header_height))
        draw.text((x_offset + 50, 40), name, fill=(0, 0, 0), font=font)
        x_offset += img.width
    
    combined_name = f"Combined_Views_{clean_fn}.png"
    combined_path = os.path.join(BASE_DIR, combined_name)
    combined_img.save(combined_path)
    return combined_path

def archive_results(clean_fn, combined_img_path):
    """
    修改后：将结果直接放入 OUTPUT_DIR，不再创建每个模型名字的子目录
    """
    # 确保输出总目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 归档 JSON
    json_source = os.path.join(BASE_DIR, "Full_Features_v33_minified2.json")
    if os.path.exists(json_source):
        # 目标路径：local_results/模型名_features.json
        dest_json = os.path.join(OUTPUT_DIR, f"{clean_fn}_features.json")
        shutil.move(json_source, dest_json)
        
    # 2. 归档拼图
    if combined_img_path and os.path.exists(combined_img_path):
        # 目标路径：local_results/模型名_combined.png
        dest_img = os.path.join(OUTPUT_DIR, f"{clean_fn}_combined.png")
        shutil.move(combined_img_path, dest_img)

def process_single_stl(stl_path):
    """处理单个 STL 文件的完整流程"""
    clean_fn = stl_path.stem 
    clean_workspace()
    
    target_stl = os.path.join(BASE_DIR, "current_task.stl")
    shutil.copy(stl_path, target_stl)
    
    for script in BASE_SCRIPTS:
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(script_path):
            return f"失败: 找不到脚本 {script}"
            
        try:
            subprocess.run([sys.executable, script], check=True, cwd=BASE_DIR)
        except subprocess.CalledProcessError as e:
            return f"失败: 运行 {script} 时崩溃"

    combined_img_path = stitch_images(clean_fn)
    archive_results(clean_fn, combined_img_path)
    
    return "成功"

def run_batch_pipeline():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    folder_path = Path(INPUT_DIR)
    stl_files = [f for f in folder_path.rglob('*.[sS][tT][lL]')] 
    
    if not stl_files:
        print(f"❌ 在 {INPUT_DIR} 目录下没有找到任何 STL 文件！")
        return

    print(f"🔎 共找到 {len(stl_files)} 个 STL 文件，启动自动化流水线...\n" + "="*40)
    
    log_records = []
    total_start = time.time()
    
    for index, stl_file in enumerate(stl_files, 1):
        print(f"\n[{index}/{len(stl_files)}] 开始处理模型: {stl_file.name}")
        
        start_time = time.time()
        status = process_single_stl(stl_file)
        end_time = time.time()
        
        duration = end_time - start_time
        
        log_records.append({
            "处理顺序": index,
            "模型文件名": stl_file.name,
            "处理状态": status,
            "耗时 (秒)": round(duration, 2)
        })
        
        if status == "成功":
            print(f"  ✅ 处理成功！耗时: {round(duration, 2)} 秒")
        else:
            print(f"  ⚠️ 处理异常: {status}")

    total_duration = time.time() - total_start
    log_records.append({
        "处理顺序": "---", 
        "模型文件名": "【总计】", 
        "处理状态": f"共扫描 {len(stl_files)} 个", 
        "耗时 (秒)": round(total_duration, 2)
    })
    
    # 生成 Excel 耗时报告
    excel_path = os.path.join(OUTPUT_DIR, "模型批量处理耗时报告.xlsx")
    try:
        df = pd.DataFrame(log_records)
        df.to_excel(excel_path, index=False)
        print(f"\n{'='*40}\n🚀 全部任务结束！总耗时: {round(total_duration, 2)} 秒。")
        print(f"📊 结果文件与报表均已保存在: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n❌ Excel 保存失败: {e}")

if __name__ == "__main__":
    run_batch_pipeline()