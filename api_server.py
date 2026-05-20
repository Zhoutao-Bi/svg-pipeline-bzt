import os
import shutil
import subprocess
import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Optional

# 锁定绝对路径，防止 FastAPI 后台运行迷路
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
app.mount("/images", StaticFiles(directory=BASE_DIR), name="images")

# ==========================================
# 422 报错“透视眼”拦截器
# ==========================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 如果需要读取 body 可以取消注释下一行，但需注意可能引发阻塞
    # body = await request.body() 
    print(f"\n{'!'*40}")
    print(f"[致命错误] 422 数据验证失败！")
    print(f"[-] 报错具体原因:\n{exc.errors()}")
    print(f"{'!'*40}\n")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# 数据模型
class DifyResult(BaseModel):
    filename: str
    model_name: str
    text_content: Any  
    json_content: Any  
    is_refined: Optional[bool] = False  # 用于区分基础分析/深度提炼

class RefineRequest(BaseModel):
    filename: str

# ==========================================
# 拼图时：由于不知道模型名，先用原文件名做专属暂存防覆盖
# ==========================================
def stitch_images(clean_fn: str):
    img_names = [ "View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png" ]
    images = [ ]
    valid_names = [ ]
    
    for name in img_names:
        full_path = os.path.join(BASE_DIR, name)
        if os.path.exists(full_path):
            images.append(Image.open(full_path))
            valid_names.append(name)
            
    if not images: return None

    header_height = 150 
    total_width = sum( [ img.width for img in images ] )
    max_height = max( [ img.height for img in images ] ) + header_height

    combined_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined_img)

    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except IOError:
        font = ImageFont.load_default()

    x_offset = 0
    for img, name in zip(images, valid_names):
        combined_img.paste(img, (x_offset, header_height))
        draw.text((x_offset + 50, 40), name, fill=(0, 0, 0), font=font)
        x_offset += img.width
    
    # 暂存图命名：加入专属文件名防混淆
    combined_name = f"Combined_Views_{clean_fn}.png"
    combined_path = os.path.join(BASE_DIR, combined_name)
    combined_img.save(combined_path)
    return combined_name

# ==========================================
# 阶段 1：基础解析
# ==========================================
@app.post("/process_stl")
async def process_stl(file: UploadFile = File(...)):
    print(f"\n{'='*40}\n[*] 收到新任务: {file.filename}")
    
    clean_fn = file.filename.strip().replace(".stl", "").replace(".STL", "")
    
    folders_to_clean = [ "Out_X", "Out_Y", "Out_Z", "Out_X_new", "Out_Y_new", "Out_Z_new" ]
    files_to_clean = [
        "Out_X.txt", "Out_Y.txt", "Out_Z.txt", "Full_Features_v33_minified.json","Full_Features_v33.json",
        "Full_Features_v34.json", "Full_Features_v34_minified.json", 
        "View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png"
    ]
    
    for folder in folders_to_clean:
        f_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(f_path): shutil.rmtree(f_path)
            
    for f_name in files_to_clean:
        f_path = os.path.join(BASE_DIR, f_name)
        if os.path.exists(f_path): os.remove(f_path)
    
    stl_path = os.path.join(BASE_DIR, "current_task.stl")
    with open(stl_path, "wb+") as f:
        f.write(await file.read())
    
    # 执行基础解析脚本 (可根据需要调整脚本列表)
    base_scripts = [ "stl2vsg11.py", "svg2svg.py", "vsg_merge.py", "svg_json_v5.py", "json_token.py" ]
    for script in base_scripts:
        print(f"[*] 正在执行基础解析: {script}")
        subprocess.run([ "python", script ], check=True, cwd=BASE_DIR)

    # 传参调用拼图
    combined_name = stitch_images(clean_fn)
    
    # 注意：Ngrok 链接需确认是否有效，建议后续转入环境变量或动态获取
    base_url = "https://fraction-slot-relax.ngrok-free.dev" 
    img_url = f"{base_url}/images/{combined_name}" if combined_name else ""
    
    json_path = os.path.join(BASE_DIR, "Full_Features_v33_minified.json")
    final_data = "{}"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            final_data = f.read()

    return {
        "status": "success",
        "step": "base_analysis",
        "filename": file.filename,
        "features": final_data,
        "combined_url": img_url
    }

# ==========================================
# 阶段 2：深度提炼
# ==========================================
@app.post("/refine_features")
async def refine_features():
    print(f"\n{'='*40}\n[*] 收到深度分析请求")
    
    try:
        refine_scripts = [ "feature_refiner2.py", "json_token2.py" ]
        for script in refine_scripts:
            print(f"[*] 正在执行深度提炼: {script}")
            subprocess.run([ "python", script ], check=True, cwd=BASE_DIR)

        json_path = os.path.join(BASE_DIR, "Full_Features_v34_minified.json")
        refined_data = "{}"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                refined_data = f.read()

        return {
            "status": "success",
            "step": "refined_analysis",
            "refined_features": refined_data
        }
    except Exception as e:
        print(f"[错误] 深度分析失败: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ==========================================
# 收到模型名字时：分类归档，并剪切暂存图进目标文件夹
# ==========================================
# ==========================================
# 收到基础分析结果时存档
# ==========================================
@app.post("/save_result")
async def save_base_result(data: DifyResult):
    data.is_refined = False  # 强制标记为基础分析
    return await process_save_logic(data)

# ==========================================
# 收到深度提炼结果时存档 (你报错找不到的就是这个)
# ==========================================
@app.post("/save_result_refined")
async def save_refined_result(data: DifyResult):
    data.is_refined = True   # 强制标记为深度提炼
    return await process_save_logic(data)

# --- 把公共的核心存档逻辑抽离出来，避免代码重复 ---
async def process_save_logic(data: DifyResult):
    type_label = "深度提炼" if data.is_refined else "基础分析"
    print(f"\n{'='*40}\n[*] 收到大模型结果 ({type_label})，触发存档: {data.filename}")
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_fn = data.filename.strip().replace(".stl", "").replace(".STL", "")
    suffix = "_refined" if data.is_refined else ""
        
    # 1. 存档文本
    text_dir = os.path.join(BASE_DIR, f"results_text{suffix}", data.model_name)
    os.makedirs(text_dir, exist_ok=True)
    text_path = os.path.join(text_dir, f"{data.model_name}_{clean_fn}_text.txt")
    
    final_text = data.text_content if isinstance(data.text_content, str) else json.dumps(data.text_content, ensure_ascii=False, indent=4)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"处理时间: {now}\n类型: {type_label}\n原始文件: {data.filename}\n使用模型: {data.model_name}\n{'-'*40}\n{final_text}")
        
    # 2. 存档 JSON
    json_dir = os.path.join(BASE_DIR, f"results_json{suffix}", data.model_name)
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f"{data.model_name}_{clean_fn}_json.txt")
    
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(f"处理时间: {now}\n类型: {type_label}\n原始文件: {data.filename}\n使用模型: {data.model_name}\n{'-'*40}\n")
        f.write(json.dumps(data.json_content, ensure_ascii=False, indent=4))

    # 3. 存档拼图：用剪切的方式直接把暂存图收走
    source_img = os.path.join(BASE_DIR, f"Combined_Views_{clean_fn}.png")
    
    if os.path.exists(source_img):
        img_archive_dir = os.path.join(BASE_DIR, f"results_images{suffix}", data.model_name)
        os.makedirs(img_archive_dir, exist_ok=True)
        archive_name = f"{data.model_name}_{clean_fn}_combined.png"
        archive_path = os.path.join(img_archive_dir, archive_name)
        shutil.move(source_img, archive_path)
        print(f"[✅ 存档成功] 拼图已被收纳至: {archive_path}")
    else:
        print(f"[❌ 存档失败] 找不到刚刚暂存的图片: {source_img}")
        if data.is_refined:
            print(f"[i] 提示: 深度分析阶段找不到原始拼图是正常现象，可能已在基础分析时被归档。")
        
    return {"status": "success", "saved_to": f"results_...{suffix}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)