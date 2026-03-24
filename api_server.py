import os
import shutil
import subprocess
import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
from datetime import datetime
from pydantic import BaseModel
from typing import Any

app = FastAPI()
app.mount("/images", StaticFiles(directory="."), name="images")

# ==========================================
# 422 报错“透视眼”拦截器
# ==========================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(f"\n{'!'*40}")
    print(f"[致命错误] 422 数据验证失败！")
    print(f"[-] Dify 发过来的原始数据:\n{body.decode(errors='replace')}")
    print(f"[-] 报错具体原因:\n{exc.errors()}")
    print(f"{'!'*40}\n")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": body.decode(errors='replace')})

# ==========================================
# 定义 Dify 传回来的双模型 JSON 数据格式
# ==========================================
class DifyResult(BaseModel):
    filename: str
    model_name: str
    text_content: Any  # 🌟 修复1：改为 Any，允许接收 LLM1 的 JSON 字典
    json_content: Any  # 接收 LLM 2 的结构化 JSON 特征

def stitch_images():
    """将三张视图拼成一张长图"""
    img_names = ["View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png"]
    images = []
    for name in img_names:
        if os.path.exists(name):
            images.append(Image.open(name))
        else:
            print(f"[警告] 找不到单视图图片: {name}")
    
    if not images:
        print("[致命错误] 没有任何深度图被生成，无法执行拼图！(可能是切片失败)")
        return None

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    combined_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    combined_name = "Combined_Views.png"
    combined_img.save(combined_name)
    print(f"[+] 拼图成功，已保存为: {combined_name}")
    return combined_name

@app.post("/process_stl")
async def process_stl(file: UploadFile = File(...)):
    print(f"\n{'='*40}\n[*] 收到新任务: {file.filename}")
    
    print("[*] 正在大扫除旧数据...")
    folders_to_clean = ["Out_X", "Out_Y", "Out_Z", "Out_X_new", "Out_Y_new", "Out_Z_new"]
    files_to_clean = [
        "Out_X.txt", "Out_Y.txt", "Out_Z.txt", 
        "Full_Features_v33.json", "Full_Features_v33_minified.json", 
        "Combined_Views.png", "View_X_Depth.png", "View_Y_Depth.png", "View_Z_Depth.png"
    ]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            
    for f_name in files_to_clean:
        if os.path.exists(f_name):
            os.remove(f_name)
    print("[*] 环境清理完毕，开始处理新模型！")
    
    with open("current_task.stl", "wb+") as f:
        f.write(await file.read())
    
    for script in ["stl2vsg11.py", "svg2svg.py", "vsg_merge.py", "svg_json_v2.py", "hole_refiner.py","json_token.py"]:
        print(f"[*] 正在执行: {script}")
        subprocess.run(["python", script], check=True)

    combined_path = stitch_images()
    base_url = "https://tandra-gaiterless-jinny.ngrok-free.dev"
    img_url = f"{base_url}/images/Combined_Views.png" if combined_path else ""
    
    final_data = "{}"
    if os.path.exists("Full_Features_v33_minified.json"):
        with open("Full_Features_v33_minified.json", "r", encoding="utf-8") as f:
            final_data = f.read()

    return {
        "status": "success",
        "features": final_data,
        "combined_url": img_url
    }

# ==========================================
# 接收双模型处理结果并分别保存到本地
# ==========================================
@app.post("/save_result")
async def save_result(data: DifyResult):
    print(f"\n{'='*40}\n[*] 收到 Dify 返回的双模型结果，关联文件: {data.filename}")
    
    now = datetime.now()
    exact_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    clean_filename = data.filename.strip()
    if clean_filename.lower().endswith('.stl'):
        clean_filename = clean_filename[:-4]
        
    # ------------------------------------------
    # 1. 保存 LLM 1 的分析报告 (自动兼容纯文本和 JSON 对象)
    # ------------------------------------------
    text_dir = os.path.join("results_text", data.model_name)
    os.makedirs(text_dir, exist_ok=True)
    text_file_path = os.path.join(text_dir, f"{data.model_name}_{clean_filename}_text.txt")
    
    # 🌟 修复2：判断如果传过来的是字典，就帮它格式化排版
    if isinstance(data.text_content, str):
        final_text = data.text_content
    else:
        final_text = json.dumps(data.text_content, ensure_ascii=False, indent=4)
        
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(f"处理时间: {exact_time}\n")
        f.write(f"原始文件: {data.filename.strip()}\n")
        f.write(f"使用模型: {data.model_name}\n")
        f.write(f"数据类型: 深度分析报告 (Text/JSON)\n")
        f.write("-" * 40 + "\n")
        f.write(final_text)
    print(f"[+] 文本分析已保存至: {text_file_path}")
        
    # ------------------------------------------
    # 2. 保存 LLM 2 的结构化 JSON 特征
    # ------------------------------------------
    json_dir = os.path.join("results_json", data.model_name)
    os.makedirs(json_dir, exist_ok=True)
    json_file_path = os.path.join(json_dir, f"{data.model_name}_{clean_filename}_json.txt")
    
    formatted_json = json.dumps(data.json_content, ensure_ascii=False, indent=4)
    with open(json_file_path, "w", encoding="utf-8") as f:
        f.write(f"处理时间: {exact_time}\n")
        f.write(f"原始文件: {data.filename.strip()}\n")
        f.write(f"使用模型: {data.model_name}\n")
        f.write(f"数据类型: 结构化特征评估数据 (JSON)\n")
        f.write("-" * 40 + "\n")
        f.write(formatted_json)
    print(f"[+] 结构化数据已保存至: {json_file_path}")
        
    return {
        "status": "success", 
        "message": "Text and JSON results saved successfully!",
        "saved_files": [text_file_path, json_file_path]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)