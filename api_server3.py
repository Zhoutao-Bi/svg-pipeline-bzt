import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from typing import Optional, Any
from fastapi import Request
app = FastAPI()

# ==========================================
# 🛑 自定义路径配置区（请根据你电脑的实际情况修改）
# ==========================================

# 1. 你的 local_results 文件夹绝对路径 (存放特征 JSON/TXT 和 拼接图片的目录)
LOCAL_RESULTS_DIR = r"D:\Github_Repository\20260410_stl2json_vsg\local_results7"

# 2. 【新增】全局 Grasp 文件的绝对路径 (写死这个路径，每次都读它)
GLOBAL_GRASP_FILE = r"D:\Github_Repository\20260410_stl2json_vsg\bsp_grasp.txt"

# 3. 你的 ngrok 域名 (每次重启 ngrok 请更新这里，结尾不要加斜杠)
NGROK_URL = "https://fraction-slot-relax.ngrok-free.dev"

# ==========================================

# 挂载静态目录，让 Dify 可以通过网络链接访问你本地的图片
app.mount("/images", StaticFiles(directory=LOCAL_RESULTS_DIR), name="images")

# 定义 Dify 传过来的数据格式 (现在只需要传一个名字了！)
class ModelRequest(BaseModel):
    txt_filename: str     # Dify传过来的，例如 "easy_1_features.txt"

@app.post("/get_local_data")
async def get_local_data(req: ModelRequest):
    print(f"\n{'='*40}")
    print(f"[*] 收到 Dify 请求，要查找的模型是: {req.txt_filename}")
    
    # 1. 提取基础模型名 (把 "easy_1_features.txt" 变成 "easy_1")
    base_name = req.txt_filename.replace("_features.txt", "").replace(".txt", "")
    
    # 2. 拼接本地真正的文件路径
    img_filename = f"{base_name}_combined.png"
    json_filename = f"{base_name}_features.txt" # 如果你是存成 .txt，这里改成 .txt
    
    img_path = os.path.join(LOCAL_RESULTS_DIR, img_filename)
    json_path = os.path.join(LOCAL_RESULTS_DIR, json_filename)

    # 3. 检查图片是否存在，生成图片 URL
    img_url = ""
    if os.path.exists(img_path):
        img_url = f"{NGROK_URL}/images/{img_filename}"
        print(f"[+] 找到专属图片: {img_filename}")
    else:
        print(f"[-] 警告: 找不到专属图片 {img_path}")

    # 4. 读取专属特征文本内容
    features_text = ""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            features_text = f.read()
        print(f"[+] 读取专属特征完成: {json_filename}")
    else:
        features_text = f"【系统提示】在本地找不到特征文件: {json_filename}"
        print(f"[-] 警告: 找不到特征文件 {json_path}")

    # 5. 【修改点】直接读取全局的 Grasp 文本内容
    grasp_text = ""
    if os.path.exists(GLOBAL_GRASP_FILE):
        with open(GLOBAL_GRASP_FILE, "r", encoding="utf-8") as f:
            grasp_text = f.read()
        print(f"[+] 读取全局 Grasp 文件完成: bsp_grasp.txt")
    else:
        grasp_text = f"【系统提示】在本地找不到全局 Grasp 文件: {GLOBAL_GRASP_FILE}"
        print(f"[-] 警告: 找不到全局 Grasp 文件 {GLOBAL_GRASP_FILE}")

    print(f"{'='*40}\n")

# 6. 把所有数据打包返回给 Dify
    return {
        "status": "success",
        "model_name": base_name,
        "img_url": img_url,
        "features_text": features_text,
        "grasp_text": grasp_text
    }


class SaveRequest(BaseModel):
    filename: str
    model_name: str
    text_content: str
    json_content: Optional[Any] = None  # 允许接收 null

@app.post("/save_result_refined")
async def save_result_refined(req: Request):
    print(f"\n{'='*40}")
    print(f"[*] 收到 Dify 发来的请求！")
    
    try:
        # 1. 强制获取并打印 Dify 发过来的原始 JSON 数据
        data = await req.json()
        print("    [调试信息] 实际收到的原始数据为:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
        # 2. 安全提取字段 (用 .get()，如果没传也不会报错)
        filename = data.get("filename", "unknown_file")
        model_name = data.get("model_name", "unknown_model")
        text_content = data.get("text_content", "")
        
        # 💡 核心修复：如果 text_content 是个字典或列表，把它强转成字符串
        if isinstance(text_content, (dict, list)):
            text_content = json.dumps(text_content, ensure_ascii=False, indent=2)
            
        # 3. 处理文件名
        base_name = str(filename).replace("_features.txt", "").replace(".txt", "")
        save_filename = f"{base_name}_refined_serial.txt" 
        save_path = os.path.join(LOCAL_RESULTS_DIR, save_filename)
        
        # 4. 写入本地文件
        with open(save_path, "w", encoding="utf-8") as f:
            # 无论 text_content 是什么，都强制转成字符串写入
            f.write(str(text_content))
            
        print(f"[+] 保存成功！文件已生成: {save_path}")
        print(f"{'='*40}\n")
        
        return {
            "status": "success", 
            "message": f"Successfully saved as {save_filename}"
        }
    
    except Exception as e:
        print(f"[-] 保存失败，出现了意料之外的错误: {e}")
        print(f"{'='*40}\n")
        return {
            "status": "error", 
            "message": str(e)
        }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)