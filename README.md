# 3D Geometry Feature Extraction Pipeline

本项目是一个自动化的 3D 模型（STL）分析与特征提取 API 服务。它通过多轴切片算法将 3D 模型降维为 2D SVG 视图，经过智能几何拟合与空间对齐后，提取出关键的机械特征（如底座范围、正向凸台、负向孔洞），并最终压缩输出为对 LLM（如 Dify）友好的极简 JSON 格式。

## 🚀 核心工作流 (Pipeline)

当通过 API 提交一个 `.stl` 文件时，系统将自动按以下顺序执行脚本：

1. **`stl2vsg11.py` (自适应切片)**
* 在 X、Y、Z 三个正交轴上对 STL 模型进行切片。
* 具备“自适应层厚”功能（最多 50 张切片），防止大尺寸模型生成过多文件导致内存溢出。


2. **`svg2svg.py` (矢量优化)**
* 提取原始切片中的路径，并利用 Shapely 和 numpy 进行智能几何拟合。
* 将复杂的离散点拟合为标准的 `<circle>` 或 `<polygon>`，大幅减少文件体积与冗余坐标。


3. **`vsg_merge.py` (视图合并)**
* 将各个轴向生成的大量离散 SVG 文件嵌套合并为单一的 `Out_X.txt` / `Out_Y.txt` / `Out_Z.txt` 文件，便于统一解析。


4. **`svg_json.py` / `svg_json11.py` (特征提取与可视化)**
* 本地几何计算引擎的核心。解析合并后的视图，执行全局 3D 坐标居中。
* 计算模型的三维边界框（Bounding Box）。
* 识别 Solid Base（实体基座层）、Positive Pillars（凸出圆柱）和 Negative Holes（凹陷孔洞）。
* 自动生成带深度映射的彩色三视图 (`View_X_Depth.png` / `View_Y_Depth.png` / `View_Z_Depth.png`) 和 3D 轴测图 (`Full_Isometric_View.png`)。


5. **`json_token.py` (JSON 压缩)**
* 移除多余的空格与格式，输出 `Full_Features_v33_minified.json`，最大程度节省 LLM 的 Token 消耗。



## 🔌 API 接口说明

项目基于 FastAPI 构建，提供以下核心接口：

### 1. 提交模型处理请求

* **Endpoint:** `POST /process_stl`
* **Content-Type:** `multipart/form-data`
* **描述:** 接收上传的 `.stl` 文件，自动清理上一次的缓存，触发完整的切片与提取流水线，并返回拼接后的三视图图片 URL (`Combined_Views.png`) 以及精简版的 JSON 几何特征。

### 2. 保存 LLM 分析结果

* **Endpoint:** `POST /save_result`
* **Content-Type:** `application/json`
* **描述:** 供 Dify 等外部大模型平台回调使用。接收双模型分析结果，自动根据 `model_name` 在本地分类归档保存文本分析报告 (`results_text/`) 和结构化评估数据 (`results_json/`)。

## 🛠️ 辅助工具

* **`stl_iou.py`**: 用于计算两个 STL 模型（如 Ground Truth 与生成模型）的三维 IoU（交并比）。内置自动形心对齐与随机点云采样算法。
* **`api_server.py`**: 主服务端代码，包含 422 数据验证拦截器与图片拼接 (`stitch_images`) 功能。
* **`json2txt.py`**: 遍历目录，将 `.json` 文件内容原封不动地复制为 `.txt` 文件的批量转换工具。