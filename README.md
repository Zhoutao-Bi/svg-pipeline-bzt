# 3D Model to SVG Pipeline & SAM3 Segmentation
**3D 模型特征提取、全向 SVG 切片与 SAM3 语义分割自动化流水线**

本项目是一套完整的自动化工具链，旨在将 3D CAD 模型（支持 `.stp`, `.step`, `.stl`）转换为高精度的 2D SVG 切片，并对切片中的几何特征进行智能拟合与跨层降维优化。此外，结合最前沿的 SAM3 (Segment Anything Model 3) 视觉大模型，实现对模型渲染图的智能部件识别、面积阈值过滤与归一化坐标提取。

---

## update 
### time 20260311
### author Terry Bi
### Email bizhoutao.tery@gmail.com
delete some files.

the lastest pipline:
1.stl2svg:sli the STL model(Don't worry about the slice thickness)
2.svg2svg:from point data to commond data
3.svg_merge: merging all svg 
4.svg_json: use some algorithms to let automation
5.json_token: delete some space to reduce token consumption
6.stl_iou:If you use the MLLM generate the .sCAD code source, you can use this program to cacluate the volume error










## ✨ 核心功能

* **多格式兼容**：自动识别 `.stp`/`.step` 并使用 `gmsh` 引擎转换为高质量 STL 网格，原生 `.stl` 文件自动直通。
* **全向精准切片**：基于 `trimesh` 沿 X、Y、Z 三轴生成带物理单位的正交投影切片，并同步输出无留白的极坐标渲染图。
* **几何智能拟合**：将由无数散点组成的切片路径（Path），智能识别并拟合为标准的圆形 (`<circle>`) 和多边形 (`<polygon>`)，大幅压缩体积。
* **跨层特征折叠**：通过几何指纹算法，将贯穿多个切片层的相同孔洞/圆柱特征进行去重与合并，生成带有 `data-layer-range`（层级范围）属性的极致优化 SVG。
* **SAM3 智能语义分割**：基于视觉大模型，支持自然语言 Prompt 交互式提取模型大类与内部细小零件，支持独立标签的面积阈值过滤与空间位置去重，最终输出像素级色彩掩膜与精准归一化坐标数据。
* **交互式主控台**：提供 `6_run_menu.py` CLI 界面，支持批量断点续传（智能跳过已完成项目）与单项目强制重跑。

---

## 📁 目录架构与数据流

运行代码后，系统会自动在根目录维护以下标准数据结构，实现**输入、中间态、输出**的完全隔离：

```text
project_root/
├── data/
│   ├── 01_input/             <- 📥 把你的 .stp, .step, .stl 扔到这里
│   ├── 02_temp/              <- 🔄 系统自动生成的中间件隔离区
│   │   └── [项目名]/
│   │       ├── stl/          (网格文件)
│   │       ├── raw_slices/   (原始三轴切片与渲染图)
│   │       └── fit_slices/   (几何拟合后的切片)
│   └── 03_output/            <- 📦 最终成果输出区
│       └── [项目名]/
│           ├── merged_svg/   (合并嵌套图层后的 SVG)
│           ├── optimized_svg/(跨层降维优化后的 SVG)
│           ├── txt_exports/  (最终导出的 TXT 格式数据)
│           └── sam3_results/ (SAM3 视觉分割掩膜图与坐标报表)
├── 0_step2stl.py
├── ...
└── 6_run_menu.py             <- 🎛️ 主控台脚本
