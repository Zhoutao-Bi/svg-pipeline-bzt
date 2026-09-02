# STL 特征提取 Python 消融流水线

本项目是一套本地 Python 消融流水线。模型调用通过已登录的 Codex CLI 完成，复用 ChatGPT/OAuth 会话，不需要在项目中保存 OpenAI API Key。

默认模型配置：

- 模型：`gpt-5.6-luna`
- 推理强度：`medium`
- 认证：Codex CLI 的 ChatGPT/OAuth 登录

## 三种实验模式

| `run_experiment.py --mode` | 处理方式 | 独立入口 |
| --- | --- | --- |
| `visual-only` | 只把 X/Y/Z 三视图交给 Luna | `run_visual_only.py` |
| `visual-json-parallel` | 一次调用同时提供三视图和几何 JSON | `run_visual_json_parallel.py` |
| `visual-json-serial` | 第一轮读取粗切三视图，第二轮使用几何 JSON 矫正；可启用动态细切 | `run_visual_json_serial.py` |
| `all` | 依次执行以上三种实验 | — |

## 安装与 OAuth 登录

安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

确认 Codex CLI 已安装，然后使用 ChatGPT 登录：

```bash
codex --version
codex login
codex login status
```

`codex login status` 必须显示 `Logged in using ChatGPT`。流水线只调用 Codex CLI，不会读取、复制或提交 OAuth 缓存文件。

## 运行

把 STL 文件放入 `input_stl/`，然后运行：

```bash
python run_experiment.py --mode visual-only
python run_experiment.py --mode visual-json-parallel
python run_experiment.py --mode visual-json-serial
python run_experiment.py --mode all
```

输出保存在 `results/`，包含：

- `{零件名}_combined.png`：三视图拼合图。
- `{零件名}_features.json`：本地几何特征。
- `{零件名}_refined_luna_*.txt`：Luna 结构化结果。
- `metrics_*.csv`：耗时和 token 统计。

## 几何处理链

每个 STL 会先执行本地几何流水线：

1. `stl_to_svg.py`：沿 X/Y/Z 三轴切片。
2. `optimize_svg.py`：简化和拟合 SVG 几何。
3. `merge_svg.py`：合并各轴切片。
4. `extract_features.py`：提取几何特征并生成深度视图。
5. `minify_features.py`：压缩特征 JSON。

`pipeline.py` 负责几何调度和结果读写，`codex_client.py` 负责 OAuth 模型调用。

### 几何 JSON 2.0

`extract_features.py` 保留兼容字段 `Positive_Pillars` / `Negative_Holes`，并新增：

- `Recognized_Features`：把低层轮廓进一步识别为通孔、盲孔、沉孔、槽、凹槽、凸台、加强筋、台阶凸台等语义特征；每项带置信度和可审计证据。
- `Feature_Relationships`：使用形状感知的三维包围盒记录正交相交、同轴重叠、投影视图重叠和切削关系。跨轴相交不再按旧规则直接删除其中一个候选。
- `Feature_Patterns`：识别线性阵列、圆周阵列和重复特征组。
- `Profile_Transitions`：从实体层尺寸变化识别外轮廓台肩。

形状分类同时覆盖圆、椭圆、胶囊、矩形、正方形、三角形、五边形和六边形。动态细切选区会优先匹配这些语义特征，因此第一 Agent 给出的“槽、凹槽、凸台、加强筋、沉孔”等类型可以直接定位到 JSON，而不再只能匹配孔/柱。

可对已有结果做不调用模型的离线评估：

```bash
python tools/evaluate_feature_recognition.py results/terra --summary-only
```

对动态串行最终输出使用与 29 件对比表相同的一对一 GT 匹配器：

```bash
python tools/evaluate_dynamic_results.py \
  --results-dir results/dynamic \
  --ground-truth /path/to/exp1.xlsx \
  --output results/dynamic/evaluation.json
```

## 动态切片串行流程

动态模式只改变 `visual-json-serial` 的两轮 Agent 流程：

1. 先用 `0.1 mm`（可配置）的粗切渲染图让第一 Agent 判断几何特征及其暂定用途。
2. 将第一 Agent 的孔/柱/槽等几何特征坐标与粗切 JSON 匹配，生成可审计的“整平面深度范围”JSON；选区不依赖第一 Agent 的暂定用途，避免一次用途误判导致后续证据缺失。
3. 只在这些范围内按 `0.01 mm` 重新切片，重新提取 JSON 并渲染深度图。
4. 第二 Agent 同时接收细切图、细切 JSON、选区 JSON 和第一 Agent 结论，输出最终判断。

例如只测试一个样本：

```bash
SLICE_MODE=dynamic \
INPUT_STL_DIR=/path/to/inputs \
INPUT_STL_PATTERN=easy_1.stl \
RESULTS_DIR=results/dynamic_easy_1 \
python run_experiment.py --mode visual-json-serial
```

动态模式额外保存：`*_first_agent.json`、`*_fine_slice_plan.json`、
`*_fine_features.json` 和 `*_fine_combined.png`，便于复现实验和核查选区。

## TER-71 配对消融分析

`tools/analyze_ter71_experiments.py` 将已发布的三流程、动态、拓扑和最终
29 件工作簿组合成统一六级对照。脚本会校验 29 件样本与 86 个 GT 特征完全
一致，执行以零件为簇的配对 bootstrap，并从最终细切计划统计选择率、耗时和
token 长尾：

```bash
python tools/analyze_ter71_experiments.py \
  --three-flow /path/to/three_flow.xlsx \
  --dynamic /path/to/dynamic_comparison.xlsx \
  --topology /path/to/topology_comparison.xlsx \
  --final /path/to/final_comparison.xlsx \
  --final-results-dir /path/to/final_all29 \
  --output-dir results/ter71_analysis
```

输出包含 Markdown 报告、六级汇总 CSV、配对区间 CSV 和带输入文件 SHA-256
的 JSON manifest。默认使用固定种子 71071 和 20,000 次 bootstrap。

## 环境变量

- `CODEX_BIN`：Codex CLI 命令，默认 `codex`。
- `CODEX_MODEL`：默认 `gpt-5.6-luna`。
- `CODEX_REASONING_EFFORT`：默认 `medium`。
- `CODEX_TIMEOUT`：单次 Codex 调用超时秒数，默认 `600`。
- `CODEX_CONCURRENCY`：并发模型调用数，默认 `1`。
- `INPUT_STL_DIR`：STL 输入目录，默认 `input_stl/`。
- `INPUT_STL_PATTERN`：STL 文件匹配模式，默认全部；例如 `easy_1.stl`。
- `RESULTS_DIR`：输出目录，默认 `results/`。
- `SLICE_MODE`：`coarse`、`fine` 或 `dynamic`。
- `DYNAMIC_COARSE_LAYER_HEIGHT`：动态模式粗切目标层厚，默认 `0.1`。
- `DYNAMIC_COARSE_MAX_SLICES`：每轴粗切最大层数，默认 `30`。
- `DYNAMIC_FINE_LAYER_HEIGHT`：选区细切层厚，默认 `0.01`。
- `DYNAMIC_RANGE_MARGIN`：粗 JSON 匹配区间两端最小余量，默认 `0.2 mm`；实际余量不小于该轴粗切间距。
- `DYNAMIC_FALLBACK_HALF_WIDTH`：视觉特征无粗 JSON 匹配时，各轴细切半宽，默认 `1.0 mm`。
- `DYNAMIC_MAX_FINE_SLICES`：单零件细切总层数安全上限，默认 `30000`。
- `REFINE_MAX_FEATURE_JSON_CHARS`：第二 Agent 特征 JSON 的传输上限，默认 `900000`；超限时只压缩 O(n²) 关系列表，完整归档不变。
- `REFINE_RELATIONSHIP_EXAMPLES_PER_TYPE`：超限后每种拓扑关系传给第二 Agent 的代表样本数，默认 `8`。
- `PIPELINE_TIMEOUT`：单个几何脚本超时秒数，默认 `300`。

如果动态细切和第一 Agent 已完成，但第二 Agent 因暂时故障或旧版输入过长失败，可复用原产物只重试第二阶段：

```bash
python tools/retry_dynamic_second_stage.py \
  --results-dir results/dynamic \
  hard_15 hard_6
```

不要把 `~/.codex/auth.json`、访问令牌或 API Key 复制到项目目录。
