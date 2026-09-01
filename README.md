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

## 动态切片串行流程

动态模式只改变 `visual-json-serial` 的两轮 Agent 流程：

1. 先用 `0.1 mm`（可配置）的粗切渲染图让第一 Agent 判断装配特征。
2. 将第一 Agent 的装配特征坐标与粗切 JSON 的孔/柱匹配，生成可审计的“整平面深度范围”JSON。
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
- `PIPELINE_TIMEOUT`：单个几何脚本超时秒数，默认 `300`。

不要把 `~/.codex/auth.json`、访问令牌或 API Key 复制到项目目录。
