# STL 特征提取 Python 消融流水线

本项目是一套本地 Python 消融流水线。模型调用通过已登录的 Codex CLI 完成，复用 ChatGPT/OAuth 会话，不需要在项目中保存 OpenAI API Key。

默认模型配置：

- 模型：`gpt-5.6-terra`
- 推理强度：`medium`
- 认证：Codex CLI 的 ChatGPT/OAuth 登录

## 三种实验模式

| `run_experiment.py --mode` | 处理方式 | 独立入口 |
| --- | --- | --- |
| `visual-only` | 只把 X/Y/Z 三视图交给模型 | `run_visual_only.py` |
| `visual-json-parallel` | 一次调用同时提供三视图和几何 JSON | `run_visual_json_parallel.py` |
| `visual-json-serial` | 第一轮读取三视图，第二轮使用几何 JSON 矫正 | `run_visual_json_serial.py` |
| `consensus` | 对前三种结果做无额外模型调用的跨流程共识融合 | `consensus_fusion.py` |
| `all` | 依次执行三种模型流程和共识融合 | — |

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
python run_experiment.py --mode consensus
python run_experiment.py --mode all
```

输出保存在 `results/`，包含：

- `{零件名}_combined.png`：三视图拼合图。
- `{零件名}_features.json`：本地几何特征。
- `{零件名}_refined_{模型标签}_*.txt`：模型结构化结果，例如 Terra 使用 `terra` 标签。
- `{零件名}_refined_{模型标签}_consensus_balanced.txt`：跨流程共识结果。
- `metrics_*.csv`：耗时和 token 统计。

## 共识融合

`consensus` 会把纯视觉、视觉+JSON、串行矫正三份输出按特征类型、归一化坐标和尺寸聚类，不会再次调用模型。默认 `balanced` 配置保留至少两个流程支持、且至少一个流程判断为装配用途的特征；数值字段取跨流程中位数，整体尺寸使用本地几何包络。

已有三类输出时，可以从只读来源目录生成到新的结果目录：

```bash
CONSENSUS_SOURCE_DIR=/path/to/baseline \
RESULTS_DIR=/path/to/new-results \
python run_experiment.py --mode consensus
```

高精确率配置要求纯视觉和视觉+JSON同时支持候选：

```bash
CONSENSUS_PROFILE=precision python run_experiment.py --mode consensus
```

## 几何处理链

每个 STL 会先执行本地几何流水线：

1. `stl_to_svg.py`：沿 X/Y/Z 三轴切片。
2. `optimize_svg.py`：简化和拟合 SVG 几何。
3. `merge_svg.py`：合并各轴切片。
4. `extract_features.py`：提取几何特征并生成深度视图。
5. `minify_features.py`：压缩特征 JSON。

`pipeline.py` 负责几何调度和结果读写，`codex_client.py` 负责 OAuth 模型调用。`refine_features.py` 只在 `SLICE_MODE=dynamic` 时执行。

## 环境变量

- `CODEX_BIN`：Codex CLI 命令，默认 `codex`。
- `CODEX_MODEL`：默认 `gpt-5.6-terra`。
- `CODEX_REASONING_EFFORT`：默认 `medium`。
- `CODEX_TIMEOUT`：单次 Codex 调用超时秒数，默认 `600`。
- `CODEX_CONCURRENCY`：并发模型调用数，默认 `1`。
- `INPUT_STL_DIR`：STL 输入目录，默认 `input_stl/`。
- `INPUT_STL_PATTERN`：输入文件匹配模式，默认匹配全部 STL；例如 `easy_1.stl` 只跑一个样本。
- `RESULTS_DIR`：输出目录，默认 `results/`。
- `SLICE_MODE`：`coarse`、`fine` 或 `dynamic`。
- `PIPELINE_TIMEOUT`：单个几何脚本超时秒数，默认 `300`。
- `CONSENSUS_SOURCE_DIR`：三类流程和几何 JSON 的来源目录；默认与 `RESULTS_DIR` 相同。
- `CONSENSUS_PROFILE`：`balanced`（默认）或 `precision`。
- `CONSENSUS_COORDINATE_TOLERANCE`：坐标聚类阈值，占零件包络对角线比例，默认 `0.05`。
- `CONSENSUS_SIZE_TOLERANCE`：尺寸相对差聚类阈值，默认 `0.5`。

不要把 `~/.codex/auth.json`、访问令牌或 API Key 复制到项目目录。
