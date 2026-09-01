# STL 特征提取 Python 消融流水线

本项目已将原 Dify 消融工作流转换为本地 Python 流水线。模型调用通过已登录的 Codex CLI 完成，复用 ChatGPT/OAuth 会话，不需要在项目中保存 OpenAI API Key。

默认模型配置：

- 模型：`gpt-5.6-luna`
- 推理强度：`medium`
- 认证：Codex CLI 的 ChatGPT/OAuth 登录

## 三种实验模式

| `run_experiment.py --mode` | 处理方式 | 独立入口 |
| --- | --- | --- |
| `visual-only` | 只把 X/Y/Z 三视图交给 Luna | `run_visual_only.py` |
| `visual-json-parallel` | 一次调用同时提供三视图和几何 JSON | `run_visual_json_parallel.py` |
| `visual-json-serial` | 第一轮读取三视图，第二轮使用几何 JSON 矫正 | `run_visual_json_serial.py` |
| `all` | 依次执行以上三种实验 | — |

三个 `dify_*.yml` 仅作为原始工作流参考，运行 Python 流水线不需要 Dify、ngrok 或回调 API。

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

`pipeline.py` 负责几何调度和结果读写，`codex_client.py` 负责 OAuth 模型调用。`refine_features.py` 只在 `SLICE_MODE=dynamic` 时执行。

## 环境变量

- `CODEX_BIN`：Codex CLI 命令，默认 `codex`。
- `CODEX_MODEL`：默认 `gpt-5.6-luna`。
- `CODEX_REASONING_EFFORT`：默认 `medium`。
- `CODEX_TIMEOUT`：单次 Codex 调用超时秒数，默认 `600`。
- `CODEX_CONCURRENCY`：并发模型调用数，默认 `1`。
- `INPUT_STL_DIR`：STL 输入目录，默认 `input_stl/`。
- `RESULTS_DIR`：输出目录，默认 `results/`。
- `SLICE_MODE`：`coarse`、`fine` 或 `dynamic`。
- `PIPELINE_TIMEOUT`：单个几何脚本超时秒数，默认 `300`。

不要把 `~/.codex/auth.json`、访问令牌或 API Key 复制到项目目录。
