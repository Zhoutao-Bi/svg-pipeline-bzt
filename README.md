# STL 特征提取消融实验

本仓库以三个可直接导入 Dify 的工作流为主，用同一批 STL 和同一套结构化输出，对比 JSON 几何信息参与方式对识别结果的影响。

## 三个实验入口

| Dify DSL | 实验方式 | 本地批处理脚本 |
| --- | --- | --- |
| `消融实验无json.yml` | 仅使用 X/Y/Z 三视图，由视觉模型直接提取特征 | `workflow_no_json.py` |
| `消融实验有并行json.yml` | 在一次模型调用中同时提供三视图和几何 JSON | `workflow_parallel_json.py` |
| `消融实验有串行json.yml` | 第一轮读取三视图，第二轮使用几何 JSON 矫正结果 | `workflow_serial_json.py` |

三个工作流接收一组 `.stl` 文件，调用本地服务取得拼合视图和几何特征，最后将模型输出回传到本地结果目录。导入 Dify 后，请把 YAML 中的两处 `https://fraction-slot-relax.ngrok-free.dev` 替换成你自己的公网服务地址。

## 本地流水线

STL 处理顺序如下：

1. `stl2vsg11.py`：沿 X/Y/Z 三轴切片。
2. `svg2svg.py`：简化和拟合 SVG 几何。
3. `vsg_merge.py`：合并各轴切片。
4. `svg_json_v6.py`：提取几何特征并生成深度视图。
5. `json_token.py`：压缩特征 JSON。

公共调度、OpenAI 调用和结果读写位于 `pipeline_utils.py`。原始 STL 默认放在 `dtqp/`，运行结果写入 `dtqp_results/`；这些目录均不会提交到 Git。

## 快速开始

安装依赖并设置密钥：

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY="你的密钥"
```

运行三种实验中的一种：

```bash
python workflow_no_json.py
python workflow_parallel_json.py
python workflow_serial_json.py
```

可用环境变量：

- `OPENAI_API_KEY`：必需，不要写入源码或配置文件。
- `OPENAI_BASE_URL`：默认 `https://api.openai.com/v1`。
- `OPENAI_MODEL`：默认 `gpt-5-mini-2025-08-07`。
- `LLM_CONCURRENCY`：模型调用并发数，默认 `1`。
- `SLICE_MODE`：`coarse`、`fine` 或 `dynamic`。
- `PIPELINE_TIMEOUT`：单个流水线脚本超时秒数，默认 `300`。

## Dify 回调服务

三个 YAML 使用 `api_server3.py` 的两个接口：

- `POST /get_local_data`：按文件名返回拼合图 URL、特征文本和抓手信息。
- `POST /save_result_refined`：按实验名称保存模型输出，避免三组结果互相覆盖。

启动服务：

```bash
export PUBLIC_BASE_URL="https://你的公网域名"
uvicorn api_server3:app --host 0.0.0.0 --port 8000
```

`LOCAL_RESULTS_DIR` 默认是 `dtqp_results/`，`GLOBAL_GRASP_FILE` 默认是 `bsp_grasp.txt`；两者都可以用同名环境变量覆盖。

## Docker

```bash
export OPENAI_API_KEY="你的密钥"
WORKFLOW=workflow_no_json.py docker compose up --build
```

将 `WORKFLOW` 改为另外两个本地批处理脚本即可切换实验。仓库不包含任何运行结果或 API Key。
