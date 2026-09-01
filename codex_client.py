"""通过 Codex CLI 的 ChatGPT OAuth 会话调用模型。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional


CODEX_BIN = os.getenv("CODEX_BIN", "codex")
CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5.6-luna")
CODEX_REASONING_EFFORT = os.getenv("CODEX_REASONING_EFFORT", "medium")
CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", "600"))


class CodexCallError(RuntimeError):
    """Codex CLI 调用失败。"""


@lru_cache(maxsize=1)
def ensure_codex_oauth() -> None:
    """确认 Codex CLI 可用，并且当前使用 ChatGPT/OAuth 登录。"""
    executable = shutil.which(CODEX_BIN)
    if not executable:
        raise CodexCallError(
            "找不到 Codex CLI。请先安装 Codex，然后运行 `codex login`。"
        )

    result = subprocess.run(
        [executable, "login", "status"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    status = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "chatgpt" not in status.lower():
        raise CodexCallError(
            "当前不是 Codex ChatGPT/OAuth 登录。请运行 `codex login` 完成浏览器登录。"
        )


def _usage_from_jsonl(stdout: str) -> dict:
    usage = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = event.get("usage")
        if isinstance(candidate, dict):
            usage = candidate

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
    }


def _parse_structured_output(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CodexCallError(f"Codex 返回的内容不是有效 JSON: {exc}") from exc
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def call_codex(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    image_path: Optional[Path] = None,
) -> tuple[str, dict]:
    """用 Codex OAuth、Luna 和 medium reasoning 获取结构化结果。"""
    ensure_codex_oauth()

    if image_path is not None:
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise CodexCallError(f"找不到输入图像: {image_path}")

    prompt = f"""<system_instructions>
{system_prompt}
</system_instructions>

<user_request>
{user_prompt}
</user_request>

只分析用户提供的文本和附加图像。不要检查工作目录，不要调用工具，不要修改文件。
最终响应必须严格符合给定 JSON Schema，不要添加 Markdown 或解释。
"""

    with tempfile.TemporaryDirectory(prefix="svg-codex-call-") as temp_dir:
        work_dir = Path(temp_dir)
        schema_path = work_dir / "output_schema.json"
        output_path = work_dir / "last_message.json"
        schema_path.write_text(
            json.dumps(json_schema, ensure_ascii=False), encoding="utf-8"
        )

        command = [
            CODEX_BIN,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            CODEX_MODEL,
            "--config",
            f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
            "--cd",
            str(work_dir),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--json",
            "--color",
            "never",
        ]
        if image_path is not None:
            command.extend(["--image", str(image_path)])
        command.append("-")

        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=CODEX_TIMEOUT,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexCallError(
                f"Codex CLI 调用超过 {CODEX_TIMEOUT} 秒，已终止。"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise CodexCallError(
                f"Codex CLI 调用失败（退出码 {result.returncode}）: {detail}"
            )
        if not output_path.is_file():
            raise CodexCallError("Codex CLI 没有生成最终响应文件。")

        content = _parse_structured_output(
            output_path.read_text(encoding="utf-8")
        )
        return content, _usage_from_jsonl(result.stdout)


def call_codex_vision(
    system_prompt: str,
    user_prompt: str,
    image_path: Path,
    json_schema: dict,
) -> tuple[str, dict]:
    return call_codex(system_prompt, user_prompt, json_schema, image_path=image_path)


def call_codex_text(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
) -> tuple[str, dict]:
    return call_codex(system_prompt, user_prompt, json_schema)
