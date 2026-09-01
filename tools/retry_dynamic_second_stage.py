"""Retry only the second agent for completed dynamic-slicing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_client import call_codex_vision, ensure_codex_oauth
from pipeline import save_result
from run_visual_json_serial import (
    REFINE_SCHEMA,
    REFINE_SYSTEM_PROMPT,
    build_dynamic_refine_user_prompt,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_metric_row(csv_path: Path, base_name: str, updates: dict) -> None:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"指标 CSV 缺少表头: {csv_path}")

    matching = [index for index, row in enumerate(rows) if row.get("base_name") == base_name]
    if not matching:
        raise RuntimeError(f"指标 CSV 中找不到 {base_name}")
    insert_at = matching[0]
    source = rows[matching[-1]].copy()
    source.update({key: str(value) for key, value in updates.items()})
    rows = [row for row in rows if row.get("base_name") != base_name]
    rows.insert(insert_at, source)

    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)


def retry_one(results_dir: Path, base_name: str) -> dict:
    first_agent = _load_json(results_dir / f"{base_name}_first_agent.json")
    plan = _load_json(results_dir / f"{base_name}_fine_slice_plan.json")
    fine_features = _load_json(results_dir / f"{base_name}_fine_features.json")
    fine_image = results_dir / f"{base_name}_fine_combined.png"
    metrics_path = results_dir / "metrics_visual_json_serial.csv"

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = [
            row for row in csv.DictReader(handle) if row.get("base_name") == base_name
        ]
    if not metric_rows:
        raise RuntimeError(f"指标 CSV 中找不到 {base_name}")
    previous = metric_rows[-1]

    user_prompt, prompt_metadata = build_dynamic_refine_user_prompt(
        first_agent,
        plan,
        fine_features,
    )
    metadata_path = results_dir / f"{base_name}_second_agent_payload_meta.json"
    metadata_path.write_text(
        json.dumps(prompt_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    started = time.time()
    result, usage = call_codex_vision(
        system_prompt=REFINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_path=fine_image,
        json_schema=REFINE_SCHEMA,
    )
    elapsed = round(time.time() - started, 1)
    save_result(
        base_name,
        "luna_visual_json_serial",
        result,
        results_dir=results_dir,
    )

    prompt_tokens = int(float(previous.get("prompt_tokens") or 0)) + usage["prompt_tokens"]
    completion_tokens = int(float(previous.get("completion_tokens") or 0)) + usage["completion_tokens"]
    model_time = round(float(previous.get("codex_time_s") or 0) + elapsed, 1)
    _replace_metric_row(metrics_path, base_name, {
        "codex_time_s": model_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "status": "OK",
        "error": "",
    })
    return {
        "base_name": base_name,
        "status": "OK",
        "retry_model_time_s": elapsed,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_compaction": prompt_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复用现有动态细切产物，仅重试第二 Agent"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("base_names", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_codex_oauth()
    for base_name in args.base_names:
        print(json.dumps(retry_one(args.results_dir.resolve(), base_name), ensure_ascii=False))


if __name__ == "__main__":
    main()
