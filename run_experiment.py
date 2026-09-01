"""统一运行三种消融实验的纯 Python 入口。"""

import argparse
import importlib
import os


EXPERIMENTS = {
    "visual-only": "run_visual_only",
    "visual-json-parallel": "run_visual_json_parallel",
    "visual-json-serial": "run_visual_json_serial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Codex OAuth 和 GPT-5.6 Luna 运行 STL 消融实验"
    )
    parser.add_argument(
        "--mode",
        choices=[*EXPERIMENTS, "all"],
        default=os.getenv("EXPERIMENT_MODE", "visual-only"),
        help="实验模式；all 会依次运行三种模式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = list(EXPERIMENTS) if args.mode == "all" else [args.mode]

    for index, mode in enumerate(modes, 1):
        if len(modes) > 1:
            print(f"\n{'#' * 70}\n[{index}/{len(modes)}] 实验模式: {mode}\n{'#' * 70}")
        module = importlib.import_module(EXPERIMENTS[mode])
        module.main()


if __name__ == "__main__":
    main()
