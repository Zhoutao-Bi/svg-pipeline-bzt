#!/usr/bin/env python3
"""Evaluate semantic feature enrichment over existing pipeline JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_recognition import enrich_feature_data, summarize_feature_data


def discover_json_files(inputs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.update(path.glob("*_features.json"))
        elif path.is_file():
            files.add(path)
    return sorted(files)


def evaluate(paths: list[Path], write_dir: Path | None = None) -> dict:
    semantic = Counter()
    relationships = Counter()
    patterns = Counter()
    per_file = []
    total_transitions = 0
    total_observations = 0
    total_projection_evidence = 0
    if write_dir:
        write_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        enriched = enrich_feature_data(source)
        summary = summarize_feature_data(enriched)
        semantic.update(summary["semantic_counts"])
        relationships.update(summary["relationship_counts"])
        patterns.update(summary["pattern_counts"])
        total_transitions += summary["profile_transitions"]
        total_observations += summary["slice_observations"]
        total_projection_evidence += summary["projection_evidence"]
        per_file.append({"file": path.name, **summary})
        if write_dir:
            destination = write_dir / path.name
            destination.write_text(
                json.dumps(enriched, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

    return {
        "files": len(paths),
        "recognized_features": sum(semantic.values()),
        "slice_observations": total_observations,
        "projection_evidence": total_projection_evidence,
        "semantic_counts": dict(sorted(semantic.items())),
        "relationships": sum(relationships.values()),
        "relationship_counts": dict(sorted(relationships.items())),
        "patterns": sum(patterns.values()),
        "pattern_counts": dict(sorted(patterns.items())),
        "profile_transitions": total_transitions,
        "per_file": per_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Feature JSON file(s) or directories")
    parser.add_argument("--write-dir", type=Path, help="Optionally write enriched JSON copies")
    parser.add_argument("--summary-only", action="store_true", help="Omit per-file detail")
    args = parser.parse_args()

    paths = discover_json_files(args.inputs)
    if not paths:
        parser.error("no *_features.json files found")
    report = evaluate(paths, args.write_dir)
    if args.summary_only:
        report.pop("per_file", None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
