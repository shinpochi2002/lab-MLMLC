#!/usr/bin/env python3
"""Select a hierarchy by mean per-sample API F1 on the fixed validation split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluation_split


def description_from_entry(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("description", ""))
    if isinstance(entry, list) and len(entry) > 1:
        return str(entry[1])
    raise ValueError("Unsupported validation record format")


def score_candidate(task: dict[str, Any], results_dir: Path, split_dir: Path, count: int) -> dict[str, Any]:
    name = task["name"]
    data_path = Path(task["evaluation_data_path"]).resolve()
    validation_root = (split_dir / "validation").resolve()
    try:
        relative_path = data_path.relative_to(validation_root)
    except ValueError as exc:
        raise ValueError(f"{name}: evaluation data must be inside {validation_root}") from exc
    source_path = evaluation_split.SOURCE_ROOT / relative_path
    verified_path = evaluation_split.resolve_split_path(source_path, "validation", split_dir)
    if verified_path != data_path:
        raise ValueError(f"{name}: unexpected validation data path: {data_path}")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if len(data) != count:
        raise ValueError(f"{name}: validation data count {len(data)} does not match manifest count {count}")

    csv_path = Path(task["out_csv_path"]).resolve()
    if not csv_path.is_relative_to(results_dir):
        raise ValueError(f"{name}: result CSV must be inside {results_dir}")
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, escapechar="\\")
        if not {"idx", "text", "f1"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{name}: CSV requires idx, text, and f1 columns: {csv_path}")
        rows = list(reader)
    if len(rows) != count:
        raise ValueError(f"{name}: CSV count {len(rows)} does not match validation count {count}")

    scores = []
    for idx, (row, entry) in enumerate(zip(rows, data.values())):
        try:
            actual_idx = int(row["idx"])
            f1 = float(row["f1"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: invalid idx or f1 at row {idx}") from exc
        if actual_idx != idx or row["text"] != description_from_entry(entry):
            raise ValueError(f"{name}: CSV row {idx} does not match validation record order/text")
        if not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError(f"{name}: f1 must be finite and within [0, 1] at row {idx}")
        scores.append(f1)
    return {
        "pattern": name,
        "layer": task["layer"],
        "sample_count": count,
        "sample_mean_f1": math.fsum(scores) / count,
        "evaluation_data_path": str(data_path),
        "csv_path": str(csv_path),
        "csv_sha256": evaluation_split.sha256_file(csv_path),
    }


def select_hierarchy(results_dir: Path, split_dir: Path = evaluation_split.DEFAULT_SPLIT_DIR) -> dict[str, Any]:
    results_dir = Path(results_dir).resolve()
    split_dir = Path(split_dir).resolve()
    manifest = evaluation_split.load_manifest(split_dir)
    count = manifest["splits"]["validation"]["count"]
    task_manifest_path = results_dir / "test_data_manifest.json"
    tasks = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Expected a non-empty task manifest: {task_manifest_path}")

    candidates = []
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or task.get("layer") not in {1, 2, 3, 4}:
            raise ValueError("Invalid hierarchy task in evaluation manifest")
        if task["layer"] == 1:
            continue
        name = task.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"Missing or duplicate hierarchy task name: {name}")
        seen.add(name)
        candidates.append(score_candidate(task, results_dir, split_dir, count))
    if not candidates:
        raise ValueError("No hierarchy candidates found; layer1_direct_api is excluded from selection")
    candidates.sort(key=lambda candidate: (-candidate["sample_mean_f1"], candidate["pattern"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "metric": "sample_mean_f1",
        "metric_description": "Arithmetic mean of per-sample final API f1; not label-macro F1.",
        "tie_break": "pattern name ascending",
        "split_manifest_path": str(split_dir / "manifest.json"),
        "split_manifest_sha256": evaluation_split.sha256_file(split_dir / "manifest.json"),
        "task_manifest_path": str(task_manifest_path),
        "task_manifest_sha256": evaluation_split.sha256_file(task_manifest_path),
        "seed": manifest["seed"],
        "validation_count": count,
        "candidate_count": len(candidates),
        "selected_pattern": candidates[0]["pattern"],
        "sample_mean_f1": candidates[0]["sample_mean_f1"],
        "rankings": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory produced by the hierarchy runner using --split validation.")
    parser.add_argument("--split-dir", type=Path, default=evaluation_split.DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, help="Selection JSON path; default: <results-dir>/selection.json.")
    args = parser.parse_args()
    try:
        selection = select_hierarchy(args.results_dir, args.split_dir)
        output = args.output or args.results_dir / "selection.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"Hierarchy selection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Selected: {selection['selected_pattern']} (validation mean F1={selection['sample_mean_f1']:.6f})")
    print(f"Selection: {output.resolve()}")
    print(f"Final evaluation: --split test --patterns {selection['selected_pattern']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
