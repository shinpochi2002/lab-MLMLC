#!/usr/bin/env python3
"""Shared loading, metric, ranking, and output helpers for API baselines."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.evaluation_split import (
    BASELINE_SOURCE,
    DEFAULT_SPLIT_DIR,
    SOURCE_ROOT,
    resolve_split_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = "16_4_3"
DEFAULT_EVAL_PATH = DEFAULT_SPLIT_DIR / "test" / BASELINE_SOURCE.relative_to(SOURCE_ROOT)
DEFAULT_API_CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "root_json" / "api_dataset_like_manifest.json"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "root_json" / "study_data_mashup_apis.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"


@dataclass(frozen=True)
class MashupRecord:
    idx: int
    mashup_name: str
    mashup_id: str
    text: str
    true_apis: list[str]


def resolve_default_eval_path(target_name: str = DEFAULT_TARGET) -> Path:
    """Match the active layer-3 script's default evaluation-data resolution."""
    base = PROJECT_ROOT / "results" / "hierarchy" / "output" / "by_pattern"
    t_dash = target_name.replace("_", "-")
    t_us = target_name.replace("-", "_")
    layer_tag = "layer3_" + t_us
    aliases = [target_name, t_dash, t_us]
    if t_us == "16_4_3":
        aliases.append("16cluster")
    if t_us == "4_4_8":
        aliases.append("4cluster")
    if t_us == "8_4_4":
        aliases.append("8cluster")

    global_candidates = sorted(
        glob.glob(str(base / "*" / f"test_data_mashup_apis_with_hierarchy_{layer_tag}_drop_unknown_samples.json"))
    )
    if global_candidates:
        return Path(global_candidates[0])
    global_known = sorted(
        glob.glob(str(base / "*" / f"test_data_mashup_apis_with_hierarchy_{layer_tag}_known_only.json"))
    )
    if global_known:
        return Path(global_known[0])

    seen: set[Path] = set()
    for alias in aliases:
        directory = base / alias
        if directory in seen or not directory.is_dir():
            continue
        seen.add(directory)
        candidates = [
            directory / f"test_data_mashup_apis_with_hierarchy_{layer_tag}_drop_unknown_samples.json",
            directory / f"test_data_mashup_apis_with_hierarchy_{layer_tag}_known_only.json",
            directory / f"test_data_mashup_apis_with_hierarchy_{alias}_drop_unknown_samples.json",
            directory / f"test_data_mashup_apis_with_hierarchy_{alias}_known_only.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        drops = sorted(glob.glob(str(directory / "test_data_mashup_apis_with_hierarchy_*_drop_unknown_samples.json")))
        if drops:
            return Path(drops[0])

    return PROJECT_ROOT / "test_data_mashup_apis.json"


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _parse_mashup_entry(name: str, entry: Any, idx: int) -> MashupRecord:
    if isinstance(entry, dict):
        text = str(entry.get("description", "") or "")
        true_apis = entry.get("apis", [])
        mashup_id = str(entry.get("id", name))
    elif isinstance(entry, list):
        text = str(entry[1]) if len(entry) > 1 else ""
        true_apis = entry[2] if len(entry) > 2 and isinstance(entry[2], list) else []
        mashup_id = str(entry[3]) if len(entry) > 3 else name
    else:
        text = ""
        true_apis = []
        mashup_id = name
    return MashupRecord(
        idx=idx,
        mashup_name=str(name),
        mashup_id=mashup_id,
        text=text,
        true_apis=unique_preserve_order(true_apis),
    )


def load_mashup_records(path: os.PathLike[str] | str, limit: int | None = None) -> list[MashupRecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[MashupRecord] = []
    if isinstance(data, dict):
        iterator = data.items()
    elif isinstance(data, list):
        iterator = ((str(i), value) for i, value in enumerate(data))
    else:
        raise ValueError(f"Unsupported mashup JSON root type: {type(data).__name__}")

    for idx, (name, entry) in enumerate(iterator):
        if limit is not None and idx >= limit:
            break
        records.append(_parse_mashup_entry(str(name), entry, idx))
    return records


def _parse_api_description(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("description", "") or "")
    if isinstance(entry, list) and len(entry) > 1:
        return str(entry[1] or "")
    return ""


def load_api_catalog(
    path: os.PathLike[str] | str,
    *,
    skip_empty_description: bool = True,
) -> tuple[list[str], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported API catalog JSON root type: {type(data).__name__}")

    api_names: list[str] = []
    api_texts: list[str] = []
    for name, entry in data.items():
        description = _parse_api_description(entry).strip()
        if skip_empty_description and not description:
            continue
        api_names.append(str(name))
        api_texts.append(description)
    return api_names, api_texts


def calculate_metrics(true_apis: Sequence[str], predicted_apis: Sequence[str]) -> dict[str, float | int]:
    """Same sample-level precision/recall/F1 definition as existing scripts."""
    true_set = set(str(x) for x in true_apis)
    predicted_set = set(str(x) for x in predicted_apis)
    true_positive = len(true_set.intersection(predicted_set))
    false_positive = len(predicted_set - true_set)
    false_negative = len(true_set - predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def average_metrics(result_records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not result_records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": sum(float(r["precision"]) for r in result_records) / len(result_records),
        "recall": sum(float(r["recall"]) for r in result_records) / len(result_records),
        "f1": sum(float(r["f1"]) for r in result_records) / len(result_records),
    }


def recommendations_from_scores(
    api_names: Sequence[str],
    scores: Any,
    top_k: int | None,
    *,
    candidate_indices: Sequence[int] | None = None,
    selection_mode: str = "topk",
    threshold: float = 0.5,
    threshold_strict: bool = False,
) -> list[dict[str, float | str]]:
    import numpy as np

    score_array = np.asarray(scores, dtype=float)
    if candidate_indices is None:
        indices = np.arange(len(api_names))
    else:
        indices = np.asarray(candidate_indices, dtype=int)
        if len(indices) != len(score_array):
            raise ValueError("candidate_indices and scores must have the same length")

    if len(score_array) == 0:
        return []
    if selection_mode == "threshold":
        if threshold_strict:
            local_selected = np.where(score_array > threshold)[0]
        else:
            local_selected = np.where(score_array >= threshold)[0]
        local_selected = sorted(local_selected, key=lambda i: (-float(score_array[i]), str(api_names[int(indices[i])])))
        return [
            {"api": str(api_names[int(indices[i])]), "score": float(score_array[i])}
            for i in local_selected
        ]
    if selection_mode != "topk":
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    if top_k is None or top_k <= 0:
        return []
    k = min(top_k, len(score_array))
    if k < len(score_array):
        local_top = np.argpartition(-score_array, k - 1)[:k]
    else:
        local_top = np.arange(len(score_array))
    local_top = sorted(local_top, key=lambda i: (-float(score_array[i]), str(api_names[int(indices[i])])))
    return [
        {"api": str(api_names[int(indices[i])]), "score": float(score_array[i])}
        for i in local_top
    ]


def make_result_record(record: MashupRecord, recommendations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    recommended_apis = [str(item["api"]) for item in recommendations]
    scores = [float(item["score"]) for item in recommendations]
    metrics = calculate_metrics(record.true_apis, recommended_apis)
    return {
        "idx": record.idx,
        "mashup_id": record.mashup_id,
        "mashup_name": record.mashup_name,
        "text": record.text,
        "true_apis": record.true_apis,
        "recommended_apis": recommended_apis,
        "scores": scores,
        "recommendations": list(recommendations),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "true_positive": metrics["true_positive"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def save_outputs(
    *,
    output_dir: os.PathLike[str] | str,
    method: str,
    top_k: int | None,
    result_records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    run_name: str | None = None,
    selection_mode: str = "topk",
    threshold: float | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if run_name:
        stem = safe_name(run_name)
    elif selection_mode == "threshold":
        stem = safe_name(f"{method}_threshold{threshold:g}")
    else:
        stem = safe_name(f"{method}_top{top_k}")
    json_path = output_path / f"{stem}.json"
    csv_path = output_path / f"{stem}.csv"
    averages = average_metrics(result_records)
    payload = {
        "metadata": {
            **metadata,
            "method": method,
            "selection_mode": selection_mode,
            "top_k": top_k,
            "threshold": threshold,
            "num_records": len(result_records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "average_metrics": averages,
        "records": list(result_records),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "idx",
        "mashup_id",
        "mashup_name",
        "text",
        "true_apis",
        "recommended_apis",
        "scores",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "false_negative",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        for record in result_records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["true_apis"] = json.dumps(row["true_apis"], ensure_ascii=False)
            row["recommended_apis"] = json.dumps(row["recommended_apis"], ensure_ascii=False)
            row["scores"] = json.dumps(row["scores"], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def print_run_summary(json_path: Path, csv_path: Path, result_records: Sequence[dict[str, Any]]) -> None:
    averages = average_metrics(result_records)
    print("[Average Metrics]")
    print(f"Precision: {averages['precision']:.6f}")
    print(f"Recall:    {averages['recall']:.6f}")
    print(f"F1:        {averages['f1']:.6f}")
    print(f"[OK] saved JSON: {json_path}")
    print(f"[OK] saved CSV:  {csv_path}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def add_common_io_args(parser: Any) -> None:
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Legacy target name used only when target resolution is requested.")
    parser.add_argument("--eval-path", default=None, help="Evaluation/test JSON path. Defaults to the fixed 236-record test split of the layer1_direct_api collaboration-filtered data.")
    parser.add_argument("--api-catalog-path", default=str(DEFAULT_API_CATALOG_PATH), help="API catalog JSON path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/CSV outputs.")
    parser.add_argument("--top-k", type=positive_int, default=10, help="Number of APIs to recommend per requirement when --selection-mode topk.")
    parser.add_argument(
        "--selection-mode",
        choices=["topk", "threshold"],
        default="topk",
        help="Select recommendations by fixed top-k or by score threshold.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold used when --selection-mode threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--run-name", default=None, help="Optional output filename stem.")


def resolved_eval_path(args: Any) -> Path:
    eval_path = getattr(args, "eval_path", None)
    if eval_path:
        return Path(eval_path)
    env_path = os.getenv("EVALUATION_DATA_PATH")
    if env_path:
        return Path(env_path)
    return resolve_split_path(BASELINE_SOURCE, "test")
