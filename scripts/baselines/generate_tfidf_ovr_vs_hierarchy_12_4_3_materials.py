#!/usr/bin/env python3
"""Generate comparison materials for TF-IDF OvR vs hierarchy 12-4-3.

This script reads existing outputs only. It combines the hierarchy 12-4-3
`shorter_desc` and `longer_desc` result CSVs into the same 336-record test set
used by the TF-IDF OvR baseline, then writes evidence for a separate LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from run_tfidf_ovr_baseline import load_records_with_keys


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
PRIMARY_RUN_DIR = BASE_DIR / "results" / "tfidf_ovr_logistic_unigram"
HIERARCHY_RESULT_ROOT = PROJECT_ROOT / "codex" / "results" / "hierarchy_existing_models_condition_2x2"
OUTPUT_DIR = BASE_DIR / "analysis" / "tfidf_ovr_vs_hierarchy_12_4_3"
HIERARCHY_FULL_SUBSETS = ["shorter_desc", "longer_desc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--case-limit", type=int, default=12)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(row.get(key, "")) for key in fieldnames})


def stringify(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def parse_json_cell(value: str, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def metric_for(true_apis: Sequence[str], predicted_apis: Sequence[str]) -> dict[str, float | int]:
    true_set = set(true_apis)
    pred_set = set(predicted_apis)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def aggregate_metrics(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    total_tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
    total_fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
    total_fn = sum(int(row[f"{prefix}_fn"]) for row in rows)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "precision": statistics.fmean(float(row[f"{prefix}_precision"]) for row in rows),
        "recall": statistics.fmean(float(row[f"{prefix}_recall"]) for row in rows),
        "f1": statistics.fmean(float(row[f"{prefix}_f1"]) for row in rows),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_true_positive": total_tp,
        "total_false_positive": total_fp,
        "total_false_negative": total_fn,
        "average_number_of_predicted_apis": statistics.fmean(int(row[f"{prefix}_pred_count"]) for row in rows),
        "num_records": len(rows),
    }


def load_primary_predictions() -> list[dict[str, Any]]:
    predictions = read_json(PRIMARY_RUN_DIR / "predictions.json")
    return sorted(predictions, key=lambda item: int(item["idx"]))


def load_hierarchy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in HIERARCHY_FULL_SUBSETS:
        path = HIERARCHY_RESULT_ROOT / subset / "layer3_12_4_3_results.csv"
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                parsed = {
                    "source_subset": subset,
                    "source_idx": int(row["idx"]),
                    "text": row["text"],
                    "true_apis": parse_json_cell(row["true_apis"], []),
                    "pred_apis": parse_json_cell(row["pred_apis"], []),
                    "true_layer1": parse_json_cell(row["true_layer1"], []),
                    "pred_layer1": parse_json_cell(row["pred_layer1"], []),
                    "true_layer2": parse_json_cell(row["true_layer2"], []),
                    "pred_layer2": parse_json_cell(row["pred_layer2"], []),
                    "true_layer3": parse_json_cell(row["true_layer3"], []),
                    "pred_layer3": parse_json_cell(row["pred_layer3"], []),
                    "layer1_precision": float(row["layer1_precision"]),
                    "layer1_recall": float(row["layer1_recall"]),
                    "layer1_f1": float(row["layer1_f1"]),
                    "layer2_precision": float(row["layer2_precision"]),
                    "layer2_recall": float(row["layer2_recall"]),
                    "layer2_f1": float(row["layer2_f1"]),
                    "layer3_precision": float(row["layer3_precision"]),
                    "layer3_recall": float(row["layer3_recall"]),
                    "layer3_f1": float(row["layer3_f1"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "inference_process": parse_json_cell(row["inference_process"], {}),
                }
                rows.append(parsed)
    return rows


def load_condition_summary() -> list[dict[str, Any]]:
    path = HIERARCHY_RESULT_ROOT / "metric_summaries_by_test_data" / "all_test_data_summary.csv"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["configuration"] == "layer3_12_4_3":
                rows.append(row)
    return rows


def load_train_counts(config: dict[str, Any]) -> Counter[str]:
    records, _stats = load_records_with_keys(
        config["train_path"],
        text_key=config.get("text_key", "description"),
        label_key=config.get("label_key", "apis"),
        id_key=config.get("id_key"),
    )
    return Counter(api for record in records for api in record.true_apis)


def make_text_index(rows: Sequence[dict[str, Any]]) -> dict[str, deque[dict[str, Any]]]:
    index: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        index[normalize_text(row["text"])].append(row)
    return index


def match_hierarchy_to_primary(primary_rows: Sequence[dict[str, Any]], hierarchy_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    hierarchy_by_text = make_text_index(hierarchy_rows)
    matched: list[dict[str, Any]] = []
    for primary in primary_rows:
        key = normalize_text(primary["text"])
        candidates = hierarchy_by_text.get(key)
        if not candidates:
            raise SystemExit(f"No hierarchy row matched primary idx={primary['idx']} text={primary['text'][:80]!r}")
        selected_idx = None
        primary_true = set(primary["true_apis"])
        for idx, candidate in enumerate(candidates):
            if set(candidate["true_apis"]) == primary_true:
                selected_idx = idx
                break
        if selected_idx is None:
            selected_idx = 0
        hierarchy = candidates[selected_idx]
        del candidates[selected_idx]
        matched.append({"primary": primary, "hierarchy": hierarchy})
    return matched


def top_leaf_categories(inference_process: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for leaf in inference_process.get("leaf", []):
        cat = leaf.get("cat", "")
        scores = leaf.get("api_scores", [])
        items.append({"category": cat, "api_count": len(scores)})
    return items


def make_record_rows(matched: Sequence[dict[str, Any]], train_counts: Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in matched:
        primary = pair["primary"]
        hierarchy = pair["hierarchy"]
        true_apis = list(primary["true_apis"])
        primary_pred = list(primary["predicted_apis"])
        hierarchy_pred = list(hierarchy["pred_apis"])
        primary_metrics = metric_for(true_apis, primary_pred)
        hierarchy_metrics = metric_for(true_apis, hierarchy_pred)
        true_set = set(true_apis)
        primary_set = set(primary_pred)
        hierarchy_set = set(hierarchy_pred)
        unseen_true = [api for api in true_apis if train_counts[api] == 0]
        rows.append(
            {
                "idx": int(primary["idx"]),
                "mashup_id": primary.get("mashup_id", primary.get("sample_id", "")),
                "mashup_name": primary.get("mashup_name", ""),
                "text": primary["text"],
                "true_apis": true_apis,
                "true_count": len(true_apis),
                "unseen_true_apis": unseen_true,
                "unseen_true_count": len(unseen_true),
                "tfidf_ovr_predicted_apis": primary_pred,
                "tfidf_ovr_pred_count": len(primary_pred),
                "tfidf_ovr_precision": primary_metrics["precision"],
                "tfidf_ovr_recall": primary_metrics["recall"],
                "tfidf_ovr_f1": primary_metrics["f1"],
                "tfidf_ovr_tp": primary_metrics["true_positive"],
                "tfidf_ovr_fp": primary_metrics["false_positive"],
                "tfidf_ovr_fn": primary_metrics["false_negative"],
                "hierarchy_12_4_3_source_subset": hierarchy["source_subset"],
                "hierarchy_12_4_3_source_idx": hierarchy["source_idx"],
                "hierarchy_12_4_3_predicted_apis": hierarchy_pred,
                "hierarchy_12_4_3_pred_count": len(hierarchy_pred),
                "hierarchy_12_4_3_precision": hierarchy_metrics["precision"],
                "hierarchy_12_4_3_recall": hierarchy_metrics["recall"],
                "hierarchy_12_4_3_f1": hierarchy_metrics["f1"],
                "hierarchy_12_4_3_tp": hierarchy_metrics["true_positive"],
                "hierarchy_12_4_3_fp": hierarchy_metrics["false_positive"],
                "hierarchy_12_4_3_fn": hierarchy_metrics["false_negative"],
                "f1_delta_tfidf_minus_hierarchy": primary_metrics["f1"] - hierarchy_metrics["f1"],
                "precision_delta_tfidf_minus_hierarchy": primary_metrics["precision"] - hierarchy_metrics["precision"],
                "recall_delta_tfidf_minus_hierarchy": primary_metrics["recall"] - hierarchy_metrics["recall"],
                "shared_true_positive_apis": sorted(true_set & primary_set & hierarchy_set),
                "tfidf_unique_true_positive_apis": sorted((true_set & primary_set) - hierarchy_set),
                "hierarchy_unique_true_positive_apis": sorted((true_set & hierarchy_set) - primary_set),
                "tfidf_false_positive_apis": sorted(primary_set - true_set),
                "hierarchy_false_positive_apis": sorted(hierarchy_set - true_set),
                "tfidf_false_negative_apis": sorted(true_set - primary_set),
                "hierarchy_false_negative_apis": sorted(true_set - hierarchy_set),
                "hierarchy_true_layer1": hierarchy["true_layer1"],
                "hierarchy_pred_layer1": hierarchy["pred_layer1"],
                "hierarchy_layer1_precision": hierarchy["layer1_precision"],
                "hierarchy_layer1_recall": hierarchy["layer1_recall"],
                "hierarchy_layer1_f1": hierarchy["layer1_f1"],
                "hierarchy_true_layer2": hierarchy["true_layer2"],
                "hierarchy_pred_layer2": hierarchy["pred_layer2"],
                "hierarchy_layer2_precision": hierarchy["layer2_precision"],
                "hierarchy_layer2_recall": hierarchy["layer2_recall"],
                "hierarchy_layer2_f1": hierarchy["layer2_f1"],
                "hierarchy_true_layer3": hierarchy["true_layer3"],
                "hierarchy_pred_layer3": hierarchy["pred_layer3"],
                "hierarchy_layer3_precision": hierarchy["layer3_precision"],
                "hierarchy_layer3_recall": hierarchy["layer3_recall"],
                "hierarchy_layer3_f1": hierarchy["layer3_f1"],
                "hierarchy_leaf_categories": top_leaf_categories(hierarchy["inference_process"]),
            }
        )
    return rows


def make_label_rows(rows: Sequence[dict[str, Any]], train_counts: Counter[str]) -> list[dict[str, Any]]:
    all_labels = sorted(
        set(train_counts)
        | {api for row in rows for api in row["true_apis"]}
        | {api for row in rows for api in row["tfidf_ovr_predicted_apis"]}
        | {api for row in rows for api in row["hierarchy_12_4_3_predicted_apis"]}
    )
    label_rows: list[dict[str, Any]] = []
    for api in all_labels:
        true_count = sum(api in row["true_apis"] for row in rows)
        tfidf_pred_count = sum(api in row["tfidf_ovr_predicted_apis"] for row in rows)
        hierarchy_pred_count = sum(api in row["hierarchy_12_4_3_predicted_apis"] for row in rows)
        tfidf_tp = sum(api in row["true_apis"] and api in row["tfidf_ovr_predicted_apis"] for row in rows)
        hierarchy_tp = sum(api in row["true_apis"] and api in row["hierarchy_12_4_3_predicted_apis"] for row in rows)
        tfidf_fp = sum(api not in row["true_apis"] and api in row["tfidf_ovr_predicted_apis"] for row in rows)
        hierarchy_fp = sum(api not in row["true_apis"] and api in row["hierarchy_12_4_3_predicted_apis"] for row in rows)
        tfidf_fn = sum(api in row["true_apis"] and api not in row["tfidf_ovr_predicted_apis"] for row in rows)
        hierarchy_fn = sum(api in row["true_apis"] and api not in row["hierarchy_12_4_3_predicted_apis"] for row in rows)
        label_rows.append(
            {
                "api": api,
                "train_count": train_counts[api],
                "test_count": true_count,
                "tfidf_predicted_count": tfidf_pred_count,
                "hierarchy_predicted_count": hierarchy_pred_count,
                "tfidf_tp": tfidf_tp,
                "tfidf_fp": tfidf_fp,
                "tfidf_fn": tfidf_fn,
                "hierarchy_tp": hierarchy_tp,
                "hierarchy_fp": hierarchy_fp,
                "hierarchy_fn": hierarchy_fn,
                "tp_delta_hierarchy_minus_tfidf": hierarchy_tp - tfidf_tp,
                "fp_delta_hierarchy_minus_tfidf": hierarchy_fp - tfidf_fp,
                "unseen_in_tfidf_train": train_counts[api] == 0,
            }
        )
    return label_rows


def compact_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "idx": row["idx"],
        "mashup_name": row["mashup_name"],
        "text": row["text"],
        "true_apis": row["true_apis"],
        "unseen_true_apis": row["unseen_true_apis"],
        "tfidf_ovr_predicted_apis": row["tfidf_ovr_predicted_apis"],
        "tfidf_ovr_precision": row["tfidf_ovr_precision"],
        "tfidf_ovr_recall": row["tfidf_ovr_recall"],
        "tfidf_ovr_f1": row["tfidf_ovr_f1"],
        "hierarchy_12_4_3_predicted_apis": row["hierarchy_12_4_3_predicted_apis"],
        "hierarchy_12_4_3_precision": row["hierarchy_12_4_3_precision"],
        "hierarchy_12_4_3_recall": row["hierarchy_12_4_3_recall"],
        "hierarchy_12_4_3_f1": row["hierarchy_12_4_3_f1"],
        "f1_delta_tfidf_minus_hierarchy": row["f1_delta_tfidf_minus_hierarchy"],
        "shared_true_positive_apis": row["shared_true_positive_apis"],
        "tfidf_unique_true_positive_apis": row["tfidf_unique_true_positive_apis"],
        "hierarchy_unique_true_positive_apis": row["hierarchy_unique_true_positive_apis"],
        "tfidf_false_positive_apis": row["tfidf_false_positive_apis"],
        "hierarchy_false_positive_apis": row["hierarchy_false_positive_apis"],
        "hierarchy_true_layer1": row["hierarchy_true_layer1"],
        "hierarchy_pred_layer1": row["hierarchy_pred_layer1"],
        "hierarchy_layer1_f1": row["hierarchy_layer1_f1"],
        "hierarchy_true_layer2": row["hierarchy_true_layer2"],
        "hierarchy_pred_layer2": row["hierarchy_pred_layer2"],
        "hierarchy_layer2_f1": row["hierarchy_layer2_f1"],
        "hierarchy_true_layer3": row["hierarchy_true_layer3"],
        "hierarchy_pred_layer3": row["hierarchy_pred_layer3"],
        "hierarchy_layer3_f1": row["hierarchy_layer3_f1"],
        "hierarchy_leaf_categories": row["hierarchy_leaf_categories"],
    }


def make_case_examples(rows: Sequence[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    tfidf_better = sorted(
        [row for row in rows if row["f1_delta_tfidf_minus_hierarchy"] > 0],
        key=lambda row: (-row["f1_delta_tfidf_minus_hierarchy"], row["idx"]),
    )
    hierarchy_better = sorted(
        [row for row in rows if row["f1_delta_tfidf_minus_hierarchy"] < 0],
        key=lambda row: (row["f1_delta_tfidf_minus_hierarchy"], row["idx"]),
    )
    hierarchy_recall_gain_precision_loss = sorted(
        [
            row
            for row in rows
            if row["hierarchy_12_4_3_recall"] > row["tfidf_ovr_recall"]
            and row["hierarchy_12_4_3_precision"] < row["tfidf_ovr_precision"]
        ],
        key=lambda row: (-(row["hierarchy_12_4_3_recall"] - row["tfidf_ovr_recall"]), row["idx"]),
    )
    hierarchy_layer_error = sorted(
        [
            row
            for row in rows
            if row["hierarchy_layer1_f1"] < 1.0 or row["hierarchy_layer2_f1"] < 1.0 or row["hierarchy_layer3_f1"] < 1.0
        ],
        key=lambda row: (row["hierarchy_layer3_f1"], row["hierarchy_layer2_f1"], row["hierarchy_layer1_f1"], row["idx"]),
    )
    layer_correct_leaf_noisy = sorted(
        [
            row
            for row in rows
            if row["hierarchy_layer3_recall"] == 1.0 and row["hierarchy_12_4_3_precision"] < 0.3
        ],
        key=lambda row: (row["hierarchy_12_4_3_precision"], -row["hierarchy_12_4_3_recall"], row["idx"]),
    )
    tfidf_zero_hierarchy_hits = sorted(
        [
            row
            for row in rows
            if row["tfidf_ovr_pred_count"] == 0 and row["hierarchy_12_4_3_tp"] > 0
        ],
        key=lambda row: (-row["hierarchy_12_4_3_f1"], row["idx"]),
    )
    both_good = sorted(
        [
            row
            for row in rows
            if row["tfidf_ovr_f1"] >= 0.5 and row["hierarchy_12_4_3_f1"] >= 0.5
        ],
        key=lambda row: (-(row["tfidf_ovr_f1"] + row["hierarchy_12_4_3_f1"]), row["idx"]),
    )
    both_bad = sorted(
        [
            row
            for row in rows
            if row["tfidf_ovr_f1"] == 0.0 and row["hierarchy_12_4_3_f1"] == 0.0
        ],
        key=lambda row: (-row["true_count"], -row["unseen_true_count"], row["idx"]),
    )
    unseen_helped_by_hierarchy = sorted(
        [
            row
            for row in rows
            if row["unseen_true_count"] > 0 and set(row["unseen_true_apis"]) & set(row["hierarchy_12_4_3_predicted_apis"])
        ],
        key=lambda row: (-len(set(row["unseen_true_apis"]) & set(row["hierarchy_12_4_3_predicted_apis"])), row["idx"]),
    )
    return {
        "tfidf_ovr_better_than_hierarchy_examples": [compact_case(row) for row in tfidf_better[:limit]],
        "hierarchy_better_than_tfidf_ovr_examples": [compact_case(row) for row in hierarchy_better[:limit]],
        "hierarchy_recall_gain_precision_loss_examples": [
            compact_case(row) for row in hierarchy_recall_gain_precision_loss[:limit]
        ],
        "hierarchy_layer_error_examples": [compact_case(row) for row in hierarchy_layer_error[:limit]],
        "hierarchy_layer_correct_but_leaf_noisy_examples": [
            compact_case(row) for row in layer_correct_leaf_noisy[:limit]
        ],
        "tfidf_zero_prediction_hierarchy_hits_examples": [
            compact_case(row) for row in tfidf_zero_hierarchy_hits[:limit]
        ],
        "both_good_examples": [compact_case(row) for row in both_good[:limit]],
        "both_bad_examples": [compact_case(row) for row in both_bad[:limit]],
        "unseen_tfidf_label_helped_by_hierarchy_examples": [
            compact_case(row) for row in unseen_helped_by_hierarchy[:limit]
        ],
    }


def make_pairwise_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tfidf_tp = sum(int(row["tfidf_ovr_tp"]) for row in rows)
    hierarchy_tp = sum(int(row["hierarchy_12_4_3_tp"]) for row in rows)
    shared_tp = sum(len(row["shared_true_positive_apis"]) for row in rows)
    return {
        "records_tfidf_higher_f1": sum(row["f1_delta_tfidf_minus_hierarchy"] > 0 for row in rows),
        "records_hierarchy_higher_f1": sum(row["f1_delta_tfidf_minus_hierarchy"] < 0 for row in rows),
        "records_equal_f1": sum(row["f1_delta_tfidf_minus_hierarchy"] == 0 for row in rows),
        "mean_tfidf_minus_hierarchy_f1": statistics.fmean(row["f1_delta_tfidf_minus_hierarchy"] for row in rows),
        "tfidf_total_tp": tfidf_tp,
        "hierarchy_total_tp": hierarchy_tp,
        "shared_true_positive_api_mentions": shared_tp,
        "tfidf_unique_true_positive_mentions": tfidf_tp - shared_tp,
        "hierarchy_unique_true_positive_mentions": hierarchy_tp - shared_tp,
    }


def make_summary(
    rows: Sequence[dict[str, Any]],
    condition_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    pairwise = make_pairwise_summary(rows)
    condition_summary = {
        row["test_data"]: {
            "sample_count": int(row["sample_count"]),
            "avg_precision": float(row["avg_precision"]),
            "avg_recall": float(row["avg_recall"]),
            "avg_f1": float(row["avg_f1"]),
            "avg_pred_apis_count": float(row["avg_pred_apis_count"]),
            "avg_true_apis_count": float(row["avg_true_apis_count"]),
            "source_csv": row["source_csv"],
        }
        for row in condition_rows
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Evidence packet for comparing TF-IDF OvR Logistic unigram with hierarchy layer3 12-4-3.",
        "record_count": len(rows),
        "tfidf_ovr_metrics": aggregate_metrics(rows, "tfidf_ovr"),
        "hierarchy_12_4_3_metrics": aggregate_metrics(rows, "hierarchy_12_4_3"),
        "pairwise": pairwise,
        "records_with_unseen_true_label_for_tfidf": sum(row["unseen_true_count"] > 0 for row in rows),
        "unseen_true_label_mentions_for_tfidf": sum(row["unseen_true_count"] for row in rows),
        "condition_summary_for_hierarchy_12_4_3": condition_summary,
        "top_hierarchy_tp_gain_labels": sorted(
            [row for row in label_rows if row["test_count"] > 0],
            key=lambda row: (-int(row["tp_delta_hierarchy_minus_tfidf"]), -int(row["test_count"]), row["api"]),
        )[:30],
        "top_hierarchy_fp_increase_labels": sorted(
            [row for row in label_rows if row["hierarchy_predicted_count"] > 0],
            key=lambda row: (-int(row["fp_delta_hierarchy_minus_tfidf"]), -int(row["hierarchy_predicted_count"]), row["api"]),
        )[:30],
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value).replace("\n", " ")


def write_markdown(path: Path, summary: dict[str, Any], case_examples: dict[str, list[dict[str, Any]]]) -> None:
    metrics = [
        [
            "tfidf_ovr_logistic_unigram",
            summary["tfidf_ovr_metrics"]["average_number_of_predicted_apis"],
            summary["tfidf_ovr_metrics"]["precision"],
            summary["tfidf_ovr_metrics"]["recall"],
            summary["tfidf_ovr_metrics"]["f1"],
            summary["tfidf_ovr_metrics"]["micro_precision"],
            summary["tfidf_ovr_metrics"]["micro_recall"],
            summary["tfidf_ovr_metrics"]["micro_f1"],
        ],
        [
            "hierarchy_layer3_12_4_3_shorter+longer",
            summary["hierarchy_12_4_3_metrics"]["average_number_of_predicted_apis"],
            summary["hierarchy_12_4_3_metrics"]["precision"],
            summary["hierarchy_12_4_3_metrics"]["recall"],
            summary["hierarchy_12_4_3_metrics"]["f1"],
            summary["hierarchy_12_4_3_metrics"]["micro_precision"],
            summary["hierarchy_12_4_3_metrics"]["micro_recall"],
            summary["hierarchy_12_4_3_metrics"]["micro_f1"],
        ],
    ]
    condition_rows = [
        [
            name,
            item["sample_count"],
            item["avg_pred_apis_count"],
            item["avg_precision"],
            item["avg_recall"],
            item["avg_f1"],
        ]
        for name, item in sorted(summary["condition_summary_for_hierarchy_12_4_3"].items())
    ]
    lines = [
        "# TF-IDF OvR vs Hierarchy 12-4-3 Analysis Materials",
        "",
        "This packet is for a separate LLM. It collects evidence and comparison tables without making the final interpretation.",
        "",
        "## Files",
        "",
        "- `summary.json`: compact machine-readable overview.",
        "- `records_comparison.csv`: one row per matched test record.",
        "- `label_comparison.csv`: per-API TP/FP/FN and prediction frequency for both methods.",
        "- `hierarchy_condition_summary.csv`: 12-4-3 metrics on shorter/longer/candidate-size subsets.",
        "- `case_examples.json`: curated examples for success, failure, recall-vs-precision, and hierarchy layer errors.",
        "",
        "## Overall Metrics On 336 Matched Records",
        "",
        markdown_table(
            ["method", "avg_pred", "precision", "recall", "F1", "micro_P", "micro_R", "micro_F1"],
            metrics,
        ),
        "",
        "## Pairwise Record Comparison",
        "",
        f"- TF-IDF higher F1 records: {summary['pairwise']['records_tfidf_higher_f1']}",
        f"- Hierarchy higher F1 records: {summary['pairwise']['records_hierarchy_higher_f1']}",
        f"- Equal F1 records: {summary['pairwise']['records_equal_f1']}",
        f"- Mean TF-IDF minus hierarchy F1: {summary['pairwise']['mean_tfidf_minus_hierarchy_f1']:.4f}",
        f"- TF-IDF unique true-positive mentions: {summary['pairwise']['tfidf_unique_true_positive_mentions']}",
        f"- Hierarchy unique true-positive mentions: {summary['pairwise']['hierarchy_unique_true_positive_mentions']}",
        "",
        "## Hierarchy 12-4-3 Condition Metrics",
        "",
        markdown_table(["condition", "n", "avg_pred", "precision", "recall", "F1"], condition_rows),
        "",
        "## Suggested Questions For The Analysis LLM",
        "",
        "1. Does hierarchy 12-4-3 mainly improve recall by expanding through predicted categories, and how much precision does that cost?",
        "2. Which examples show the hierarchy selecting the right upper categories but over-predicting leaf APIs?",
        "3. Which examples show TF-IDF OvR being more selective and therefore better despite missing some true APIs?",
        "4. Which API labels are rescued by hierarchy but impossible or hard for TF-IDF OvR because they are unseen or low-frequency in training?",
        "5. Are 12-4-3 errors dominated by layer/category mistakes or by noisy leaf API expansion after correct category prediction?",
        "",
        "## Case Example Index",
        "",
    ]
    for group_name, examples in case_examples.items():
        lines.append(f"### {group_name}")
        for example in examples[:3]:
            lines.append(
                f"- idx={example['idx']} name={example['mashup_name']} "
                f"tfidf_f1={example['tfidf_ovr_f1']:.3f} "
                f"hier_f1={example['hierarchy_12_4_3_f1']:.3f} "
                f"true={example['true_apis']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = load_primary_predictions()
    hierarchy_rows = load_hierarchy_rows()
    matched = match_hierarchy_to_primary(primary_rows, hierarchy_rows)
    primary_config = read_json(PRIMARY_RUN_DIR / "config.json")
    train_counts = load_train_counts(primary_config)
    rows = make_record_rows(matched, train_counts)
    label_rows = make_label_rows(rows, train_counts)
    condition_rows = load_condition_summary()
    summary = make_summary(rows, condition_rows, label_rows)
    case_examples = make_case_examples(rows, args.case_limit)

    record_fieldnames = [
        "idx",
        "mashup_id",
        "mashup_name",
        "text",
        "true_apis",
        "true_count",
        "unseen_true_apis",
        "unseen_true_count",
        "tfidf_ovr_predicted_apis",
        "tfidf_ovr_pred_count",
        "tfidf_ovr_precision",
        "tfidf_ovr_recall",
        "tfidf_ovr_f1",
        "tfidf_ovr_tp",
        "tfidf_ovr_fp",
        "tfidf_ovr_fn",
        "hierarchy_12_4_3_source_subset",
        "hierarchy_12_4_3_source_idx",
        "hierarchy_12_4_3_predicted_apis",
        "hierarchy_12_4_3_pred_count",
        "hierarchy_12_4_3_precision",
        "hierarchy_12_4_3_recall",
        "hierarchy_12_4_3_f1",
        "hierarchy_12_4_3_tp",
        "hierarchy_12_4_3_fp",
        "hierarchy_12_4_3_fn",
        "f1_delta_tfidf_minus_hierarchy",
        "precision_delta_tfidf_minus_hierarchy",
        "recall_delta_tfidf_minus_hierarchy",
        "shared_true_positive_apis",
        "tfidf_unique_true_positive_apis",
        "hierarchy_unique_true_positive_apis",
        "tfidf_false_positive_apis",
        "hierarchy_false_positive_apis",
        "tfidf_false_negative_apis",
        "hierarchy_false_negative_apis",
        "hierarchy_true_layer1",
        "hierarchy_pred_layer1",
        "hierarchy_layer1_precision",
        "hierarchy_layer1_recall",
        "hierarchy_layer1_f1",
        "hierarchy_true_layer2",
        "hierarchy_pred_layer2",
        "hierarchy_layer2_precision",
        "hierarchy_layer2_recall",
        "hierarchy_layer2_f1",
        "hierarchy_true_layer3",
        "hierarchy_pred_layer3",
        "hierarchy_layer3_precision",
        "hierarchy_layer3_recall",
        "hierarchy_layer3_f1",
        "hierarchy_leaf_categories",
    ]
    label_fieldnames = [
        "api",
        "train_count",
        "test_count",
        "tfidf_predicted_count",
        "hierarchy_predicted_count",
        "tfidf_tp",
        "tfidf_fp",
        "tfidf_fn",
        "hierarchy_tp",
        "hierarchy_fp",
        "hierarchy_fn",
        "tp_delta_hierarchy_minus_tfidf",
        "fp_delta_hierarchy_minus_tfidf",
        "unseen_in_tfidf_train",
    ]
    condition_fieldnames = [
        "test_data",
        "configuration",
        "layer",
        "target",
        "sample_count",
        "avg_precision",
        "avg_recall",
        "avg_f1",
        "avg_pred_apis_count",
        "avg_true_apis_count",
        "source_csv",
        "evaluation_data_path",
    ]

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "case_examples.json", case_examples)
    write_csv(output_dir / "records_comparison.csv", rows, record_fieldnames)
    write_csv(output_dir / "label_comparison.csv", label_rows, label_fieldnames)
    write_csv(output_dir / "hierarchy_condition_summary.csv", condition_rows, condition_fieldnames)
    write_markdown(output_dir / "README_FOR_ANALYSIS_LLM.md", summary, case_examples)
    write_json(
        output_dir / "manifest.json",
        {
            "generated_at": summary["generated_at"],
            "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
            "primary_run_dir": str(PRIMARY_RUN_DIR),
            "hierarchy_result_root": str(HIERARCHY_RESULT_ROOT),
            "hierarchy_full_subsets": HIERARCHY_FULL_SUBSETS,
        },
    )
    print(f"[OK] wrote comparison materials to {output_dir}")


if __name__ == "__main__":
    main()
