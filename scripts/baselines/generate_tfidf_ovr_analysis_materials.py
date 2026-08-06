#!/usr/bin/env python3
"""Generate analysis materials for the strong TF-IDF OvR baseline.

The script does not retrain models or modify existing result files. It reads
saved outputs under codex/baseline/results and writes a compact analysis packet
for a separate LLM under codex/baseline/analysis/tfidf_ovr_materials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from baseline_utils import DEFAULT_API_CATALOG_PATH
from run_tfidf_ovr_baseline import load_records_with_keys


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_OUTPUT_DIR = BASE_DIR / "analysis" / "tfidf_ovr_materials"
PRIMARY_RUN = "tfidf_ovr_logistic_unigram"


RUN_SPECS = {
    "tfidf_ovr_logistic_unigram": {
        "kind": "ovr",
        "path": RESULTS_DIR / "tfidf_ovr_logistic_unigram",
        "label": "TF-IDF OvR Logistic unigram",
    },
    "tfidf_ovr_logistic_bigram": {
        "kind": "ovr",
        "path": RESULTS_DIR / "tfidf_ovr_logistic",
        "label": "TF-IDF OvR Logistic unigram+bigram",
    },
    "tfidf_ovr_svm_unigram": {
        "kind": "ovr",
        "path": RESULTS_DIR / "tfidf_ovr_svm_unigram",
        "label": "TF-IDF OvR SVM unigram",
    },
    "sbert_ovr_logistic": {
        "kind": "ovr",
        "path": RESULTS_DIR / "sbert_ovr_logistic",
        "label": "SBERT OvR Logistic",
    },
    "cross_encoder_top1": {
        "kind": "ranking",
        "path": RESULTS_DIR / "cross_encoder_all909_top1.json",
        "label": "Cross Encoder all909 top1",
    },
    "cross_encoder_top5": {
        "kind": "ranking",
        "path": RESULTS_DIR / "cross_encoder_all909_top5.json",
        "label": "Cross Encoder all909 top5",
    },
    "cross_encoder_threshold05": {
        "kind": "ranking",
        "path": RESULTS_DIR / "cross_encoder_all909_threshold05.json",
        "label": "Cross Encoder threshold 0.5",
    },
    "tfidf_cosine_top1": {
        "kind": "ranking",
        "path": RESULTS_DIR / "tfidf_top1_909api.json",
        "label": "TF-IDF cosine top1",
    },
    "tfidf_cosine_top5": {
        "kind": "ranking",
        "path": RESULTS_DIR / "tfidf_top5_909api.json",
        "label": "TF-IDF cosine top5",
    },
    "tfidf_cosine_top10": {
        "kind": "ranking",
        "path": RESULTS_DIR / "tfidf_top10_909api.json",
        "label": "TF-IDF cosine top10",
    },
    "sbert_mpnet_top1": {
        "kind": "ranking",
        "path": RESULTS_DIR / "sbert_top1_909api_mpnet.json",
        "label": "SBERT cosine mpnet top1",
    },
    "sbert_mpnet_top5": {
        "kind": "ranking",
        "path": RESULTS_DIR / "sbert_top5_909api_mpnet.json",
        "label": "SBERT cosine mpnet top5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-feature-terms", type=int, default=12)
    parser.add_argument("--case-limit", type=int, default=10)
    parser.add_argument("--include-full-text", action="store_true")
    return parser.parse_args()


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
            writer.writerow({key: stringify_csv(row.get(key, "")) for key in fieldnames})


def stringify_csv(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def load_labels(run_dir: Path) -> list[str]:
    payload = read_json(run_dir / "labels.json")
    idx_to_api = payload["idx_to_api"]
    return [idx_to_api[str(idx)] for idx in range(len(idx_to_api))]


def load_ovr_run(run_dir: Path) -> dict[str, Any]:
    predictions = read_json(run_dir / "predictions.json")
    return {
        "config": read_json(run_dir / "config.json"),
        "metrics": read_json(run_dir / "metrics.json"),
        "predictions": {int(item["idx"]): normalize_prediction(item, "predicted_apis") for item in predictions},
        "raw_predictions": predictions,
    }


def load_ranking_run(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "metadata": payload.get("metadata", {}),
        "metrics": payload.get("average_metrics", {}),
        "predictions": {int(item["idx"]): normalize_prediction(item, "recommended_apis") for item in payload["records"]},
        "raw_records": payload["records"],
    }


def enrich_metrics_from_predictions(run: dict[str, Any]) -> None:
    predictions = list(run["predictions"].values())
    if not predictions:
        return
    metrics = dict(run.get("metrics", {}))
    total_tp = sum(int(item["true_positive"]) for item in predictions)
    total_fp = sum(int(item["false_positive"]) for item in predictions)
    total_fn = sum(int(item["false_negative"]) for item in predictions)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    metrics.setdefault("micro_precision", micro_precision)
    metrics.setdefault("micro_recall", micro_recall)
    metrics.setdefault("micro_f1", micro_f1)
    metrics.setdefault("total_true_positive", total_tp)
    metrics.setdefault("total_false_positive", total_fp)
    metrics.setdefault("total_false_negative", total_fn)
    metrics.setdefault(
        "average_number_of_predicted_apis",
        statistics.fmean(len(item["predicted_apis"]) for item in predictions),
    )
    metrics.setdefault("num_records", len(predictions))
    run["metrics"] = metrics


def normalize_prediction(item: dict[str, Any], prediction_key: str) -> dict[str, Any]:
    predicted = list(item.get(prediction_key, []))
    true_apis = list(item.get("true_apis", []))
    return {
        "idx": int(item["idx"]),
        "mashup_id": str(item.get("mashup_id", item.get("sample_id", item["idx"]))),
        "mashup_name": str(item.get("mashup_name", "")),
        "text": str(item.get("text", "")),
        "true_apis": true_apis,
        "predicted_apis": predicted,
        "scores": item.get("scores", {}),
        "recommendations": item.get("recommendations", []),
        "precision": float(item.get("precision", 0.0)),
        "recall": float(item.get("recall", 0.0)),
        "f1": float(item.get("f1", 0.0)),
        "true_positive": int(item.get("true_positive", 0)),
        "false_positive": int(item.get("false_positive", 0)),
        "false_negative": int(item.get("false_negative", 0)),
    }


def metric_for(true_apis: Sequence[str], predicted_apis: Sequence[str]) -> dict[str, float | int]:
    true_set = set(true_apis)
    pred_set = set(predicted_apis)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def summarize_numeric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "mean": statistics.fmean(ordered),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def percentile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def load_train_counts(config: dict[str, Any]) -> Counter[str]:
    train_path = Path(config["train_path"])
    records, _stats = load_records_with_keys(
        train_path,
        text_key=config.get("text_key", "description"),
        label_key=config.get("label_key", "apis"),
        id_key=config.get("id_key"),
    )
    return Counter(api for record in records for api in record.true_apis)


def compute_full_primary_scores(run_dir: Path, texts: Sequence[str]) -> tuple[list[str], Any, Any]:
    import joblib

    labels = load_labels(run_dir)
    vectorizer = joblib.load(run_dir / "vectorizer.joblib")
    classifier = joblib.load(run_dir / "model.joblib")
    matrix = vectorizer.transform(texts)
    scores = classifier.predict_proba(matrix)
    return labels, vectorizer, scores


def top_score_items(labels: Sequence[str], sample_scores: Sequence[float], n: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(
        ((label, float(score)) for label, score in zip(labels, sample_scores)),
        key=lambda item: (-item[1], item[0]),
    )
    return [{"api": label, "score": score} for label, score in ordered[:n]]


def make_record_rows(
    runs: dict[str, dict[str, Any]],
    primary_scores: Any,
    primary_labels: Sequence[str],
    train_counts: Counter[str],
) -> list[dict[str, Any]]:
    primary = runs[PRIMARY_RUN]["predictions"]
    rows: list[dict[str, Any]] = []
    for idx in sorted(primary):
        item = primary[idx]
        true_apis = item["true_apis"]
        unseen_true = [api for api in true_apis if train_counts.get(api, 0) == 0]
        seen_true = [api for api in true_apis if train_counts.get(api, 0) > 0]
        predicted = item["predicted_apis"]
        row: dict[str, Any] = {
            "idx": idx,
            "mashup_id": item["mashup_id"],
            "mashup_name": item["mashup_name"],
            "text": item["text"],
            "true_apis": true_apis,
            "true_count": len(true_apis),
            "seen_true_apis": seen_true,
            "seen_true_count": len(seen_true),
            "unseen_true_apis": unseen_true,
            "unseen_true_count": len(unseen_true),
            "train_seen_recall_cap": len(seen_true) / len(true_apis) if true_apis else 0.0,
            "primary_predicted_apis": predicted,
            "primary_pred_count": len(predicted),
            "primary_top_scores": top_score_items(primary_labels, primary_scores[idx], n=10),
        }
        for run_name, run in runs.items():
            pred = run["predictions"][idx]
            row[f"{run_name}_predicted_apis"] = pred["predicted_apis"]
            row[f"{run_name}_pred_count"] = len(pred["predicted_apis"])
            row[f"{run_name}_tp"] = pred["true_positive"]
            row[f"{run_name}_fp"] = pred["false_positive"]
            row[f"{run_name}_fn"] = pred["false_negative"]
            row[f"{run_name}_precision"] = pred["precision"]
            row[f"{run_name}_recall"] = pred["recall"]
            row[f"{run_name}_f1"] = pred["f1"]
        row["primary_minus_tfidf_cosine_top1_f1"] = row[f"{PRIMARY_RUN}_f1"] - row["tfidf_cosine_top1_f1"]
        row["primary_minus_tfidf_cosine_top5_f1"] = row[f"{PRIMARY_RUN}_f1"] - row["tfidf_cosine_top5_f1"]
        row["primary_minus_cross_encoder_top1_f1"] = row[f"{PRIMARY_RUN}_f1"] - row["cross_encoder_top1_f1"]
        row["primary_minus_sbert_ovr_f1"] = row[f"{PRIMARY_RUN}_f1"] - row["sbert_ovr_logistic_f1"]
        rows.append(row)
    return rows


def make_label_rows(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    scores: Any,
    train_counts: Counter[str],
    feature_terms_by_label: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    test_counts = Counter(api for row in rows for api in row["true_apis"])
    pred_counts = Counter(api for row in rows for api in row["primary_predicted_apis"])
    tp_counts: Counter[str] = Counter()
    fp_counts: Counter[str] = Counter()
    fn_counts: Counter[str] = Counter()
    true_scores: defaultdict[str, list[float]] = defaultdict(list)
    false_scores: defaultdict[str, list[float]] = defaultdict(list)
    tp_scores: defaultdict[str, list[float]] = defaultdict(list)
    fn_scores: defaultdict[str, list[float]] = defaultdict(list)
    fp_scores: defaultdict[str, list[float]] = defaultdict(list)

    for row in rows:
        true_set = set(row["true_apis"])
        pred_set = set(row["primary_predicted_apis"])
        for api in true_set & pred_set:
            tp_counts[api] += 1
        for api in pred_set - true_set:
            fp_counts[api] += 1
        for api in true_set - pred_set:
            fn_counts[api] += 1
        for api in labels:
            score = float(scores[int(row["idx"]), label_to_idx[api]])
            if api in true_set:
                true_scores[api].append(score)
                if api in pred_set:
                    tp_scores[api].append(score)
                else:
                    fn_scores[api].append(score)
            else:
                false_scores[api].append(score)
                if api in pred_set:
                    fp_scores[api].append(score)

    all_labels = sorted(set(train_counts) | set(test_counts) | set(pred_counts) | set(labels))
    label_rows: list[dict[str, Any]] = []
    for api in all_labels:
        tp = tp_counts[api]
        fp = fp_counts[api]
        fn = fn_counts[api]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        terms = feature_terms_by_label.get(api, {"positive_terms": [], "negative_terms": []})
        label_rows.append(
            {
                "api": api,
                "train_count": train_counts[api],
                "test_count": test_counts[api],
                "predicted_count": pred_counts[api],
                "unseen_in_train": train_counts[api] == 0,
                "in_primary_label_space": api in label_to_idx,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_score_when_true": statistics.fmean(true_scores[api]) if true_scores[api] else "",
                "mean_score_when_false": statistics.fmean(false_scores[api]) if false_scores[api] else "",
                "mean_score_true_positive": statistics.fmean(tp_scores[api]) if tp_scores[api] else "",
                "mean_score_false_negative": statistics.fmean(fn_scores[api]) if fn_scores[api] else "",
                "mean_score_false_positive": statistics.fmean(fp_scores[api]) if fp_scores[api] else "",
                "top_positive_terms": terms["positive_terms"],
                "top_negative_terms": terms["negative_terms"],
            }
        )
    return label_rows


def extract_feature_terms(run_dir: Path, labels: Sequence[str], top_n: int) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]]]:
    import joblib
    import numpy as np

    vectorizer = joblib.load(run_dir / "vectorizer.joblib")
    classifier = joblib.load(run_dir / "model.joblib")
    feature_names = vectorizer.get_feature_names_out()
    by_label: dict[str, dict[str, list[str]]] = {}
    rows: list[dict[str, Any]] = []
    for label, estimator in zip(labels, classifier.estimators_):
        coef = estimator.coef_[0]
        if hasattr(coef, "toarray"):
            coef = coef.toarray()[0]
        coef = np.asarray(coef, dtype=float)
        positive_idx = np.argsort(coef)[-top_n:][::-1]
        negative_idx = np.argsort(coef)[:top_n]
        positive_terms = [str(feature_names[idx]) for idx in positive_idx]
        negative_terms = [str(feature_names[idx]) for idx in negative_idx]
        by_label[label] = {"positive_terms": positive_terms, "negative_terms": negative_terms}
        rows.append(
            {
                "api": label,
                "top_positive_terms": positive_terms,
                "top_positive_weights": [float(coef[idx]) for idx in positive_idx],
                "top_negative_terms": negative_terms,
                "top_negative_weights": [float(coef[idx]) for idx in negative_idx],
            }
        )
    return by_label, rows


def make_pairwise_summary(rows: Sequence[dict[str, Any]], run_names: Sequence[str]) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    primary_name = PRIMARY_RUN
    for other in run_names:
        if other == primary_name:
            continue
        better = [row for row in rows if row[f"{primary_name}_f1"] > row[f"{other}_f1"]]
        worse = [row for row in rows if row[f"{primary_name}_f1"] < row[f"{other}_f1"]]
        tie = len(rows) - len(better) - len(worse)
        primary_tp = sum(len(set(row[f"{primary_name}_predicted_apis"]) & set(row["true_apis"])) for row in rows)
        other_tp = sum(len(set(row[f"{other}_predicted_apis"]) & set(row["true_apis"])) for row in rows)
        shared_tp = sum(
            len(set(row[f"{primary_name}_predicted_apis"]) & set(row[f"{other}_predicted_apis"]) & set(row["true_apis"]))
            for row in rows
        )
        pair_rows.append(
            {
                "primary_run": primary_name,
                "comparison_run": other,
                "records_primary_higher_f1": len(better),
                "records_primary_lower_f1": len(worse),
                "records_equal_f1": tie,
                "mean_primary_minus_other_f1": statistics.fmean(
                    row[f"{primary_name}_f1"] - row[f"{other}_f1"] for row in rows
                ),
                "primary_total_tp": primary_tp,
                "comparison_total_tp": other_tp,
                "shared_true_positive_api_mentions": shared_tp,
                "primary_unique_true_positive_mentions": primary_tp - shared_tp,
                "comparison_unique_true_positive_mentions": other_tp - shared_tp,
            }
        )
    return pair_rows


def make_case_examples(rows: Sequence[dict[str, Any]], labels: Sequence[str], scores: Any, case_limit: int) -> dict[str, list[dict[str, Any]]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        idx = int(row["idx"])
        true_set = set(row["true_apis"])
        pred_set = set(row["primary_predicted_apis"])
        true_api_scores = {
            api: float(scores[idx, label_to_idx[api]])
            for api in row["true_apis"]
            if api in label_to_idx
        }
        false_positives = sorted(pred_set - true_set)
        false_negatives = sorted(true_set - pred_set)
        return {
            "idx": row["idx"],
            "mashup_name": row["mashup_name"],
            "text": row["text"],
            "true_apis": row["true_apis"],
            "unseen_true_apis": row["unseen_true_apis"],
            "primary_predicted_apis": row["primary_predicted_apis"],
            "primary_precision": row[f"{PRIMARY_RUN}_precision"],
            "primary_recall": row[f"{PRIMARY_RUN}_recall"],
            "primary_f1": row[f"{PRIMARY_RUN}_f1"],
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_api_scores_in_primary_model": true_api_scores,
            "primary_top_scores": row["primary_top_scores"],
            "tfidf_cosine_top1": row["tfidf_cosine_top1_predicted_apis"],
            "tfidf_cosine_top5": row["tfidf_cosine_top5_predicted_apis"],
            "cross_encoder_top1": row["cross_encoder_top1_predicted_apis"],
            "sbert_ovr_logistic": row["sbert_ovr_logistic_predicted_apis"],
            "primary_minus_tfidf_cosine_top1_f1": row["primary_minus_tfidf_cosine_top1_f1"],
            "primary_minus_cross_encoder_top1_f1": row["primary_minus_cross_encoder_top1_f1"],
            "primary_minus_sbert_ovr_f1": row["primary_minus_sbert_ovr_f1"],
        }

    exact_or_full = [
        row
        for row in rows
        if row[f"{PRIMARY_RUN}_recall"] == 1.0 and row[f"{PRIMARY_RUN}_precision"] >= 0.5
    ]
    exact_or_full = sorted(exact_or_full, key=lambda row: (-row[f"{PRIMARY_RUN}_f1"], -row["true_count"], row["idx"]))

    primary_beats_cosine = sorted(
        [row for row in rows if row["primary_minus_tfidf_cosine_top5_f1"] > 0],
        key=lambda row: (-row["primary_minus_tfidf_cosine_top5_f1"], -row[f"{PRIMARY_RUN}_f1"], row["idx"]),
    )
    primary_beats_sbert = sorted(
        [row for row in rows if row["primary_minus_sbert_ovr_f1"] > 0],
        key=lambda row: (-row["primary_minus_sbert_ovr_f1"], -row[f"{PRIMARY_RUN}_f1"], row["idx"]),
    )
    primary_worse_cross_top1 = sorted(
        [row for row in rows if row["primary_minus_cross_encoder_top1_f1"] < 0],
        key=lambda row: (row["primary_minus_cross_encoder_top1_f1"], row["idx"]),
    )
    unigram_better_bigram = sorted(
        [row for row in rows if row[f"{PRIMARY_RUN}_f1"] > row["tfidf_ovr_logistic_bigram_f1"]],
        key=lambda row: (-(row[f"{PRIMARY_RUN}_f1"] - row["tfidf_ovr_logistic_bigram_f1"]), row["idx"]),
    )
    zero_predictions = sorted(
        [row for row in rows if row[f"{PRIMARY_RUN}_pred_count"] == 0],
        key=lambda row: (-row["true_count"], -row["unseen_true_count"], row["idx"]),
    )
    unseen_label_cases = sorted(
        [row for row in rows if row["unseen_true_count"] > 0],
        key=lambda row: (-row["unseen_true_count"], row[f"{PRIMARY_RUN}_f1"], row["idx"]),
    )
    near_miss = []
    for row in rows:
        idx = int(row["idx"])
        pred_set = set(row["primary_predicted_apis"])
        candidates = []
        for api in row["true_apis"]:
            if api in label_to_idx and api not in pred_set:
                candidates.append((api, float(scores[idx, label_to_idx[api]])))
        if candidates:
            best_api, best_score = max(candidates, key=lambda item: item[1])
            if 0.35 <= best_score < 0.5:
                enriched = dict(row)
                enriched["near_miss_api"] = best_api
                enriched["near_miss_score"] = best_score
                near_miss.append(enriched)
    near_miss = sorted(near_miss, key=lambda row: (-row["near_miss_score"], row["idx"]))

    high_conf_fp = []
    for row in rows:
        idx = int(row["idx"])
        true_set = set(row["true_apis"])
        for api in set(row["primary_predicted_apis"]) - true_set:
            if api in label_to_idx:
                enriched = dict(row)
                enriched["false_positive_api"] = api
                enriched["false_positive_score"] = float(scores[idx, label_to_idx[api]])
                high_conf_fp.append(enriched)
    high_conf_fp = sorted(high_conf_fp, key=lambda row: (-row["false_positive_score"], row["idx"]))

    return {
        "primary_exact_or_full_recall_examples": [compact(row) for row in exact_or_full[:case_limit]],
        "primary_beats_tfidf_cosine_top5_examples": [compact(row) for row in primary_beats_cosine[:case_limit]],
        "primary_beats_sbert_ovr_examples": [compact(row) for row in primary_beats_sbert[:case_limit]],
        "primary_worse_than_cross_encoder_top1_examples": [compact(row) for row in primary_worse_cross_top1[:case_limit]],
        "unigram_logistic_better_than_bigram_logistic_examples": [compact(row) for row in unigram_better_bigram[:case_limit]],
        "zero_prediction_examples": [compact(row) for row in zero_predictions[:case_limit]],
        "unseen_label_failure_examples": [compact(row) for row in unseen_label_cases[:case_limit]],
        "near_miss_false_negative_examples": [
            {**compact(row), "near_miss_api": row["near_miss_api"], "near_miss_score": row["near_miss_score"]}
            for row in near_miss[:case_limit]
        ],
        "high_confidence_false_positive_examples": [
            {
                **compact(row),
                "false_positive_api": row["false_positive_api"],
                "false_positive_score": row["false_positive_score"],
            }
            for row in high_conf_fp[:case_limit]
        ],
    }


def make_summary(
    runs: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
    pairwise_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    primary_metrics = runs[PRIMARY_RUN]["metrics"]
    primary_pred_counts = [row[f"{PRIMARY_RUN}_pred_count"] for row in rows]
    true_counts = [row["true_count"] for row in rows]
    unseen_counts = [row["unseen_true_count"] for row in rows]
    labels_seen_in_train = [row for row in label_rows if row["train_count"] > 0]
    labels_unseen_in_train = [row for row in label_rows if row["test_count"] > 0 and row["train_count"] == 0]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Evidence packet for a separate LLM to analyze why TF-IDF OvR Logistic unigram performed well.",
        "primary_run": PRIMARY_RUN,
        "primary_metrics": primary_metrics,
        "method_metrics": {
            run_name: run["metrics"]
            for run_name, run in runs.items()
        },
        "record_count": len(rows),
        "true_api_count_summary": summarize_numeric([float(value) for value in true_counts]),
        "primary_predicted_api_count_summary": summarize_numeric([float(value) for value in primary_pred_counts]),
        "zero_prediction_records": sum(1 for count in primary_pred_counts if count == 0),
        "records_with_unseen_true_label": sum(1 for count in unseen_counts if count > 0),
        "unseen_true_label_mentions": sum(unseen_counts),
        "train_seen_recall_cap_sample_average": statistics.fmean(row["train_seen_recall_cap"] for row in rows),
        "labels_seen_in_train": len(labels_seen_in_train),
        "test_label_types_unseen_in_train": len(labels_unseen_in_train),
        "pairwise_against_primary": pairwise_rows,
        "top_primary_label_f1_seen_in_test": sorted(
            [row for row in label_rows if row["test_count"] > 0 and row["in_primary_label_space"]],
            key=lambda row: (-float(row["f1"]), -int(row["test_count"]), row["api"]),
        )[:20],
        "lowest_primary_label_f1_seen_in_test": sorted(
            [row for row in label_rows if row["test_count"] > 0 and row["in_primary_label_space"]],
            key=lambda row: (float(row["f1"]), -int(row["test_count"]), row["api"]),
        )[:20],
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value).replace("\n", " ")


def write_markdown_packet(
    path: Path,
    summary: dict[str, Any],
    pairwise_rows: Sequence[dict[str, Any]],
    case_examples: dict[str, list[dict[str, Any]]],
) -> None:
    method_rows = []
    for run_name, metrics in summary["method_metrics"].items():
        method_rows.append(
            [
                run_name,
                metrics.get("average_number_of_predicted_apis", ""),
                metrics.get("precision", ""),
                metrics.get("recall", ""),
                metrics.get("f1", ""),
                metrics.get("micro_precision", ""),
                metrics.get("micro_recall", ""),
                metrics.get("micro_f1", ""),
            ]
        )
    method_rows = sorted(method_rows, key=lambda row: float(row[4] or 0.0), reverse=True)

    pair_rows = [
        [
            row["comparison_run"],
            row["records_primary_higher_f1"],
            row["records_primary_lower_f1"],
            row["records_equal_f1"],
            row["mean_primary_minus_other_f1"],
            row["primary_unique_true_positive_mentions"],
            row["comparison_unique_true_positive_mentions"],
        ]
        for row in pairwise_rows
    ]

    lines = [
        "# TF-IDF OvR Analysis Materials",
        "",
        "This packet is for a separate LLM. It is intentionally evidence-oriented: use it to form hypotheses, but do not treat the bullet points as final conclusions.",
        "",
        "## Files",
        "",
        "- `summary.json`: compact machine-readable overview.",
        "- `records_comparison.csv`: one row per test record, with predictions/metrics from each run.",
        "- `label_diagnostics.csv`: per-API train/test frequency, TP/FP/FN, score averages, and top coefficient terms.",
        "- `feature_terms_by_label.csv`: top positive/negative TF-IDF coefficient terms for every primary-run label.",
        "- `pairwise_primary_vs_other_runs.csv`: record-level and true-positive overlap against the primary run.",
        "- `case_examples.json`: curated full-text examples for success, failure, near miss, and method-difference analysis.",
        "",
        "## Primary Run",
        "",
        f"- Primary run: `{summary['primary_run']}`",
        f"- Records: {summary['record_count']}",
        f"- Macro/sample precision: {summary['primary_metrics']['precision']:.4f}",
        f"- Macro/sample recall: {summary['primary_metrics']['recall']:.4f}",
        f"- Macro/sample F1: {summary['primary_metrics']['f1']:.4f}",
        f"- Micro precision: {summary['primary_metrics']['micro_precision']:.4f}",
        f"- Micro recall: {summary['primary_metrics']['micro_recall']:.4f}",
        f"- Micro F1: {summary['primary_metrics']['micro_f1']:.4f}",
        f"- Average predicted APIs: {summary['primary_metrics']['average_number_of_predicted_apis']:.4f}",
        f"- Zero-prediction records: {summary['zero_prediction_records']}",
        f"- Records with unseen true labels: {summary['records_with_unseen_true_label']}",
        f"- Unseen true-label mentions: {summary['unseen_true_label_mentions']}",
        "",
        "## Method Metrics",
        "",
        markdown_table(
            [
                "run",
                "avg_pred",
                "precision",
                "recall",
                "F1",
                "micro_P",
                "micro_R",
                "micro_F1",
            ],
            method_rows,
        ),
        "",
        "## Pairwise Against Primary",
        "",
        markdown_table(
            [
                "comparison",
                "primary higher F1 records",
                "primary lower F1 records",
                "equal F1 records",
                "mean F1 delta",
                "primary unique TP",
                "comparison unique TP",
            ],
            pair_rows,
        ),
        "",
        "## Suggested Questions For The Analysis LLM",
        "",
        "1. Does the primary run's strength come mainly from high precision, reasonable recall, or a favorable average recommendation count?",
        "2. Which labels are easy for unigram TF-IDF OvR, and do their top coefficient terms look like direct lexical matches to mashup descriptions?",
        "3. Where does unigram logistic beat TF-IDF cosine ranking, and is the difference mostly from supervised label priors or from filtering noisy cosine matches?",
        "4. Where does Cross Encoder top1 beat the primary run, especially on records where the primary run predicts nothing?",
        "5. How much of the remaining error is unavoidable for OvR because true labels are unseen in train?",
        "6. Does adding bigrams help precision at the cost of recall, or mostly add sparsity?",
        "",
        "## Case Example Index",
        "",
    ]
    for group_name, examples in case_examples.items():
        lines.append(f"### {group_name}")
        for example in examples[:3]:
            lines.append(
                f"- idx={example['idx']} name={example['mashup_name']} "
                f"F1={example['primary_f1']:.3f} true={example['true_apis']} "
                f"pred={example['primary_predicted_apis']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, Any]] = {}
    for run_name, spec in RUN_SPECS.items():
        if spec["kind"] == "ovr":
            runs[run_name] = load_ovr_run(spec["path"])
        else:
            runs[run_name] = load_ranking_run(spec["path"])
        runs[run_name]["label"] = spec["label"]
        enrich_metrics_from_predictions(runs[run_name])

    primary_run_dir = RUN_SPECS[PRIMARY_RUN]["path"]
    primary_predictions = runs[PRIMARY_RUN]["raw_predictions"]
    texts = [item["text"] for item in sorted(primary_predictions, key=lambda item: int(item["idx"]))]
    labels, _vectorizer, primary_scores = compute_full_primary_scores(primary_run_dir, texts)
    train_counts = load_train_counts(runs[PRIMARY_RUN]["config"])
    feature_terms_by_label, feature_rows = extract_feature_terms(primary_run_dir, labels, args.top_feature_terms)

    rows = make_record_rows(runs, primary_scores, labels, train_counts)
    label_rows = make_label_rows(rows, labels, primary_scores, train_counts, feature_terms_by_label)
    pairwise_rows = make_pairwise_summary(rows, list(RUN_SPECS))
    case_examples = make_case_examples(rows, labels, primary_scores, args.case_limit)
    summary = make_summary(runs, rows, label_rows, pairwise_rows)

    record_fieldnames = [
        "idx",
        "mashup_id",
        "mashup_name",
        "text",
        "true_apis",
        "true_count",
        "seen_true_apis",
        "seen_true_count",
        "unseen_true_apis",
        "unseen_true_count",
        "train_seen_recall_cap",
        "primary_predicted_apis",
        "primary_pred_count",
        "primary_top_scores",
    ]
    for run_name in RUN_SPECS:
        record_fieldnames.extend(
            [
                f"{run_name}_predicted_apis",
                f"{run_name}_pred_count",
                f"{run_name}_tp",
                f"{run_name}_fp",
                f"{run_name}_fn",
                f"{run_name}_precision",
                f"{run_name}_recall",
                f"{run_name}_f1",
            ]
        )
    record_fieldnames.extend(
        [
            "primary_minus_tfidf_cosine_top1_f1",
            "primary_minus_tfidf_cosine_top5_f1",
            "primary_minus_cross_encoder_top1_f1",
            "primary_minus_sbert_ovr_f1",
        ]
    )

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "case_examples.json", case_examples)
    write_csv(output_dir / "records_comparison.csv", rows, record_fieldnames)
    write_csv(
        output_dir / "label_diagnostics.csv",
        label_rows,
        [
            "api",
            "train_count",
            "test_count",
            "predicted_count",
            "unseen_in_train",
            "in_primary_label_space",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_score_when_true",
            "mean_score_when_false",
            "mean_score_true_positive",
            "mean_score_false_negative",
            "mean_score_false_positive",
            "top_positive_terms",
            "top_negative_terms",
        ],
    )
    write_csv(
        output_dir / "feature_terms_by_label.csv",
        feature_rows,
        [
            "api",
            "top_positive_terms",
            "top_positive_weights",
            "top_negative_terms",
            "top_negative_weights",
        ],
    )
    write_csv(
        output_dir / "pairwise_primary_vs_other_runs.csv",
        pairwise_rows,
        [
            "primary_run",
            "comparison_run",
            "records_primary_higher_f1",
            "records_primary_lower_f1",
            "records_equal_f1",
            "mean_primary_minus_other_f1",
            "primary_total_tp",
            "comparison_total_tp",
            "shared_true_positive_api_mentions",
            "primary_unique_true_positive_mentions",
            "comparison_unique_true_positive_mentions",
        ],
    )
    write_markdown_packet(output_dir / "README_FOR_ANALYSIS_LLM.md", summary, pairwise_rows, case_examples)

    manifest = {
        "generated_at": summary["generated_at"],
        "source_base_dir": str(BASE_DIR),
        "primary_run": PRIMARY_RUN,
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "api_catalog_path": str(DEFAULT_API_CATALOG_PATH),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"[OK] wrote analysis materials to {output_dir}")


if __name__ == "__main__":
    main()
