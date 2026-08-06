#!/usr/bin/env python3
"""TF-IDF + One-vs-Rest multi-label API classification baselines.

Examples:
  python codex/baseline/run_tfidf_ovr_baseline.py \
    --data_path study_data_mashup_apis.json \
    --text_key description \
    --label_key apis \
    --model_type logistic \
    --output_dir codex/baseline/results/tfidf_ovr_logistic

  python codex/baseline/run_tfidf_ovr_baseline.py \
    --train_path study_data_mashup_apis.json \
    --test_path codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
    --model_type svm \
    --tune_threshold \
    --output_dir codex/baseline/results/tfidf_ovr_svm_tuned
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from baseline_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_PATH,
    MashupRecord,
    average_metrics,
    calculate_metrics,
    resolved_eval_path,
    unique_preserve_order,
)


LOGGER = logging.getLogger("tfidf_ovr_baseline")


def parse_min_df(value: str) -> int | float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("min_df must be > 0")
    if parsed.is_integer() and parsed >= 1:
        return int(parsed)
    if parsed > 1:
        raise argparse.ArgumentTypeError("float min_df must be <= 1.0")
    return parsed


def parse_max_df(value: str) -> int | float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max_df must be > 0")
    if parsed.is_integer() and parsed >= 1:
        return int(parsed)
    if parsed > 1:
        raise argparse.ArgumentTypeError("float max_df must be <= 1.0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + One-vs-Rest Logistic Regression or Linear SVM API baseline."
    )
    parser.add_argument(
        "--data_path",
        "--data-path",
        default=None,
        help="Full mashup JSON path used when explicit train/test paths are not supplied.",
    )
    parser.add_argument(
        "--train_path",
        "--train-path",
        default=None,
        help="Optional training JSON path. If --test_path is also set, no random test split is made.",
    )
    parser.add_argument("--test_path", "--test-path", "--eval-path", default=None, help="Optional test JSON path.")
    parser.add_argument("--valid_path", "--valid-path", default=None, help="Optional validation JSON path.")
    parser.add_argument("--text_key", "--text-key", default="description", help="Text key for dict records.")
    parser.add_argument("--label_key", "--label-key", default="apis", help="Label key for dict records.")
    parser.add_argument("--id_key", "--id-key", default=None, help="Optional sample id key for dict records.")
    parser.add_argument("--model_type", "--model-type", choices=["logistic", "svm"], required=True)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR / "tfidf_ovr"),
        help="Directory where metrics, predictions, labels, config, model, and vectorizer are saved.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Global score threshold.")
    parser.add_argument("--tune_threshold", "--tune-threshold", action="store_true")
    parser.add_argument("--test_size", "--test-size", type=float, default=0.2)
    parser.add_argument("--valid_size", "--valid-size", type=float, default=0.2)
    parser.add_argument("--random_state", "--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file record limit for smoke tests.")
    parser.add_argument("--train_limit", "--train-limit", type=int, default=None)
    parser.add_argument("--test_limit", "--test-limit", type=int, default=None)
    parser.add_argument("--valid_limit", "--valid-limit", type=int, default=None)

    parser.add_argument("--ngram_min", "--ngram-min", type=int, default=1)
    parser.add_argument("--ngram_max", "--ngram-max", type=int, default=2)
    parser.add_argument("--min_df", "--min-df", type=parse_min_df, default=1)
    parser.add_argument("--max_df", "--max-df", type=parse_max_df, default=1.0)
    parser.add_argument("--max_features", "--max-features", type=int, default=None)
    parser.add_argument("--no_sublinear_tf", "--no-sublinear-tf", action="store_true")
    parser.add_argument(
        "--stop_words",
        "--stop-words",
        default="english",
        help='TfidfVectorizer stop_words. Use "none" for None. Default: english.',
    )

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", "--max-iter", type=int, default=None)
    parser.add_argument("--class_weight", "--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--n_jobs", "--n-jobs", type=int, default=-1)
    parser.add_argument("--save_all_scores", "--save-all-scores", action="store_true")
    parser.add_argument(
        "--save_legacy_json",
        "--save-legacy-json",
        action="store_true",
        help="Also save a compact existing-style JSON with average_metrics and records.",
    )
    parser.add_argument(
        "--target",
        default="16_4_3",
        help="Legacy target name used only when resolving the default existing test path.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def _get_by_key(entry: Any, key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    if isinstance(entry, dict):
        return entry.get(key, default)
    if isinstance(entry, list) and str(key).isdigit():
        idx = int(str(key))
        return entry[idx] if 0 <= idx < len(entry) else default
    return default


def _extract_text(entry: Any, text_key: str) -> str:
    value = _get_by_key(entry, text_key)
    if value is None and isinstance(entry, dict):
        value = entry.get("description")
    if value is None and isinstance(entry, list):
        value = entry[1] if len(entry) > 1 else ""
    return str(value or "")


def _extract_labels(entry: Any, label_key: str) -> list[str]:
    value = _get_by_key(entry, label_key)
    if value is None and isinstance(entry, dict):
        value = entry.get("apis", [])
    if value is None and isinstance(entry, list):
        value = entry[2] if len(entry) > 2 else []
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        return [str(value)]
    return unique_preserve_order(value)


def _extract_id(entry: Any, fallback_name: str, id_key: str | None) -> str:
    value = _get_by_key(entry, id_key) if id_key else None
    if value is None and isinstance(entry, dict):
        value = entry.get("id", fallback_name)
    if value is None and isinstance(entry, list):
        value = entry[3] if len(entry) > 3 else fallback_name
    return str(value if value is not None else fallback_name)


def load_records_with_keys(
    path: os.PathLike[str] | str,
    *,
    text_key: str,
    label_key: str,
    id_key: str | None,
    limit: int | None = None,
    skip_empty: bool = True,
) -> tuple[list[MashupRecord], dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        iterator = data.items()
    elif isinstance(data, list):
        iterator = ((str(i), value) for i, value in enumerate(data))
    else:
        raise ValueError(f"Unsupported mashup JSON root type: {type(data).__name__}")

    records: list[MashupRecord] = []
    stats = {"loaded": 0, "skipped_empty_text": 0, "skipped_empty_labels": 0}
    for idx, (name, entry) in enumerate(iterator):
        if limit is not None and stats["loaded"] >= limit:
            break
        text = _extract_text(entry, text_key).strip()
        labels = _extract_labels(entry, label_key)
        if skip_empty and not text:
            stats["skipped_empty_text"] += 1
            continue
        if skip_empty and not labels:
            stats["skipped_empty_labels"] += 1
            continue
        sample_id = _extract_id(entry, str(name), id_key)
        records.append(
            MashupRecord(
                idx=len(records),
                mashup_name=str(name),
                mashup_id=sample_id,
                text=text,
                true_apis=labels,
            )
        )
        stats["loaded"] += 1
    return records, stats


def split_records(args: argparse.Namespace) -> tuple[list[MashupRecord], list[MashupRecord], list[MashupRecord], dict[str, Any]]:
    from sklearn.model_selection import train_test_split

    split_info: dict[str, Any] = {}
    common_kwargs = {
        "text_key": args.text_key,
        "label_key": args.label_key,
        "id_key": args.id_key,
    }

    if args.train_path or args.test_path or args.valid_path:
        train_path = Path(args.train_path or args.data_path or DEFAULT_TRAIN_PATH)
        if args.test_path:
            test_path = Path(args.test_path)
        elif args.data_path:
            test_path = Path(args.data_path)
        else:
            test_path = resolved_eval_path(args)
        valid_path = Path(args.valid_path) if args.valid_path else None

        train_records, train_stats = load_records_with_keys(
            train_path,
            limit=args.train_limit if args.train_limit is not None else args.limit,
            **common_kwargs,
        )
        test_records, test_stats = load_records_with_keys(
            test_path,
            limit=args.test_limit if args.test_limit is not None else args.limit,
            **common_kwargs,
        )
        valid_records: list[MashupRecord] = []
        valid_stats = None
        if valid_path:
            valid_records, valid_stats = load_records_with_keys(
                valid_path,
                limit=args.valid_limit if args.valid_limit is not None else args.limit,
                **common_kwargs,
            )
        split_info.update(
            {
                "split_source": "explicit_files",
                "train_path": str(train_path.resolve()),
                "test_path": str(test_path.resolve()),
                "valid_path": str(valid_path.resolve()) if valid_path else None,
                "load_stats": {"train": train_stats, "test": test_stats, "valid": valid_stats},
            }
        )
    elif args.data_path:
        data_path = Path(args.data_path)
        records, stats = load_records_with_keys(
            data_path,
            limit=args.limit,
            **common_kwargs,
        )
        train_records, test_records = train_test_split(
            records,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
        )
        valid_records = []
        split_info.update(
            {
                "split_source": "random_split_from_data_path",
                "data_path": str(data_path.resolve()),
                "test_size": args.test_size,
                "load_stats": {"data": stats},
            }
        )
    else:
        train_path = DEFAULT_TRAIN_PATH
        test_path = resolved_eval_path(args)
        train_records, train_stats = load_records_with_keys(train_path, **common_kwargs)
        test_records, test_stats = load_records_with_keys(test_path, **common_kwargs)
        valid_records = []
        split_info.update(
            {
                "split_source": "existing_default_files",
                "train_path": str(train_path.resolve()),
                "test_path": str(test_path.resolve()),
                "valid_path": None,
                "load_stats": {"train": train_stats, "test": test_stats, "valid": None},
            }
        )

    if args.tune_threshold and not valid_records:
        train_records, valid_records = train_test_split(
            train_records,
            test_size=args.valid_size,
            random_state=args.random_state,
            shuffle=True,
        )
        split_info["validation_split"] = {
            "source": "train_records",
            "valid_size": args.valid_size,
            "random_state": args.random_state,
        }
    elif args.tune_threshold:
        split_info["validation_split"] = {"source": "valid_path"}
    else:
        split_info["validation_split"] = None

    if not train_records:
        raise SystemExit("No training records are available after loading/skipping empty samples.")
    if not test_records:
        raise SystemExit("No test records are available after loading/skipping empty samples.")
    if args.tune_threshold and not valid_records:
        raise SystemExit("No validation records are available for threshold tuning.")

    split_info["num_train_records"] = len(train_records)
    split_info["num_valid_records"] = len(valid_records)
    split_info["num_test_records"] = len(test_records)
    return train_records, valid_records, test_records, split_info


def build_label_list(records: Sequence[MashupRecord]) -> list[str]:
    labels = sorted({api for record in records for api in record.true_apis})
    if not labels:
        raise SystemExit("No API labels found in training records.")
    return labels


def make_multihot(records: Sequence[MashupRecord], api_to_idx: dict[str, int]) -> Any:
    import numpy as np

    matrix = np.zeros((len(records), len(api_to_idx)), dtype=int)
    for row_idx, record in enumerate(records):
        for api in record.true_apis:
            col_idx = api_to_idx.get(api)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1
    return matrix


def warn_unknown_labels(
    split_name: str,
    records: Sequence[MashupRecord],
    api_to_idx: dict[str, int],
) -> list[str]:
    unknown = sorted({api for record in records for api in record.true_apis if api not in api_to_idx})
    if unknown:
        LOGGER.warning(
            "%s contains %d labels that are not in the training label list. Examples: %s",
            split_name,
            len(unknown),
            ", ".join(unknown[:10]),
        )
    return unknown


def log_skipped_records(split_info: dict[str, Any]) -> None:
    load_stats = split_info.get("load_stats", {})
    for split_name, stats in load_stats.items():
        if not stats:
            continue
        skipped_text = int(stats.get("skipped_empty_text", 0))
        skipped_labels = int(stats.get("skipped_empty_labels", 0))
        if skipped_text or skipped_labels:
            LOGGER.warning(
                "%s skipped %d empty-text records and %d empty-label records.",
                split_name,
                skipped_text,
                skipped_labels,
            )


def build_vectorizer(args: argparse.Namespace) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    stop_words = None if str(args.stop_words).lower() == "none" else args.stop_words
    return TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=not args.no_sublinear_tf,
    )


def build_classifier(args: argparse.Namespace) -> Any:
    from sklearn.multiclass import OneVsRestClassifier

    class_weight = None if args.class_weight == "none" else args.class_weight
    max_iter = args.max_iter
    if args.model_type == "logistic":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(
            C=args.C,
            solver="liblinear",
            max_iter=max_iter if max_iter is not None else 1000,
            class_weight=class_weight,
            random_state=args.random_state,
        )
    else:
        from sklearn.svm import LinearSVC

        estimator = LinearSVC(
            C=args.C,
            class_weight=class_weight,
            max_iter=max_iter if max_iter is not None else 5000,
            random_state=args.random_state,
        )
    return OneVsRestClassifier(estimator, n_jobs=args.n_jobs)


def fit_model(
    args: argparse.Namespace,
    train_records: Sequence[MashupRecord],
    api_to_idx: dict[str, int],
) -> tuple[Any, Any]:
    vectorizer = build_vectorizer(args)
    x_train = vectorizer.fit_transform([record.text for record in train_records])
    y_train = make_multihot(train_records, api_to_idx)
    classifier = build_classifier(args)
    classifier.fit(x_train, y_train)
    return vectorizer, classifier


def score_records(args: argparse.Namespace, vectorizer: Any, classifier: Any, records: Sequence[MashupRecord]) -> Any:
    import numpy as np

    x = vectorizer.transform([record.text for record in records])
    if args.model_type == "logistic":
        scores = classifier.predict_proba(x)
    else:
        scores = classifier.decision_function(x)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores


def sorted_predicted_apis(label_names: Sequence[str], sample_scores: Sequence[float], threshold: float) -> list[str]:
    selected = [
        (label_names[idx], float(score))
        for idx, score in enumerate(sample_scores)
        if float(score) >= threshold
    ]
    return [api for api, _score in sorted(selected, key=lambda item: (-item[1], item[0]))]


def scores_for_record(
    *,
    label_names: Sequence[str],
    sample_scores: Sequence[float],
    true_apis: Sequence[str],
    predicted_apis: Sequence[str],
    save_all_scores: bool,
) -> dict[str, float]:
    if save_all_scores:
        indices = range(len(label_names))
    else:
        keep = set(true_apis).union(predicted_apis)
        indices = [idx for idx, api in enumerate(label_names) if api in keep]
    return {label_names[idx]: float(sample_scores[idx]) for idx in indices}


def build_predictions(
    *,
    records: Sequence[MashupRecord],
    scores: Any,
    label_names: Sequence[str],
    threshold: float,
    save_all_scores: bool,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record, sample_scores in zip(records, scores):
        predicted_apis = sorted_predicted_apis(label_names, sample_scores, threshold)
        metrics = calculate_metrics(record.true_apis, predicted_apis)
        predictions.append(
            {
                "sample_id": record.mashup_id,
                "idx": record.idx,
                "mashup_id": record.mashup_id,
                "mashup_name": record.mashup_name,
                "text": record.text,
                "true_apis": record.true_apis,
                "predicted_apis": predicted_apis,
                "scores": scores_for_record(
                    label_names=label_names,
                    sample_scores=sample_scores,
                    true_apis=record.true_apis,
                    predicted_apis=predicted_apis,
                    save_all_scores=save_all_scores,
                ),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
            }
        )
    return predictions


def prediction_metrics(predictions: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    averages = average_metrics(predictions)
    total_tp = sum(int(item["true_positive"]) for item in predictions)
    total_fp = sum(int(item["false_positive"]) for item in predictions)
    total_fn = sum(int(item["false_negative"]) for item in predictions)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    num_records = len(predictions)
    avg_predicted = (
        sum(len(item["predicted_apis"]) for item in predictions) / num_records if num_records else 0.0
    )
    avg_true = sum(len(item["true_apis"]) for item in predictions) / num_records if num_records else 0.0
    return {
        "precision": averages["precision"],
        "recall": averages["recall"],
        "f1": averages["f1"],
        "macro_precision": averages["precision"],
        "macro_recall": averages["recall"],
        "macro_f1": averages["f1"],
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "average_number_of_predicted_apis": avg_predicted,
        "average_number_of_true_apis": avg_true,
        "total_true_positive": total_tp,
        "total_false_positive": total_fp,
        "total_false_negative": total_fn,
        "num_records": num_records,
    }


def tune_threshold(
    *,
    args: argparse.Namespace,
    train_records: Sequence[MashupRecord],
    valid_records: Sequence[MashupRecord],
    label_names: Sequence[str],
    api_to_idx: dict[str, int],
) -> tuple[float, list[dict[str, float]]]:
    import numpy as np

    vectorizer, classifier = fit_model(args, train_records, api_to_idx)
    valid_scores = score_records(args, vectorizer, classifier, valid_records)
    if args.model_type == "logistic":
        candidates = np.round(np.arange(0.05, 1.0, 0.05), 10)
    else:
        min_score = float(valid_scores.min())
        max_score = float(valid_scores.max())
        candidates = np.asarray([min_score]) if min_score == max_score else np.linspace(min_score, max_score, 101)

    history: list[dict[str, float]] = []
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for candidate in candidates:
        threshold = float(candidate)
        predictions = build_predictions(
            records=valid_records,
            scores=valid_scores,
            label_names=label_names,
            threshold=threshold,
            save_all_scores=False,
        )
        metrics = prediction_metrics(predictions)
        item = {
            "threshold": threshold,
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }
        history.append(item)
        if item["f1"] > best_f1:
            best_f1 = item["f1"]
            best_threshold = threshold
    LOGGER.info("Tuned threshold on validation set: %.6f (macro F1 %.6f)", best_threshold, best_f1)
    return best_threshold, history


def json_dump(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_required_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    vectorizer: Any,
    classifier: Any,
    label_names: Sequence[str],
    threshold: float,
    predictions: Sequence[dict[str, Any]],
    metrics: dict[str, float | int],
    split_info: dict[str, Any],
    unknown_labels: dict[str, list[str]],
    threshold_history: list[dict[str, float]],
) -> None:
    import joblib

    api_to_idx = {api: idx for idx, api in enumerate(label_names)}
    labels_payload = {
        "api_to_idx": api_to_idx,
        "idx_to_api": {str(idx): api for idx, api in enumerate(label_names)},
    }
    tfidf_params = {
        "lowercase": True,
        "stop_words": None if str(args.stop_words).lower() == "none" else args.stop_words,
        "ngram_range": [args.ngram_min, args.ngram_max],
        "min_df": args.min_df,
        "max_df": args.max_df,
        "max_features": args.max_features,
        "sublinear_tf": not args.no_sublinear_tf,
    }
    classifier_params = {
        "C": args.C,
        "max_iter": args.max_iter if args.max_iter is not None else (1000 if args.model_type == "logistic" else 5000),
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "n_jobs": args.n_jobs,
    }
    if args.model_type == "logistic":
        classifier_params["solver"] = "liblinear"
    config = {
        "model_type": args.model_type,
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "train_path": split_info.get("train_path"),
        "test_path": split_info.get("test_path"),
        "valid_path": split_info.get("valid_path"),
        "text_key": args.text_key,
        "label_key": args.label_key,
        "id_key": args.id_key,
        "threshold": threshold,
        "requested_threshold": args.threshold,
        "tuned_threshold": bool(args.tune_threshold),
        "tfidf_params": tfidf_params,
        "classifier_params": classifier_params,
        "random_state": args.random_state,
        "split": split_info,
        "num_labels": len(label_names),
        "score_type": "predict_proba" if args.model_type == "logistic" else "decision_function",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unknown_labels": unknown_labels,
    }

    json_dump(output_dir / "metrics.json", metrics)
    json_dump(output_dir / "predictions.json", list(predictions))
    json_dump(output_dir / "labels.json", labels_payload)
    json_dump(output_dir / "config.json", config)
    if threshold_history:
        json_dump(output_dir / "threshold_tuning.json", threshold_history)

    joblib.dump(classifier, output_dir / "model.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")


def save_legacy_json(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    predictions: Sequence[dict[str, Any]],
    metrics: dict[str, float | int],
    threshold: float,
) -> None:
    records: list[dict[str, Any]] = []
    for item in predictions:
        predicted_apis = list(item["predicted_apis"])
        records.append(
            {
                "idx": item["idx"],
                "mashup_id": item["mashup_id"],
                "mashup_name": item["mashup_name"],
                "text": item["text"],
                "true_apis": item["true_apis"],
                "recommended_apis": predicted_apis,
                "scores": [item["scores"].get(api) for api in predicted_apis],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "true_positive": item["true_positive"],
                "false_positive": item["false_positive"],
                "false_negative": item["false_negative"],
            }
        )
    payload = {
        "metadata": {
            "method": f"tfidf_ovr_{args.model_type}",
            "selection_mode": "threshold",
            "threshold": threshold,
            "score_type": "predict_proba" if args.model_type == "logistic" else "decision_function",
            "num_records": len(records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "average_metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "records": records,
    }
    json_dump(output_dir / "legacy_results.json", payload)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    if args.ngram_min <= 0 or args.ngram_max < args.ngram_min:
        raise SystemExit("--ngram_min must be positive and <= --ngram_max")
    if args.threshold is None:
        args.threshold = 0.5 if args.model_type == "logistic" else 0.0

    train_records, valid_records, test_records, split_info = split_records(args)
    label_source_records = train_records + valid_records if args.tune_threshold else train_records
    label_names = build_label_list(label_source_records)
    api_to_idx = {api: idx for idx, api in enumerate(label_names)}
    unknown_labels = {
        "valid": warn_unknown_labels("validation set", valid_records, api_to_idx) if valid_records else [],
        "test": warn_unknown_labels("test set", test_records, api_to_idx),
    }
    log_skipped_records(split_info)

    LOGGER.info(
        "Loaded %d train, %d validation, %d test records with %d labels.",
        len(train_records),
        len(valid_records),
        len(test_records),
        len(label_names),
    )

    threshold_history: list[dict[str, float]] = []
    threshold = float(args.threshold)
    if args.tune_threshold:
        threshold, threshold_history = tune_threshold(
            args=args,
            train_records=train_records,
            valid_records=valid_records,
            label_names=label_names,
            api_to_idx=api_to_idx,
        )

    LOGGER.info("Fitting final %s model.", args.model_type)
    final_train_records = train_records + valid_records if args.tune_threshold else train_records
    vectorizer, classifier = fit_model(args, final_train_records, api_to_idx)
    test_scores = score_records(args, vectorizer, classifier, test_records)
    predictions = build_predictions(
        records=test_records,
        scores=test_scores,
        label_names=label_names,
        threshold=threshold,
        save_all_scores=args.save_all_scores,
    )
    metrics = prediction_metrics(predictions)
    metrics.update(
        {
            "threshold": threshold,
            "tuned_threshold": bool(args.tune_threshold),
            "model_type": args.model_type,
            "num_labels": len(label_names),
        }
    )

    save_required_outputs(
        output_dir=output_dir,
        args=args,
        vectorizer=vectorizer,
        classifier=classifier,
        label_names=label_names,
        threshold=threshold,
        predictions=predictions,
        metrics=metrics,
        split_info=split_info,
        unknown_labels=unknown_labels,
        threshold_history=threshold_history,
    )
    if args.save_legacy_json:
        save_legacy_json(
            output_dir=output_dir,
            args=args,
            predictions=predictions,
            metrics=metrics,
            threshold=threshold,
        )

    LOGGER.info("[Average Metrics]")
    LOGGER.info("Precision: %.6f", metrics["precision"])
    LOGGER.info("Recall:    %.6f", metrics["recall"])
    LOGGER.info("F1:        %.6f", metrics["f1"])
    LOGGER.info("[OK] saved outputs in %s", output_dir)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
