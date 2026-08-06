#!/usr/bin/env python3
"""Sentence-BERT + One-vs-Rest multi-label API classification baselines.

Examples:
  python scripts/baselines/run_sbert_ovr_baseline.py \
    --train_path data/processed/root_json/study_data_mashup_apis.json \
    --test_path results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
    --model_type logistic \
    --threshold 0.5 \
    --output_dir results/baselines/sbert_ovr_logistic

  python scripts/baselines/run_sbert_ovr_baseline.py \
    --train_path data/processed/root_json/study_data_mashup_apis.json \
    --test_path results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
    --model_type logistic \
    --tune_threshold \
    --output_dir results/baselines/sbert_ovr_logistic_tuned
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from baseline_utils import DEFAULT_OUTPUT_DIR, MashupRecord
from run_tfidf_ovr_baseline import (
    build_classifier,
    build_label_list,
    build_predictions,
    json_dump,
    make_multihot,
    prediction_metrics,
    split_records,
)


DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOGGER = logging.getLogger("sbert_ovr_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sentence-BERT + One-vs-Rest Logistic Regression or Linear SVM API baseline."
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
    parser.add_argument("--test_path", "--test-path", default=None, help="Optional test JSON path.")
    parser.add_argument("--eval_path", "--eval-path", default=None, help="Alias used for the default test JSON path.")
    parser.add_argument("--valid_path", "--valid-path", default=None, help="Optional validation JSON path.")
    parser.add_argument("--text_key", "--text-key", default="description", help="Text key for dict records.")
    parser.add_argument("--label_key", "--label-key", default="apis", help="Label key for dict records.")
    parser.add_argument("--id_key", "--id-key", default=None, help="Optional sample id key for dict records.")
    parser.add_argument("--model_type", "--model-type", choices=["logistic", "svm"], required=True)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR / "sbert_ovr"),
        help="Directory where metrics, predictions, labels, config, and model are saved.",
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

    parser.add_argument("--model_name", "--model-name", default=DEFAULT_SBERT_MODEL, help="SentenceTransformer model name/path.")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--device", default=None, help="Optional device passed to SentenceTransformer, e.g. cuda or cpu.")
    parser.add_argument(
        "--no_normalize_embeddings",
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization for Sentence-BERT embeddings.",
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
    args = parser.parse_args()
    if args.test_path is None and args.eval_path is not None:
        args.test_path = args.eval_path
    return args


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


class SbertEncoder:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.device = args.device
        self.normalize_embeddings = not args.no_normalize_embeddings
        if self.device:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = SentenceTransformer(self.model_name)

    def encode_records(self, records: Sequence[MashupRecord], *, show_progress_bar: bool) -> Any:
        embeddings = self.model.encode(
            [record.text for record in records],
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.astype("float32")

    def config(self, embedding_dim: int | None = None) -> dict[str, Any]:
        return {
            "feature_extractor": "sentence_bert",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_dim": embedding_dim,
        }


def fit_model(
    args: argparse.Namespace,
    encoder: SbertEncoder,
    train_records: Sequence[MashupRecord],
    api_to_idx: dict[str, int],
) -> tuple[Any, Any]:
    x_train = encoder.encode_records(train_records, show_progress_bar=True)
    y_train = make_multihot(train_records, api_to_idx)
    classifier = build_classifier(args)
    classifier.fit(x_train, y_train)
    return x_train, classifier


def score_records(
    args: argparse.Namespace,
    encoder: SbertEncoder,
    classifier: Any,
    records: Sequence[MashupRecord],
) -> Any:
    import numpy as np

    x = encoder.encode_records(records, show_progress_bar=False)
    if args.model_type == "logistic":
        scores = classifier.predict_proba(x)
    else:
        scores = classifier.decision_function(x)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores


def tune_threshold(
    *,
    args: argparse.Namespace,
    encoder: SbertEncoder,
    train_records: Sequence[MashupRecord],
    valid_records: Sequence[MashupRecord],
    label_names: Sequence[str],
    api_to_idx: dict[str, int],
) -> tuple[float, list[dict[str, float]]]:
    import numpy as np

    _, classifier = fit_model(args, encoder, train_records, api_to_idx)
    valid_scores = score_records(args, encoder, classifier, valid_records)
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


def classifier_params(args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "C": args.C,
        "max_iter": args.max_iter if args.max_iter is not None else (1000 if args.model_type == "logistic" else 5000),
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "n_jobs": args.n_jobs,
    }
    if args.model_type == "logistic":
        params["solver"] = "liblinear"
    return params


def save_required_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    encoder: SbertEncoder,
    embedding_dim: int | None,
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
    encoder_config = encoder.config(embedding_dim)
    config = {
        "method": f"sbert_ovr_{args.model_type}",
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
        "encoder": encoder_config,
        "classifier_params": classifier_params(args),
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
    json_dump(output_dir / "encoder_config.json", encoder_config)
    if threshold_history:
        json_dump(output_dir / "threshold_tuning.json", threshold_history)

    joblib.dump(classifier, output_dir / "model.joblib")


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
            "method": f"sbert_ovr_{args.model_type}",
            "selection_mode": "threshold",
            "threshold": threshold,
            "score_type": "predict_proba" if args.model_type == "logistic" else "decision_function",
            "model_name": args.model_name,
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

    if args.threshold is None:
        args.threshold = 0.5 if args.model_type == "logistic" else 0.0

    train_records, valid_records, test_records, split_info = split_records(args)
    label_names = build_label_list(train_records)
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
    LOGGER.info("Loading Sentence-BERT encoder: %s", args.model_name)
    encoder = SbertEncoder(args)

    threshold_history: list[dict[str, float]] = []
    threshold = float(args.threshold)
    if args.tune_threshold:
        threshold, threshold_history = tune_threshold(
            args=args,
            encoder=encoder,
            train_records=train_records,
            valid_records=valid_records,
            label_names=label_names,
            api_to_idx=api_to_idx,
        )

    LOGGER.info("Fitting final %s model.", args.model_type)
    final_train_records = train_records + valid_records if args.tune_threshold else train_records
    train_embeddings, classifier = fit_model(args, encoder, final_train_records, api_to_idx)
    embedding_dim = int(train_embeddings.shape[1]) if train_embeddings.ndim == 2 else None
    test_scores = score_records(args, encoder, classifier, test_records)
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
            "embedding_dim": embedding_dim,
        }
    )

    save_required_outputs(
        output_dir=output_dir,
        args=args,
        encoder=encoder,
        embedding_dim=embedding_dim,
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
