#!/usr/bin/env python3
"""TF-IDF + cosine similarity API recommendation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline_utils import (
    add_common_io_args,
    load_api_catalog,
    load_mashup_records,
    make_result_record,
    print_run_summary,
    recommendations_from_scores,
    resolved_eval_path,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF + cosine API recommendation baseline.")
    add_common_io_args(parser)
    parser.add_argument("--batch-size", type=int, default=64, help="Number of requirements scored at once.")
    parser.add_argument("--max-features", type=int, default=None, help="Optional max_features for TfidfVectorizer.")
    parser.add_argument("--min-df", type=int, default=1, help="min_df for TfidfVectorizer.")
    parser.add_argument("--max-ngram", type=int, default=1, help="Use ngrams from 1 to this value.")
    parser.add_argument(
        "--stop-words",
        choices=["none", "english"],
        default="none",
        help="Optional sklearn stop_words setting.",
    )
    parser.add_argument(
        "--threshold-strict",
        action="store_true",
        help="With --selection-mode threshold, recommend only APIs with score > threshold.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from exc

    eval_path = resolved_eval_path(args)
    records = load_mashup_records(eval_path, limit=args.limit)
    api_names, api_texts = load_api_catalog(args.api_catalog_path)
    if not records:
        raise SystemExit(f"No evaluation records loaded from {eval_path}")
    if not api_names:
        raise SystemExit(f"No API candidates loaded from {args.api_catalog_path}")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.max_ngram),
        stop_words=None if args.stop_words == "none" else args.stop_words,
    )
    api_matrix = vectorizer.fit_transform(api_texts)

    result_records = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        query_matrix = vectorizer.transform([record.text for record in batch_records])
        similarity = cosine_similarity(query_matrix, api_matrix)
        for record, scores in zip(batch_records, similarity):
            recommendations = recommendations_from_scores(
                api_names,
                scores,
                args.top_k,
                selection_mode=args.selection_mode,
                threshold=args.threshold,
                threshold_strict=args.threshold_strict,
            )
            result_records.append(make_result_record(record, recommendations))

    metadata = {
        "eval_path": str(Path(eval_path).resolve()),
        "api_catalog_path": str(Path(args.api_catalog_path).resolve()),
        "num_api_candidates": len(api_names),
        "score_type": "tfidf_cosine_similarity",
        "vectorizer": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "ngram_range": [1, args.max_ngram],
            "stop_words": args.stop_words,
        },
        "threshold_strict": bool(args.threshold_strict),
    }
    json_path, csv_path = save_outputs(
        output_dir=args.output_dir,
        method="tfidf",
        top_k=args.top_k if args.selection_mode == "topk" else None,
        result_records=result_records,
        metadata=metadata,
        run_name=args.run_name,
        selection_mode=args.selection_mode,
        threshold=args.threshold if args.selection_mode == "threshold" else None,
    )
    print_run_summary(json_path, csv_path, result_records)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
