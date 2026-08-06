#!/usr/bin/env python3
"""Sentence-BERT embedding + cosine similarity Top-k API recommendation baseline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from baseline_utils import (
    add_common_io_args,
    load_api_catalog,
    load_mashup_records,
    make_result_record,
    print_run_summary,
    recommendations_from_scores,
    resolved_eval_path,
    safe_name,
    save_outputs,
)


DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sentence-BERT + cosine Top-k API recommendation baseline.")
    add_common_io_args(parser)
    parser.add_argument("--model-name", default=DEFAULT_SBERT_MODEL, help="sentence-transformers model name/path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--device", default=None, help="Optional device passed to SentenceTransformer, e.g. cuda or cpu.")
    parser.add_argument("--api-embedding-cache", default=None, help="Optional .npz cache path for API embeddings.")
    parser.add_argument("--no-cache", action="store_true", help="Disable loading/saving API embedding cache.")
    return parser.parse_args()


def default_cache_path(output_dir: str, model_name: str, api_catalog_path: str) -> Path:
    digest = hashlib.sha1(str(Path(api_catalog_path).resolve()).encode("utf-8")).hexdigest()[:10]
    return Path(output_dir) / "cache" / f"api_embeddings_{safe_name(model_name)}_{digest}.npz"


def load_or_compute_api_embeddings(
    *,
    model: object,
    model_name: str,
    api_names: list[str],
    api_texts: list[str],
    cache_path: Path | None,
    batch_size: int,
) -> tuple[object, bool]:
    import numpy as np

    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_model = str(cached["model_name"][0])
        cached_names = [str(x) for x in cached["api_names"]]
        if cached_model == model_name and cached_names == api_names:
            return cached["embeddings"].astype("float32"), True

    embeddings = model.encode(
        api_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            model_name=np.asarray([model_name], dtype=str),
            api_names=np.asarray(api_names, dtype=str),
            embeddings=embeddings,
        )
    return embeddings, False


def run(args: argparse.Namespace) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc
    import numpy as np

    eval_path = resolved_eval_path(args)
    records = load_mashup_records(eval_path, limit=args.limit)
    api_names, api_texts = load_api_catalog(args.api_catalog_path)
    if not records:
        raise SystemExit(f"No evaluation records loaded from {eval_path}")
    if not api_names:
        raise SystemExit(f"No API candidates loaded from {args.api_catalog_path}")

    model = SentenceTransformer(args.model_name, device=args.device) if args.device else SentenceTransformer(args.model_name)
    cache_path = None
    if not args.no_cache:
        cache_path = Path(args.api_embedding_cache) if args.api_embedding_cache else default_cache_path(
            args.output_dir, args.model_name, args.api_catalog_path
        )
    api_embeddings, cache_hit = load_or_compute_api_embeddings(
        model=model,
        model_name=args.model_name,
        api_names=api_names,
        api_texts=api_texts,
        cache_path=cache_path,
        batch_size=args.batch_size,
    )

    result_records = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        query_embeddings = model.encode(
            [record.text for record in batch_records],
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        similarity = np.matmul(query_embeddings, api_embeddings.T)
        for record, scores in zip(batch_records, similarity):
            recommendations = recommendations_from_scores(
                api_names,
                scores,
                args.top_k,
                selection_mode=args.selection_mode,
                threshold=args.threshold,
            )
            result_records.append(make_result_record(record, recommendations))

    metadata = {
        "eval_path": str(Path(eval_path).resolve()),
        "api_catalog_path": str(Path(args.api_catalog_path).resolve()),
        "num_api_candidates": len(api_names),
        "score_type": "sbert_cosine_similarity",
        "model_name": args.model_name,
        "api_embedding_cache": str(cache_path.resolve()) if cache_path else None,
        "api_embedding_cache_hit": cache_hit,
    }
    json_path, csv_path = save_outputs(
        output_dir=args.output_dir,
        method="sbert",
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
