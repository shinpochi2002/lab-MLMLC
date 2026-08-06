#!/usr/bin/env python3
"""BERT cross-encoder fine-tuning and Top-k API recommendation baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from baseline_utils import (
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    add_common_io_args,
    load_api_catalog,
    load_mashup_records,
    make_result_record,
    print_run_summary,
    recommendations_from_scores,
    resolved_eval_path,
    safe_name,
    save_outputs,
    unique_preserve_order,
)


DEFAULT_CROSS_ENCODER_MODEL = "bert-base-uncased"
DEFAULT_SBERT_CANDIDATE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune/evaluate a BERT cross-encoder API recommendation baseline."
    )
    add_common_io_args(parser)
    parser.add_argument("--train", action="store_true", help="Fine-tune before evaluation.")
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="Training mashup JSON path.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional training-record limit for smoke tests.")
    parser.add_argument("--model-name", default=DEFAULT_CROSS_ENCODER_MODEL, help="Base Hugging Face model name/path.")
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "codex" / "baseline" / "models" / "cross_encoder_bert_base_uncased"),
        help="Directory to save/load the fine-tuned cross-encoder.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Inference pair batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument("--negative-ratio", type=int, default=4, help="Negative APIs sampled per positive API.")
    parser.add_argument("--max-train-pairs", type=int, default=None, help="Optional cap on generated train pairs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--candidate-method",
        choices=["tfidf", "sbert", "all"],
        default="tfidf",
        help="Candidate pruning before cross-encoder reranking.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=200,
        help="Number of candidates to rerank. Use 0 or a negative value to score all APIs.",
    )
    parser.add_argument("--candidate-max-features", type=int, default=None, help="TF-IDF candidate max_features.")
    parser.add_argument("--candidate-min-df", type=int, default=1, help="TF-IDF candidate min_df.")
    parser.add_argument(
        "--candidate-sbert-model-name",
        default=DEFAULT_SBERT_CANDIDATE_MODEL,
        help="SentenceTransformer model for SBERT candidate pruning.",
    )
    parser.add_argument("--candidate-device", default=None, help="Optional device for SBERT candidate model.")
    return parser.parse_args()


def choose_torch_device(requested_device: str | None) -> Any:
    import torch

    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_training_examples(
    *,
    records: list[Any],
    api_names: list[str],
    api_texts: list[str],
    negative_ratio: int,
    seed: int,
    max_pairs: int | None,
) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    api_text_by_name = dict(zip(api_names, api_texts))
    examples: list[tuple[str, str, float]] = []

    for record in records:
        positives = [api for api in unique_preserve_order(record.true_apis) if api in api_text_by_name]
        if not positives:
            continue
        positive_set = set(positives)
        for api_name in positives:
            examples.append((record.text, api_text_by_name[api_name], 1.0))
            if max_pairs and len(examples) >= max_pairs:
                return examples[:max_pairs]

        needed_negatives = max(0, negative_ratio) * len(positives)
        sampled_negatives: set[str] = set()
        attempts = 0
        max_attempts = needed_negatives * 50 + 100
        while len(sampled_negatives) < needed_negatives and attempts < max_attempts:
            attempts += 1
            candidate = rng.choice(api_names)
            if candidate not in positive_set and candidate not in sampled_negatives:
                sampled_negatives.add(candidate)
        for api_name in sampled_negatives:
            examples.append((record.text, api_text_by_name[api_name], 0.0))
            if max_pairs and len(examples) >= max_pairs:
                return examples[:max_pairs]

    rng.shuffle(examples)
    return examples


def train_cross_encoder(
    *,
    args: argparse.Namespace,
    api_names: list[str],
    api_texts: list[str],
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("torch and transformers are required. Install with: pip install torch transformers") from exc

    torch.manual_seed(args.seed)
    device = choose_torch_device(args.device)
    train_records = load_mashup_records(args.train_path, limit=args.train_limit)
    examples = build_training_examples(
        records=train_records,
        api_names=api_names,
        api_texts=api_texts,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        max_pairs=args.max_train_pairs,
    )
    if not examples:
        raise SystemExit("No cross-encoder training pairs were generated.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def collate(batch: list[tuple[str, str, float]]) -> dict[str, Any]:
        texts = [item[0] for item in batch]
        api_descriptions = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float, device=device)
        encoded = tokenizer(
            texts,
            api_descriptions,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        encoded["labels"] = labels
        return encoded

    dataloader = DataLoader(examples, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_batches = 0
        for batch in dataloader:
            labels = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch).logits.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        avg_loss = total_loss / total_batches if total_batches else 0.0
        print(f"[train] epoch {epoch + 1}/{args.epochs} loss={avg_loss:.6f} pairs={len(examples)}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"[train] saved fine-tuned model: {model_dir}")
    model.eval()
    return model, tokenizer, device


def load_cross_encoder_for_eval(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Install with: pip install transformers") from exc

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(
            f"No fine-tuned model found in {model_dir}. Run once with --train, or set --model-dir to an existing model."
        )
    device = choose_torch_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def top_candidate_indices(scores: Any, candidate_k: int, api_names: list[str]) -> list[int]:
    import numpy as np

    score_array = np.asarray(scores, dtype=float)
    if candidate_k <= 0 or candidate_k >= len(score_array):
        indices = np.arange(len(score_array))
    else:
        indices = np.argpartition(-score_array, candidate_k - 1)[:candidate_k]
    return [int(i) for i in sorted(indices, key=lambda i: (-float(score_array[i]), api_names[int(i)]))]


def build_all_candidate_indices(num_records: int, num_apis: int) -> list[list[int]]:
    all_indices = list(range(num_apis))
    return [all_indices for _ in range(num_records)]


def build_tfidf_candidate_indices(
    *,
    records: list[Any],
    api_names: list[str],
    api_texts: list[str],
    candidate_k: int,
    max_features: int | None,
    min_df: int,
) -> list[list[int]]:
    if candidate_k <= 0 or candidate_k >= len(api_names):
        return build_all_candidate_indices(len(records), len(api_names))
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for TF-IDF candidate pruning.") from exc

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    api_matrix = vectorizer.fit_transform(api_texts)
    query_matrix = vectorizer.transform([record.text for record in records])
    similarity = cosine_similarity(query_matrix, api_matrix)
    return [top_candidate_indices(scores, candidate_k, api_names) for scores in similarity]


def build_sbert_candidate_indices(
    *,
    records: list[Any],
    api_names: list[str],
    api_texts: list[str],
    candidate_k: int,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> list[list[int]]:
    if candidate_k <= 0 or candidate_k >= len(api_names):
        return build_all_candidate_indices(len(records), len(api_names))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers is required for SBERT candidate pruning.") from exc
    import numpy as np

    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    api_embeddings = model.encode(
        api_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    query_embeddings = model.encode(
        [record.text for record in records],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    similarity = np.matmul(query_embeddings, api_embeddings.T)
    return [top_candidate_indices(scores, candidate_k, api_names) for scores in similarity]


def build_candidate_indices(
    *,
    args: argparse.Namespace,
    records: list[Any],
    api_names: list[str],
    api_texts: list[str],
) -> list[list[int]]:
    if args.candidate_method == "all":
        return build_all_candidate_indices(len(records), len(api_names))
    if args.candidate_method == "tfidf":
        return build_tfidf_candidate_indices(
            records=records,
            api_names=api_names,
            api_texts=api_texts,
            candidate_k=args.candidate_k,
            max_features=args.candidate_max_features,
            min_df=args.candidate_min_df,
        )
    return build_sbert_candidate_indices(
        records=records,
        api_names=api_names,
        api_texts=api_texts,
        candidate_k=args.candidate_k,
        model_name=args.candidate_sbert_model_name,
        device=args.candidate_device,
        batch_size=args.eval_batch_size,
    )


def score_candidate_pairs(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    record: Any,
    candidate_indices: list[int],
    api_texts: list[str],
    batch_size: int,
    max_length: int,
) -> Any:
    import numpy as np
    import torch

    all_scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[start : start + batch_size]
            encoded = tokenizer(
                [record.text] * len(batch_indices),
                [api_texts[i] for i in batch_indices],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.view(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            all_scores.extend(float(score) for score in scores)
    return np.asarray(all_scores, dtype=float)


def run(args: argparse.Namespace) -> None:
    eval_path = resolved_eval_path(args)
    records = load_mashup_records(eval_path, limit=args.limit)
    api_names, api_texts = load_api_catalog(args.api_catalog_path)
    if not records:
        raise SystemExit(f"No evaluation records loaded from {eval_path}")
    if not api_names:
        raise SystemExit(f"No API candidates loaded from {args.api_catalog_path}")

    if args.train:
        model, tokenizer, device = train_cross_encoder(args=args, api_names=api_names, api_texts=api_texts)
    else:
        model, tokenizer, device = load_cross_encoder_for_eval(args)

    candidate_indices_by_record = build_candidate_indices(
        args=args,
        records=records,
        api_names=api_names,
        api_texts=api_texts,
    )

    result_records = []
    for i, record in enumerate(records):
        candidate_indices = candidate_indices_by_record[i]
        scores = score_candidate_pairs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            record=record,
            candidate_indices=candidate_indices,
            api_texts=api_texts,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
        )
        recommendations = recommendations_from_scores(
            api_names,
            scores,
            args.top_k,
            candidate_indices=candidate_indices,
            selection_mode=args.selection_mode,
            threshold=args.threshold,
        )
        result_records.append(make_result_record(record, recommendations))
        print(f"[eval] {i + 1}/{len(records)} candidates={len(candidate_indices)}")

    metadata = {
        "eval_path": str(Path(eval_path).resolve()),
        "train_path": str(Path(args.train_path).resolve()) if args.train else None,
        "api_catalog_path": str(Path(args.api_catalog_path).resolve()),
        "num_api_candidates": len(api_names),
        "score_type": "cross_encoder_sigmoid_probability",
        "model_name": args.model_name,
        "model_dir": str(Path(args.model_dir).resolve()),
        "trained_this_run": bool(args.train),
        "negative_ratio": args.negative_ratio,
        "epochs": args.epochs if args.train else None,
        "candidate_method": args.candidate_method,
        "candidate_k": args.candidate_k,
    }
    method = f"cross_encoder_{safe_name(args.candidate_method)}"
    json_path, csv_path = save_outputs(
        output_dir=args.output_dir,
        method=method,
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
