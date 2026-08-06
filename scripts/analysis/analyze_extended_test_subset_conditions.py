#!/usr/bin/env python3
"""Extended subset-condition analysis for the 336 API-recommendation test samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_test_subset_conditions as base

NA = base.NA
RANDOM_SEED = 42
NEAR_DUPLICATE_THRESHOLD = 0.90
LARGE_CATEGORY_CANDIDATE_THRESHOLD = 50


class TfidfLite:
    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit_transform(self, documents: list[str]) -> list[dict[str, float]]:
        tokenized = [self.tokenize(doc) for doc in documents]
        doc_count = len(tokenized)
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        self.idf = {token: math.log((1 + doc_count) / (1 + count)) + 1 for token, count in df.items()}
        return [self._transform_tokens(tokens) for tokens in tokenized]

    def _transform_tokens(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        values = {token: (count / total) * self.idf.get(token, 0.0) for token, count in tf.items()}
        norm = math.sqrt(sum(value * value for value in values.values()))
        if norm <= 0:
            return {}
        return {token: value / norm for token, value in values.items()}


def cosine_sparse(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(token, 0.0) for token, value in a.items())


def backup_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)


def parse_json_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def average_or_na(values: list[int | float]) -> str:
    if not values:
        return NA
    return f"{mean(values):.3f}"


def min_or_na(values: list[int | float]) -> int | float | str:
    return min(values) if values else NA


def max_or_na(values: list[int | float]) -> int | float | str:
    return max(values) if values else NA


def sum_or_na(values: list[int | float]) -> int | float | str:
    return sum(values) if values else NA


def ratio(count: int, total: int) -> str:
    return base.ratio(count, total)


def load_api_descriptions(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = base.load_json(path)
    if not isinstance(data, dict):
        return {}
    descriptions: dict[str, str] = {}
    for api, entry in data.items():
        description = ""
        if isinstance(entry, dict):
            description = str(entry.get("description", entry.get("text", "")) or "")
        elif isinstance(entry, list) and len(entry) > 1:
            description = str(entry[1] or "")
        if description.strip():
            descriptions[str(api)] = description
    return descriptions


def api_category_members(api_to_category: Mapping[str, str]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for api, category in api_to_category.items():
        if category:
            members[str(category)].append(str(api))
    return {category: sorted(apis) for category, apis in members.items()}


def category_candidate_bin(value: int | str) -> str:
    if value == NA:
        return NA
    numeric = int(value)
    if numeric <= 20:
        return "small_candidate"
    if numeric <= 100:
        return "medium_candidate"
    return "large_candidate"


def candidate_quantile_bin(value: int | float | str, q25: float, q75: float) -> str:
    if value == NA:
        return NA
    bucket = base.quantile_bucket(float(value), q25, q75)
    if bucket == "low":
        return "low_candidate_size"
    if bucket == "middle":
        return "middle_candidate_size"
    return "high_candidate_size"


def enrich_oracle_category_candidates(
    sample_rows: list[dict[str, Any]],
    api_to_category: Mapping[str, str],
) -> tuple[dict[str, list[str]], tuple[float, float]]:
    category_to_apis = api_category_members(api_to_category)
    sums: list[int] = []
    for row in sample_rows:
        categories = parse_json_cell(row["ground_truth_categories"])
        sizes = [len(category_to_apis.get(category, [])) for category in categories]
        row["min_oracle_category_candidate_size"] = min_or_na(sizes)
        row["mean_oracle_category_candidate_size"] = average_or_na(sizes)
        row["max_oracle_category_candidate_size"] = max_or_na(sizes)
        row["sum_oracle_category_candidate_size"] = sum_or_na(sizes)
        row["oracle_category_candidate_size_bin"] = category_candidate_bin(row["sum_oracle_category_candidate_size"])
        row["large_candidate_category_including"] = (
            "yes"
            if sizes and max(sizes) >= LARGE_CATEGORY_CANDIDATE_THRESHOLD
            else "no"
        )
        if sizes:
            sums.append(sum(sizes))
    q25 = base.percentile([float(value) for value in sums], 0.25) if sums else 0.0
    q75 = base.percentile([float(value) for value in sums], 0.75) if sums else 0.0
    for row in sample_rows:
        row["oracle_category_candidate_size_quantile_bin"] = candidate_quantile_bin(
            row["sum_oracle_category_candidate_size"], q25, q75
        )
    return category_to_apis, (q25, q75)


def branch_candidate_maps(
    hierarchy: base.HierarchySpec,
    api_to_category: Mapping[str, str],
) -> dict[str, dict[str, set[str]]]:
    maps = {"layer1": defaultdict(set), "layer2": defaultdict(set)}
    for api, category in api_to_category.items():
        path = hierarchy.category_paths.get(category)
        if not path:
            continue
        if len(path) >= 1:
            maps["layer1"][str(path[0])].add(api)
        if len(path) >= 2:
            maps["layer2"][f"{path[0]}-{path[1]}"].add(api)
    return {
        layer: {branch: set(apis) for branch, apis in layer_map.items()}
        for layer, layer_map in maps.items()
    }


def branch_keys_for_categories(categories: list[str], hierarchy: base.HierarchySpec, layer: str) -> set[str]:
    keys: set[str] = set()
    for category in categories:
        path = hierarchy.category_paths.get(category)
        if not path:
            continue
        if layer == "layer1" and len(path) >= 1:
            keys.add(str(path[0]))
        elif layer == "layer2" and len(path) >= 2:
            keys.add(f"{path[0]}-{path[1]}")
    return keys


def enrich_oracle_branch_candidates(
    sample_rows: list[dict[str, Any]],
    hierarchies: list[base.HierarchySpec],
    api_to_category: Mapping[str, str],
) -> list[str]:
    added_fields: list[str] = []
    for hierarchy in hierarchies:
        maps = branch_candidate_maps(hierarchy, api_to_category)
        for layer in ("layer1", "layer2"):
            for stat in ("sum", "mean", "max"):
                field = f"{hierarchy.label}_{layer}_{stat}_oracle_branch_candidate_size"
                added_fields.append(field)
        for row in sample_rows:
            categories = parse_json_cell(row["ground_truth_categories"])
            for layer in ("layer1", "layer2"):
                keys = branch_keys_for_categories(categories, hierarchy, layer)
                sizes = [len(maps[layer].get(key, set())) for key in keys]
                row[f"{hierarchy.label}_{layer}_sum_oracle_branch_candidate_size"] = sum_or_na(sizes)
                row[f"{hierarchy.label}_{layer}_mean_oracle_branch_candidate_size"] = average_or_na(sizes)
                row[f"{hierarchy.label}_{layer}_max_oracle_branch_candidate_size"] = max_or_na(sizes)
    return added_fields


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def max_jaccard(target: set[str], candidates: list[set[str]]) -> float:
    if not candidates:
        return 0.0
    best = 0.0
    for candidate in candidates:
        score = jaccard(target, candidate)
        if score > best:
            best = score
            if best == 1.0:
                break
    return best


def combination_status(prefix: str, score: float | str) -> str:
    if score == NA:
        return NA
    numeric = float(score)
    if numeric >= 1.0:
        return f"{prefix}_combination_exact_seen"
    if numeric >= 0.5:
        return f"{prefix}_combination_similar"
    return f"{prefix}_combination_novel"


def enrich_combination_novelty(
    sample_rows: list[dict[str, Any]],
    train_records: list[base.MashupRecord],
    api_to_category: Mapping[str, str],
) -> bool:
    if not train_records:
        for row in sample_rows:
            row["max_train_category_set_jaccard"] = NA
            row["category_combination_status"] = NA
            row["max_train_api_set_jaccard"] = NA
            row["api_combination_status"] = NA
        return False

    train_category_sets: list[set[str]] = []
    train_api_sets: list[set[str]] = []
    for record in train_records:
        categories, _ = base.categories_for_record(record, api_to_category)
        train_category_sets.append(set(categories))
        train_api_sets.append(set(record.ground_truth_apis))

    for row in sample_rows:
        categories = set(parse_json_cell(row["ground_truth_categories"]))
        apis = set(parse_json_cell(row["ground_truth_apis"]))
        category_score = max_jaccard(categories, train_category_sets)
        api_score = max_jaccard(apis, train_api_sets)
        row["max_train_category_set_jaccard"] = f"{category_score:.3f}"
        row["category_combination_status"] = combination_status("category", category_score)
        row["max_train_api_set_jaccard"] = f"{api_score:.3f}"
        row["api_combination_status"] = combination_status("api", api_score)
    return True


def enrich_requirement_similarity(
    sample_rows: list[dict[str, Any]],
    api_descriptions: Mapping[str, str],
) -> tuple[bool, tuple[float, float], tuple[float, float]]:
    available_apis = sorted(api_descriptions)
    if not available_apis:
        for row in sample_rows:
            for field in [
                "mean_similarity_to_gt_apis",
                "max_similarity_to_gt_apis",
                "mean_similarity_to_non_gt_apis",
                "max_similarity_to_non_gt_apis",
                "similarity_margin_mean",
                "similarity_margin_max",
                "requirement_clarity_status",
                "requirement_clarity_mean_status",
            ]:
                row[field] = NA
        return False, (0.0, 0.0), (0.0, 0.0)

    sample_docs = [str(row["description"]) for row in sample_rows]
    api_docs = [api_descriptions[api] for api in available_apis]
    vectors = TfidfLite().fit_transform(sample_docs + api_docs)
    sample_vectors = vectors[: len(sample_docs)]
    api_vectors = dict(zip(available_apis, vectors[len(sample_docs) :]))

    for row, sample_vector in zip(sample_rows, sample_vectors):
        gt_apis = set(parse_json_cell(row["ground_truth_apis"]))
        gt_scores = [cosine_sparse(sample_vector, api_vectors[api]) for api in gt_apis if api in api_vectors]
        non_gt_scores = [
            cosine_sparse(sample_vector, api_vectors[api])
            for api in available_apis
            if api not in gt_apis
        ]
        mean_gt = mean(gt_scores) if gt_scores else 0.0
        max_gt = max(gt_scores) if gt_scores else 0.0
        mean_non = mean(non_gt_scores) if non_gt_scores else 0.0
        max_non = max(non_gt_scores) if non_gt_scores else 0.0
        row["mean_similarity_to_gt_apis"] = f"{mean_gt:.6f}"
        row["max_similarity_to_gt_apis"] = f"{max_gt:.6f}"
        row["mean_similarity_to_non_gt_apis"] = f"{mean_non:.6f}"
        row["max_similarity_to_non_gt_apis"] = f"{max_non:.6f}"
        row["similarity_margin_mean"] = f"{mean_gt - mean_non:.6f}"
        row["similarity_margin_max"] = f"{max_gt - max_non:.6f}"

    mean_margins = [float(row["similarity_margin_mean"]) for row in sample_rows]
    max_margins = [float(row["similarity_margin_max"]) for row in sample_rows]
    mean_q25 = base.percentile(mean_margins, 0.25)
    mean_q75 = base.percentile(mean_margins, 0.75)
    max_q25 = base.percentile(max_margins, 0.25)
    max_q75 = base.percentile(max_margins, 0.75)
    for row in sample_rows:
        row["requirement_clarity_status"] = requirement_clarity_status(float(row["similarity_margin_max"]), max_q25, max_q75)
        row["requirement_clarity_mean_status"] = requirement_clarity_status(float(row["similarity_margin_mean"]), mean_q25, mean_q75)
    return True, (max_q25, max_q75), (mean_q25, mean_q75)


def requirement_clarity_status(value: float, q25: float, q75: float) -> str:
    if value <= q25:
        return "Ambiguous-requirement"
    if value <= q75:
        return "Middle-requirement"
    return "Clear-requirement"


def enrich_duplicate_descriptions(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptions = [str(row["description"]).strip() for row in sample_rows]
    exact_counts = Counter(descriptions)
    vectors = TfidfLite().fit_transform(descriptions)
    duplicate_pairs: list[dict[str, Any]] = []

    max_scores = [0.0 for _ in sample_rows]
    for i in range(len(sample_rows)):
        for j in range(i + 1, len(sample_rows)):
            score = cosine_sparse(vectors[i], vectors[j])
            if score > max_scores[i]:
                max_scores[i] = score
            if score > max_scores[j]:
                max_scores[j] = score
            if score >= NEAR_DUPLICATE_THRESHOLD:
                duplicate_pairs.append(
                    {
                        "sample_id_1": sample_rows[i]["sample_id"],
                        "sample_id_2": sample_rows[j]["sample_id"],
                        "description_similarity": f"{score:.6f}",
                        "exact_duplicate": "yes" if descriptions[i] == descriptions[j] else "no",
                    }
                )

    for row, description, score in zip(sample_rows, descriptions, max_scores):
        exact_duplicate = exact_counts[description] > 1 and description != ""
        row["max_description_similarity_to_other_test_sample"] = f"{score:.6f}"
        if exact_duplicate:
            row["duplicate_description_status"] = "Exact-duplicate-description"
        elif score >= NEAR_DUPLICATE_THRESHOLD:
            row["duplicate_description_status"] = "Near-duplicate-description"
        else:
            row["duplicate_description_status"] = "Duplicate-free-description"
    return duplicate_pairs


def append_extended_summary(
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    has_train: bool,
    has_api_similarity: bool,
) -> None:
    total = len(sample_rows)

    def add(name: str, definition: str, predicate: Any, notes: str = "") -> None:
        if predicate is None:
            count: int | str = NA
        else:
            count = sum(1 for row in sample_rows if predicate(row))
        base.add_summary_row(summary_rows, total, name, definition, count, notes)

    add("Small-oracle-candidate", "sum_oracle_category_candidate_size between 1 and 20", lambda r: r["oracle_category_candidate_size_bin"] == "small_candidate")
    add("Medium-oracle-candidate", "sum_oracle_category_candidate_size between 21 and 100", lambda r: r["oracle_category_candidate_size_bin"] == "medium_candidate")
    add("Large-oracle-candidate", "sum_oracle_category_candidate_size >= 101", lambda r: r["oracle_category_candidate_size_bin"] == "large_candidate")
    add("Large-candidate-category-including", f"max_oracle_category_candidate_size >= {LARGE_CATEGORY_CANDIDATE_THRESHOLD}", lambda r: r["large_candidate_category_including"] == "yes")
    for bucket in ["low_candidate_size", "middle_candidate_size", "high_candidate_size"]:
        add(f"{bucket} (quantile)", f"oracle_category_candidate_size_quantile_bin == {bucket}", lambda r, b=bucket: r["oracle_category_candidate_size_quantile_bin"] == b)

    if has_train:
        add("Seen-category-combination", "test category set exactly appears in train", lambda r: r["max_train_category_set_jaccard"] == "1.000")
        add("Novel-category-combination", "test category set does not exactly appear in train", lambda r: r["max_train_category_set_jaccard"] != "1.000")
        for status in ["category_combination_exact_seen", "category_combination_similar", "category_combination_novel"]:
            add(status, f"category_combination_status == {status}", lambda r, s=status: r["category_combination_status"] == s)
        add("Seen-API-combination", "test API set exactly appears in train", lambda r: r["max_train_api_set_jaccard"] == "1.000")
        add("Novel-API-combination", "test API set does not exactly appear in train", lambda r: r["max_train_api_set_jaccard"] != "1.000")
        for status in ["api_combination_exact_seen", "api_combination_similar", "api_combination_novel"]:
            add(status, f"api_combination_status == {status}", lambda r, s=status: r["api_combination_status"] == s)
    else:
        for name in [
            "Seen-category-combination",
            "Novel-category-combination",
            "category_combination_exact_seen",
            "category_combination_similar",
            "category_combination_novel",
            "Seen-API-combination",
            "Novel-API-combination",
            "api_combination_exact_seen",
            "api_combination_similar",
            "api_combination_novel",
        ]:
            add(name, "train data unavailable", None, "Skipped because train data was not available.")

    if has_api_similarity:
        for status in ["Clear-requirement", "Middle-requirement", "Ambiguous-requirement"]:
            add(status, f"requirement_clarity_status == {status}", lambda r, s=status: r["requirement_clarity_status"] == s)
    else:
        for status in ["Clear-requirement", "Middle-requirement", "Ambiguous-requirement"]:
            add(status, "API descriptions unavailable", None, "Skipped because API descriptions were not available.")

    add("Exact-duplicate-description", "duplicate_description_status == Exact-duplicate-description", lambda r: r["duplicate_description_status"] == "Exact-duplicate-description")
    add("Near-duplicate-description", "duplicate_description_status == Near-duplicate-description", lambda r: r["duplicate_description_status"] == "Near-duplicate-description")
    add("Duplicate-free-description", "duplicate_description_status == Duplicate-free-description", lambda r: r["duplicate_description_status"] == "Duplicate-free-description")


def distribution_rows(sample_rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key, NA)) for row in sample_rows)
    total = len(sample_rows)
    return [
        {"metric": key, "bucket": bucket, "count": count, "ratio": ratio(count, total)}
        for bucket, count in sorted(counts.items())
    ]


def representative_subset_by_keys(sample_rows: list[dict[str, Any]], size: int, keys: list[str]) -> list[dict[str, Any]]:
    if size >= len(sample_rows):
        return list(sample_rows)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        groups[tuple(str(row.get(key, NA)) for key in keys)].append(row)

    allocation: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, tuple[str, ...]]] = []
    for key, rows in groups.items():
        exact = len(rows) * size / len(sample_rows)
        base_count = min(int(math.floor(exact)), len(rows))
        allocation[key] = base_count
        remainders.append((exact - base_count, key))

    selected: list[dict[str, Any]] = []
    for key, rows in groups.items():
        selected.extend(rows[: allocation[key]])

    leftover = size - len(selected)
    while leftover > 0:
        changed = False
        for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if leftover <= 0:
                break
            if allocation[key] < len(groups[key]):
                selected.append(groups[key][allocation[key]])
                allocation[key] += 1
                leftover -= 1
                changed = True
        if not changed:
            break

    if len(selected) < size:
        selected_ids = {row["sample_id"] for row in selected}
        for row in sample_rows:
            if row["sample_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["sample_id"])
                if len(selected) == size:
                    break

    selected.sort(key=lambda row: int(row.get("_original_index", 0)))
    return selected[:size]


def write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['sample_id']}\n")


def subset_feasibility_rows(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    sizes: list[int],
) -> list[dict[str, Any]]:
    candidate_dir = output_dir / "subset_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "representative_with_length": [
            "num_gt_apis_bin",
            "num_gt_categories_bin",
            "api_frequency_status",
            "single_or_multi_category",
            "word_count_bin",
        ],
        "representative_extended": [
            "num_gt_apis_bin",
            "num_gt_categories_bin",
            "api_frequency_status",
            "single_or_multi_category",
            "word_count_bin",
            "oracle_category_candidate_size_bin",
            "category_combination_status",
        ],
    }
    delta_keys = {
        "max_delta_num_gt_apis": "num_gt_apis_bin",
        "max_delta_num_gt_categories": "num_gt_categories_bin",
        "max_delta_api_frequency_status": "api_frequency_status",
        "max_delta_single_or_multi_category": "single_or_multi_category",
        "max_delta_word_count_bin": "word_count_bin",
        "max_delta_oracle_candidate_size_bin": "oracle_category_candidate_size_bin",
        "max_delta_category_combination_status": "category_combination_status",
        "max_delta_api_combination_status": "api_combination_status",
        "max_delta_requirement_clarity_status": "requirement_clarity_status",
    }
    rows: list[dict[str, Any]] = []
    total = len(sample_rows)
    for method, keys in methods.items():
        for size in sizes:
            if size > total:
                rows.append({"method": method, "size": size, "feasible": "no", "available": total, "candidate_file": NA})
                continue
            subset = representative_subset_by_keys(sample_rows, size, keys)
            path = candidate_dir / f"{method}_{size}_ids.txt"
            write_ids(path, subset)
            row = {
                "method": method,
                "size": size,
                "feasible": "yes",
                "available": total,
                "candidate_file": str(path.relative_to(output_dir)),
            }
            for output_key, source_key in delta_keys.items():
                row[output_key] = f"{base.max_ratio_delta(sample_rows, subset, source_key):.3f}"
            rows.append(row)
    return rows


def write_diagnostic_id_lists(output_dir: Path, sample_rows: list[dict[str, Any]]) -> None:
    candidate_dir = output_dir / "subset_candidates"
    diagnostics = {
        "large_oracle_candidate_ids.txt": lambda r: r["oracle_category_candidate_size_bin"] == "large_candidate",
        "novel_category_combination_ids.txt": lambda r: r["category_combination_status"] == "category_combination_novel",
        "novel_api_combination_ids.txt": lambda r: r["api_combination_status"] == "api_combination_novel",
        "ambiguous_requirement_ids.txt": lambda r: r["requirement_clarity_status"] == "Ambiguous-requirement",
        "near_duplicate_description_ids.txt": lambda r: r["duplicate_description_status"] in {"Exact-duplicate-description", "Near-duplicate-description"},
    }
    for filename, predicate in diagnostics.items():
        write_ids(candidate_dir / filename, [row for row in sample_rows if predicate(row)])


def build_extended_report(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    feasibility_rows: list[dict[str, Any]],
    has_train: bool,
    has_api_similarity: bool,
    has_branch: bool,
    candidate_quantiles: tuple[float, float],
    clarity_quantiles: tuple[float, float],
    duplicate_pair_count: int,
) -> str:
    total = len(sample_rows)
    wanted = {
        "Description short",
        "Description medium",
        "Description long",
        "Small-oracle-candidate",
        "Medium-oracle-candidate",
        "Large-oracle-candidate",
        "Large-candidate-category-including",
        "Seen-category-combination",
        "Novel-category-combination",
        "Seen-API-combination",
        "Novel-API-combination",
        "Clear-requirement",
        "Middle-requirement",
        "Ambiguous-requirement",
        "Exact-duplicate-description",
        "Near-duplicate-description",
        "Duplicate-free-description",
    }
    selected_summary = [row for row in summary_rows if row["condition_name"] in wanted]
    length_feasibility = [row for row in feasibility_rows if row["method"] == "representative_with_length"]
    extended_feasibility = [row for row in feasibility_rows if row["method"] == "representative_extended"]

    large_count = sum(1 for row in sample_rows if row["oracle_category_candidate_size_bin"] == "large_candidate")
    novel_cat = sum(1 for row in sample_rows if row["category_combination_status"] == "category_combination_novel")
    novel_api = sum(1 for row in sample_rows if row["api_combination_status"] == "api_combination_novel")
    ambiguous = sum(1 for row in sample_rows if row["requirement_clarity_status"] == "Ambiguous-requirement")

    lines = [
        "# Extended Test Dataset Subset Condition Analysis",
        "",
        "## Overview",
        f"- total samples: {total}",
        "- previous conditions are preserved in summary_counts.csv.",
        "- newly added conditions cover description length stratification, oracle candidate size, combination novelty, TF-IDF requirement clarity, and duplicate descriptions.",
        "",
        "## Newly Added Conditions Summary",
        base.markdown_table(selected_summary, ["condition_name", "condition_definition", "count", "ratio", "notes"]),
        "",
        "## Description Length",
        "- word_count_bin is now included in representative_with_length stratification.",
        base.markdown_table(length_feasibility, ["method", "size", "feasible", "candidate_file", "max_delta_word_count_bin"]),
        "",
        "## Oracle Candidate Size",
        f"- category candidate-size quantiles: q25={candidate_quantiles[0]:.3f}, q75={candidate_quantiles[1]:.3f}",
        f"- large oracle candidate samples: {large_count} ({ratio(large_count, total)})",
        "- Large candidate sets are useful as an explanatory axis for E-F1 analysis because they approximate API-layer ambiguity after a correct category or branch decision.",
        "",
        "## Combination Novelty",
        f"- train-based novelty analysis: {'enabled' if has_train else 'skipped'}",
        f"- category_combination_novel samples: {novel_cat} ({ratio(novel_cat, total)})",
        f"- api_combination_novel samples: {novel_api} ({ratio(novel_api, total)})",
        "",
        "## Requirement Clarity",
        f"- API-description TF-IDF analysis: {'enabled' if has_api_similarity else 'skipped'}",
        f"- similarity_margin_max quantiles: q25={clarity_quantiles[0]:.6f}, q75={clarity_quantiles[1]:.6f}",
        f"- ambiguous requirement samples: {ambiguous} ({ratio(ambiguous, total)})",
        "",
        "## Duplicate Analysis",
        f"- near-duplicate threshold: {NEAR_DUPLICATE_THRESHOLD}",
        f"- duplicate or near-duplicate pairs: {duplicate_pair_count}",
        "",
        "## Updated Subset Feasibility",
        "### representative_with_length",
        base.markdown_table(length_feasibility, ["method", "size", "feasible", "candidate_file", "max_delta_word_count_bin", "max_delta_oracle_candidate_size_bin", "max_delta_category_combination_status", "max_delta_api_combination_status", "max_delta_requirement_clarity_status"]),
        "",
        "### representative_extended",
        base.markdown_table(extended_feasibility, ["method", "size", "feasible", "candidate_file", "max_delta_word_count_bin", "max_delta_oracle_candidate_size_bin", "max_delta_category_combination_status", "max_delta_api_combination_status", "max_delta_requirement_clarity_status"]),
        "",
        "## Recommended Subset Strategy",
        "- Main experiment: representative_with_length is the safest default because it preserves the original stratification and adds description length.",
        "- Richer diagnostic experiment: representative_extended adds oracle candidate size and category-combination novelty when the target size remains stable.",
        "- Auxiliary diagnostics: use large_oracle_candidate_ids.txt, novel_category_combination_ids.txt, novel_api_combination_ids.txt, ambiguous_requirement_ids.txt, and near_duplicate_description_ids.txt for focused checks.",
        "- Very small condition groups should be treated as diagnostic slices rather than balanced main subsets.",
        "",
        "## Notes",
        f"- branch candidate-size analysis: {'enabled' if has_branch else 'skipped'}",
        f"- output directory: {output_dir}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended condition analysis for the 336 test samples.")
    parser.add_argument("--test-path", default=str(base.DEFAULT_TEST_PATH))
    parser.add_argument("--api-category-path", default=str(base.DEFAULT_API_CATEGORY_PATH))
    parser.add_argument("--api-catalog-path", default=str(base.DEFAULT_API_CATALOG_PATH))
    parser.add_argument("--train-path", default=str(base.DEFAULT_TRAIN_PATH), help="Use empty string to skip train-based novelty.")
    parser.add_argument("--hierarchy-paths", nargs="*", default=None)
    parser.add_argument("--hierarchy-glob", nargs="*", default=None)
    parser.add_argument("--output-dir", default=str(base.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-sizes", default="50,100,150,200,250,336")
    parser.add_argument("--expected-test-size", type=int, default=336)
    parser.add_argument("--no-backup", action="store_true", help="Do not back up CSV files before overwriting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test_path)
    api_category_path = Path(args.api_category_path) if args.api_category_path else None
    api_catalog_path = Path(args.api_catalog_path) if args.api_catalog_path else None
    train_path = Path(args.train_path) if args.train_path else None
    output_dir = Path(args.output_dir)

    records = base.load_mashup_records(test_path)
    if args.expected_test_size and len(records) != args.expected_test_size:
        print(f"[WARN] expected {args.expected_test_size} test samples, but loaded {len(records)}")

    api_to_category = base.load_api_to_category(api_category_path, api_catalog_path)
    if not api_to_category:
        raise ValueError("No API-to-category metadata could be loaded.")

    has_train = train_path is not None and train_path.exists()
    train_records = base.load_mashup_records(train_path) if has_train else []
    train_api_count = base.count_apis(train_records) if has_train else None
    train_category_count = base.count_categories(train_records, api_to_category) if has_train else None

    test_api_count = base.count_apis(records)
    test_category_count = base.count_categories(records, api_to_category)
    api_universe = sorted(set(api_to_category) | set(test_api_count) | set(train_api_count or {}))
    if train_api_count is not None:
        api_frequency_values = [int(train_api_count.get(api, 0)) for api in api_universe]
        api_q25 = base.percentile(api_frequency_values, 0.25)
        api_q75 = base.percentile(api_frequency_values, 0.75)
    else:
        api_q25 = 0.0
        api_q75 = 0.0

    if train_category_count:
        source_category_count = train_category_count
        category_frequency_source = "train"
    else:
        source_category_count = test_category_count
        category_frequency_source = "test"
    category_universe = sorted(set(source_category_count) | set(test_category_count) | set(api_to_category.values()))
    category_frequency_values = [int(source_category_count.get(category, 0)) for category in category_universe]
    category_q25 = base.percentile(category_frequency_values, 0.25)
    category_q75 = base.percentile(category_frequency_values, 0.75)

    hierarchy_paths = base.resolve_hierarchy_paths(args)
    hierarchies = [hierarchy for path in hierarchy_paths if (hierarchy := base.load_hierarchy(path))]
    primary = base.primary_hierarchy(hierarchies)

    sample_rows, missing_api_category_counter = base.build_sample_rows(
        records=records,
        api_to_category=api_to_category,
        train_api_count=train_api_count,
        api_q25=api_q25,
        api_q75=api_q75,
        category_count=source_category_count,
        category_q25=category_q25,
        category_q75=category_q75,
        primary=primary,
    )
    for row, record in zip(sample_rows, records):
        row["_original_index"] = record.index
    base.attach_internal_frequency_lists(sample_rows, records, train_api_count)

    branch_by_config = base.build_branch_rows(sample_rows, hierarchies) if hierarchies else {}
    category_to_apis, candidate_quantiles = enrich_oracle_category_candidates(sample_rows, api_to_category)
    branch_candidate_fields = enrich_oracle_branch_candidates(sample_rows, hierarchies, api_to_category)
    has_novelty = enrich_combination_novelty(sample_rows, train_records, api_to_category)
    api_descriptions = load_api_descriptions(api_catalog_path)
    has_api_similarity, clarity_quantiles, mean_clarity_quantiles = enrich_requirement_similarity(sample_rows, api_descriptions)
    duplicate_pairs = enrich_duplicate_descriptions(sample_rows)

    summary_rows, composite_counts = base.build_summary_counts(
        sample_rows=sample_rows,
        has_train=has_train,
        has_branch=bool(hierarchies),
        branch_by_config=branch_by_config,
        primary_label=primary.label if primary else NA,
    )
    append_extended_summary(summary_rows, sample_rows, has_train=has_train, has_api_similarity=has_api_similarity)

    output_dir.mkdir(parents=True, exist_ok=True)
    files_to_overwrite = [
        output_dir / "sample_condition_flags.csv",
        output_dir / "summary_counts.csv",
        output_dir / "subset_feasibility.csv",
    ]
    if not args.no_backup:
        for path in files_to_overwrite:
            backup_existing(path)

    sample_fieldnames = [
        "sample_id",
        "description",
        "ground_truth_apis",
        "num_gt_apis",
        "ground_truth_categories",
        "num_gt_categories",
        "word_count",
        "word_count_bin",
        "min_api_train_freq",
        "mean_api_train_freq",
        "max_api_train_freq",
        "num_unseen_apis",
        "seen_status",
        "api_frequency_status",
        "api_frequency_quantile_status",
        "min_category_freq",
        "mean_category_freq",
        "max_category_freq",
        "category_frequency_status",
        "single_or_multi_category",
        "num_branches_layer1",
        "num_branches_layer2",
        "single_or_multi_branch_layer1",
        "single_or_multi_branch_layer2",
        "min_oracle_category_candidate_size",
        "mean_oracle_category_candidate_size",
        "max_oracle_category_candidate_size",
        "sum_oracle_category_candidate_size",
        "oracle_category_candidate_size_bin",
        "oracle_category_candidate_size_quantile_bin",
        "large_candidate_category_including",
        *branch_candidate_fields,
        "max_train_category_set_jaccard",
        "category_combination_status",
        "max_train_api_set_jaccard",
        "api_combination_status",
        "mean_similarity_to_gt_apis",
        "max_similarity_to_gt_apis",
        "mean_similarity_to_non_gt_apis",
        "max_similarity_to_non_gt_apis",
        "similarity_margin_mean",
        "similarity_margin_max",
        "requirement_clarity_status",
        "requirement_clarity_mean_status",
        "max_description_similarity_to_other_test_sample",
        "duplicate_description_status",
        "missing_api_categories",
        "missing_branch_categories",
        "num_gt_apis_bin",
        "num_gt_categories_bin",
    ]
    base.write_csv(output_dir / "sample_condition_flags.csv", sample_fieldnames, base.strip_internal_fields(sample_rows))
    base.write_csv(output_dir / "summary_counts.csv", ["condition_name", "condition_definition", "count", "ratio", "notes"], summary_rows)

    sizes = base.parse_sizes(args.candidate_sizes)
    feasibility_rows = subset_feasibility_rows(output_dir, sample_rows, sizes)
    feasibility_fieldnames = [
        "method",
        "size",
        "feasible",
        "available",
        "candidate_file",
        "max_delta_num_gt_apis",
        "max_delta_num_gt_categories",
        "max_delta_api_frequency_status",
        "max_delta_single_or_multi_category",
        "max_delta_word_count_bin",
        "max_delta_oracle_candidate_size_bin",
        "max_delta_category_combination_status",
        "max_delta_api_combination_status",
        "max_delta_requirement_clarity_status",
    ]
    base.write_csv(output_dir / "subset_feasibility.csv", feasibility_fieldnames, feasibility_rows)
    base.write_csv(output_dir / "subset_feasibility_extended.csv", feasibility_fieldnames, feasibility_rows)
    write_diagnostic_id_lists(output_dir, sample_rows)

    base.write_csv(
        output_dir / "oracle_candidate_size_distribution.csv",
        ["metric", "bucket", "count", "ratio"],
        distribution_rows(sample_rows, "oracle_category_candidate_size_bin")
        + distribution_rows(sample_rows, "oracle_category_candidate_size_quantile_bin")
        + distribution_rows(sample_rows, "large_candidate_category_including"),
    )
    base.write_csv(
        output_dir / "combination_novelty_distribution.csv",
        ["metric", "bucket", "count", "ratio"],
        distribution_rows(sample_rows, "category_combination_status")
        + distribution_rows(sample_rows, "api_combination_status"),
    )
    base.write_csv(
        output_dir / "requirement_clarity_distribution.csv",
        ["metric", "bucket", "count", "ratio"],
        distribution_rows(sample_rows, "requirement_clarity_status")
        + distribution_rows(sample_rows, "requirement_clarity_mean_status"),
    )
    base.write_csv(
        output_dir / "duplicate_description_pairs.csv",
        ["sample_id_1", "sample_id_2", "description_similarity", "exact_duplicate"],
        duplicate_pairs,
    )

    extended_report = build_extended_report(
        output_dir=output_dir,
        sample_rows=sample_rows,
        summary_rows=summary_rows,
        feasibility_rows=feasibility_rows,
        has_train=has_train,
        has_api_similarity=has_api_similarity,
        has_branch=bool(hierarchies),
        candidate_quantiles=candidate_quantiles,
        clarity_quantiles=clarity_quantiles,
        duplicate_pair_count=len(duplicate_pairs),
    )
    (output_dir / "extended_subset_condition_report.md").write_text(extended_report, encoding="utf-8")

    metadata_path = output_dir / "analysis_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata["extended_analysis"] = {
        "script": str(Path(__file__).resolve()),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "oracle_category_candidate_quantiles": {"q25": candidate_quantiles[0], "q75": candidate_quantiles[1]},
        "requirement_similarity_margin_max_quantiles": {"q25": clarity_quantiles[0], "q75": clarity_quantiles[1]} if has_api_similarity else None,
        "requirement_similarity_margin_mean_quantiles": {"q25": mean_clarity_quantiles[0], "q75": mean_clarity_quantiles[1]} if has_api_similarity else None,
        "api_descriptions_loaded": len(api_descriptions),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "large_candidate_category_threshold": LARGE_CATEGORY_CANDIDATE_THRESHOLD,
        "has_train_novelty": has_novelty,
        "hierarchy_labels": [hierarchy.label for hierarchy in hierarchies],
        "missing_api_category_counts": dict(missing_api_category_counter.most_common()),
        "category_frequency_source": category_frequency_source,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded test samples: {len(records)}")
    print(f"Extended sample flags: {output_dir / 'sample_condition_flags.csv'}")
    print(f"Extended summary counts: {output_dir / 'summary_counts.csv'}")
    print(f"Extended feasibility: {output_dir / 'subset_feasibility.csv'}")
    print(f"Extended report: {output_dir / 'extended_subset_condition_report.md'}")


if __name__ == "__main__":
    main()
