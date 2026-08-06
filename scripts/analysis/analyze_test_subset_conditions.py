#!/usr/bin/env python3
"""Analyze test subset condition counts for API recommendation data.

The script is intentionally dependency-free so it can run in the same
environment as the existing data preparation scripts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_PATH = (
    PROJECT_ROOT
    / "codex"
    / "output"
    / "layer1_direct_api"
    / "test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json"
)
DEFAULT_API_CATEGORY_PATH = PROJECT_ROOT / "api_category_pairs.json"
DEFAULT_API_CATALOG_PATH = PROJECT_ROOT / "api_dataset_like_manifest.json"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "study_data_mashup_apis.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "codex" / "analysis_test_subset_conditions"
NA = "NA"


@dataclass(frozen=True)
class MashupRecord:
    index: int
    sample_id: str
    numeric_id: str
    description: str
    ground_truth_apis: list[str]
    entry_categories: list[str]


@dataclass(frozen=True)
class HierarchySpec:
    label: str
    path: Path
    category_paths: dict[str, list[int]]


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return unique_preserve_order(value)
    if isinstance(value, tuple):
        return unique_preserve_order(value)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_mashup_entry(name: str, entry: Any, index: int) -> MashupRecord:
    if isinstance(entry, dict):
        description = str(entry.get("description", entry.get("text", "")) or "")
        apis = as_string_list(
            entry.get("apis", entry.get("ground_truth_apis", entry.get("true_apis", [])))
        )
        categories = as_string_list(
            entry.get("categories", entry.get("ground_truth_categories", entry.get("gt_layer3", [])))
        )
        numeric_id = str(entry.get("id", entry.get("mashup_id", name)) or name)
    elif isinstance(entry, list):
        categories = as_string_list(entry[0]) if len(entry) > 0 else []
        description = str(entry[1]) if len(entry) > 1 and entry[1] is not None else ""
        apis = as_string_list(entry[2]) if len(entry) > 2 else []
        numeric_id = str(entry[3]) if len(entry) > 3 and entry[3] is not None else name
    else:
        description = ""
        apis = []
        categories = []
        numeric_id = name

    return MashupRecord(
        index=index,
        sample_id=str(name),
        numeric_id=numeric_id,
        description=description,
        ground_truth_apis=apis,
        entry_categories=categories,
    )


def load_mashup_records(path: Path, limit: int | None = None) -> list[MashupRecord]:
    data = load_json(path)
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(str(i), entry) for i, entry in enumerate(data)]
    else:
        raise ValueError(f"Unsupported mashup JSON root type in {path}: {type(data).__name__}")

    records: list[MashupRecord] = []
    for index, (name, entry) in enumerate(items):
        if limit is not None and index >= limit:
            break
        records.append(parse_mashup_entry(str(name), entry, index))
    return records


def extract_first_category(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("category", "first_category"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        categories = value.get("categories")
        if isinstance(categories, list) and categories:
            return str(categories[0]).strip()
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, list) and first:
            return str(first[0]).strip()
        if isinstance(first, str):
            return first.strip()
    return ""


def load_api_to_category(api_category_path: Path | None, api_catalog_path: Path | None) -> dict[str, str]:
    api_to_category: dict[str, str] = {}

    for path in [api_category_path, api_catalog_path]:
        if path is None or not path.exists():
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for api, value in data.items():
            if str(api) in api_to_category:
                continue
            category = extract_first_category(value)
            if category:
                api_to_category[str(api)] = category

    return api_to_category


def categories_for_record(
    record: MashupRecord,
    api_to_category: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    categories: list[str] = []
    missing_apis: list[str] = []
    for api in record.ground_truth_apis:
        category = api_to_category.get(api, "")
        if category:
            categories.append(category)
        else:
            missing_apis.append(api)

    if missing_apis:
        categories.extend(record.entry_categories)
    if not categories:
        categories = list(record.entry_categories)
    return unique_preserve_order(categories), missing_apis


def count_apis(records: Iterable[MashupRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(set(record.ground_truth_apis))
    return counts


def count_categories(
    records: Iterable[MashupRecord],
    api_to_category: Mapping[str, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        categories, _ = categories_for_record(record, api_to_category)
        counts.update(set(categories))
    return counts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def quantile_bucket(value: int | float, q25: float, q75: float) -> str:
    if value <= q25:
        return "low"
    if value <= q75:
        return "middle"
    return "high"


def numeric_summary(values: list[int | float]) -> dict[str, str]:
    if not values:
        return {"mean": NA, "median": NA, "min": NA, "max": NA}
    return {
        "mean": f"{mean(values):.3f}",
        "median": f"{median(values):.3f}",
        "min": str(min(values)),
        "max": str(max(values)),
    }


def label_count_bin(value: int) -> str:
    if value <= 0:
        return "0 APIs"
    if value == 1:
        return "1 API"
    if value in (2, 3, 4):
        return f"{value} APIs"
    return "5+ APIs"


def category_count_bin(value: int) -> str:
    if value <= 0:
        return "0 categories"
    if value == 1:
        return "1 category"
    if value in (2, 3, 4):
        return f"{value} categories"
    return "5+ categories"


def word_count_bin(value: int) -> str:
    if value <= 0:
        return "0 words"
    if value <= 20:
        return "short"
    if value <= 50:
        return "medium"
    return "long"


def seen_status(freqs: list[int]) -> str:
    if not freqs:
        return NA
    if all(freq > 0 for freq in freqs):
        return "Seen-only"
    if all(freq == 0 for freq in freqs):
        return "All-unseen"
    return "Partially-unseen"


def fixed_api_frequency_status(freqs: list[int]) -> str:
    if not freqs:
        return NA
    if any(1 <= freq <= 2 for freq in freqs):
        return "Rare-including"
    if any(3 <= freq <= 9 for freq in freqs):
        return "Medium-including"
    if all(freq >= 10 for freq in freqs):
        return "Frequent-only"
    if all(freq == 0 for freq in freqs):
        return "All-unseen"
    return "Unseen-including"


def quantile_frequency_status(bucket_values: list[str], item_name: str) -> str:
    if not bucket_values:
        return NA
    if "low" in bucket_values:
        return f"Low-frequency-{item_name}-including"
    if "middle" in bucket_values:
        return f"Middle-frequency-{item_name}-including"
    return f"High-frequency-{item_name}-only"


def single_or_multi_category(num_categories: int) -> str:
    if num_categories <= 0:
        return NA
    if num_categories == 1:
        return "Single-category"
    return "Multi-category"


def list_as_cell(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def value_or_na(value: Any) -> Any:
    if value is None:
        return NA
    return value


def average_or_na(values: list[int | float]) -> str:
    if not values:
        return NA
    return f"{mean(values):.3f}"


def load_hierarchy(path: Path) -> HierarchySpec | None:
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    raw_paths = data.get("category_paths")
    category_paths: dict[str, list[int]] = {}
    if isinstance(raw_paths, dict):
        for category, path_values in raw_paths.items():
            if isinstance(path_values, list):
                try:
                    category_paths[str(category)] = [int(v) for v in path_values]
                except (TypeError, ValueError):
                    continue
    elif isinstance(data.get("category_routes"), list):
        for route in data["category_routes"]:
            if not isinstance(route, dict) or "category" not in route:
                continue
            path_values: list[int] = []
            layer = 1
            while f"layer{layer}" in route:
                try:
                    path_values.append(int(route[f"layer{layer}"]))
                except (TypeError, ValueError):
                    break
                layer += 1
            if path_values:
                category_paths[str(route["category"])] = path_values

    if not category_paths:
        return None
    return HierarchySpec(label=infer_hierarchy_label(path), path=path, category_paths=category_paths)


def infer_hierarchy_label(path: Path) -> str:
    text = f"{path.parent.name}/{path.stem}"
    if "4_4_8" in text:
        return "low_e_4_4_8"
    if "12_4_3" in text or "12cluster" in text:
        return "middle_e_12_4_3"
    if "51-3" in text or "51_3" in text:
        return "high_e_51-3"
    return path.parent.name


def default_hierarchy_paths() -> list[Path]:
    candidate_groups = [
        [
            PROJECT_ROOT / "codex" / "output" / "by_pattern" / "4cluster" / "category_hierarchy_4_4_8.json",
            PROJECT_ROOT / "codex" / "output" / "4cluster" / "category_hierarchy_4_4_8.json",
        ],
        [
            PROJECT_ROOT / "codex" / "output" / "by_pattern" / "12_4_3" / "category_hierarchy_12_4_3.json",
            PROJECT_ROOT / "codex" / "output" / "12cluster" / "category_hierarchy_12cluster.json",
        ],
        [
            PROJECT_ROOT / "codex" / "output" / "by_pattern" / "51-3" / "category_hierarchy_51-3.json",
        ],
    ]
    representative: list[Path] = []
    for group in candidate_groups:
        for path in group:
            if path.exists():
                representative.append(path)
                break
    if representative:
        return unique_paths(representative)
    all_paths = sorted((PROJECT_ROOT / "codex" / "output").glob("**/category_hierarchy_*.json"))
    return unique_paths(all_paths)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def resolve_hierarchy_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.hierarchy_paths:
        paths.extend(Path(p) for p in args.hierarchy_paths)
    if args.hierarchy_glob:
        for pattern in args.hierarchy_glob:
            paths.extend(Path(p) for p in glob.glob(pattern))
    if not paths:
        paths = default_hierarchy_paths()
    return [path for path in unique_paths(paths) if path.exists()]


def primary_hierarchy(hierarchies: list[HierarchySpec]) -> HierarchySpec | None:
    if not hierarchies:
        return None
    for hierarchy in hierarchies:
        if hierarchy.label.startswith("middle_e"):
            return hierarchy
    return hierarchies[0]


def branch_info_for_categories(
    categories: list[str],
    hierarchy: HierarchySpec,
) -> dict[str, Any]:
    layer1: set[str] = set()
    layer2: set[str] = set()
    missing: list[str] = []
    for category in categories:
        path = hierarchy.category_paths.get(category)
        if not path:
            missing.append(category)
            continue
        if len(path) >= 1:
            layer1.add(str(path[0]))
        if len(path) >= 2:
            layer2.add(f"{path[0]}-{path[1]}")

    def status(values: set[str]) -> str:
        if not values:
            return NA
        if len(values) == 1:
            return "Single-branch"
        return "Multi-branch"

    return {
        "num_branches_layer1": len(layer1) if layer1 else NA,
        "num_branches_layer2": len(layer2) if layer2 else NA,
        "single_or_multi_branch_layer1": status(layer1),
        "single_or_multi_branch_layer2": status(layer2),
        "missing_branch_categories": missing,
    }


def build_sample_rows(
    records: list[MashupRecord],
    api_to_category: Mapping[str, str],
    train_api_count: Counter[str] | None,
    api_q25: float,
    api_q75: float,
    category_count: Counter[str],
    category_q25: float,
    category_q75: float,
    primary: HierarchySpec | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    missing_api_category_counter: Counter[str] = Counter()

    for record in records:
        gt_categories, missing_apis = categories_for_record(record, api_to_category)
        missing_api_category_counter.update(missing_apis)
        num_gt_apis = len(record.ground_truth_apis)
        num_gt_categories = len(gt_categories)
        word_count = len(record.description.split())

        if train_api_count is None:
            api_freqs: list[int] = []
            api_bucket_values: list[str] = []
        else:
            api_freqs = [int(train_api_count.get(api, 0)) for api in record.ground_truth_apis]
            api_bucket_values = [quantile_bucket(freq, api_q25, api_q75) for freq in api_freqs]

        category_freqs = [int(category_count.get(category, 0)) for category in gt_categories]
        category_bucket_values = [
            quantile_bucket(freq, category_q25, category_q75) for freq in category_freqs
        ]

        branch_info = (
            branch_info_for_categories(gt_categories, primary)
            if primary is not None
            else {
                "num_branches_layer1": NA,
                "num_branches_layer2": NA,
                "single_or_multi_branch_layer1": NA,
                "single_or_multi_branch_layer2": NA,
                "missing_branch_categories": [],
            }
        )

        row = {
            "sample_id": record.sample_id,
            "description": record.description,
            "ground_truth_apis": list_as_cell(record.ground_truth_apis),
            "num_gt_apis": num_gt_apis,
            "num_gt_apis_bin": label_count_bin(num_gt_apis),
            "ground_truth_categories": list_as_cell(gt_categories),
            "num_gt_categories": num_gt_categories,
            "num_gt_categories_bin": category_count_bin(num_gt_categories),
            "word_count": word_count,
            "word_count_bin": word_count_bin(word_count),
            "min_api_train_freq": min(api_freqs) if api_freqs else NA,
            "mean_api_train_freq": average_or_na(api_freqs),
            "max_api_train_freq": max(api_freqs) if api_freqs else NA,
            "num_unseen_apis": sum(1 for freq in api_freqs if freq == 0) if api_freqs else NA,
            "seen_status": seen_status(api_freqs) if train_api_count is not None else NA,
            "api_frequency_status": (
                fixed_api_frequency_status(api_freqs) if train_api_count is not None else NA
            ),
            "api_frequency_quantile_status": (
                quantile_frequency_status(api_bucket_values, "api")
                if train_api_count is not None
                else NA
            ),
            "min_category_freq": min(category_freqs) if category_freqs else NA,
            "mean_category_freq": average_or_na(category_freqs),
            "max_category_freq": max(category_freqs) if category_freqs else NA,
            "category_frequency_status": quantile_frequency_status(
                category_bucket_values, "category"
            ),
            "single_or_multi_category": single_or_multi_category(num_gt_categories),
            "num_branches_layer1": branch_info["num_branches_layer1"],
            "num_branches_layer2": branch_info["num_branches_layer2"],
            "single_or_multi_branch_layer1": branch_info["single_or_multi_branch_layer1"],
            "single_or_multi_branch_layer2": branch_info["single_or_multi_branch_layer2"],
            "missing_api_categories": list_as_cell(missing_apis),
            "missing_branch_categories": list_as_cell(branch_info["missing_branch_categories"]),
        }
        rows.append(row)

    return rows, missing_api_category_counter


def ratio(count: int, total: int) -> str:
    if total == 0:
        return NA
    return f"{count / total:.3f}"


def add_summary_row(
    rows: list[dict[str, Any]],
    total: int,
    condition_name: str,
    condition_definition: str,
    count: int | str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "condition_name": condition_name,
            "condition_definition": condition_definition,
            "count": count,
            "ratio": ratio(count, total) if isinstance(count, int) else NA,
            "notes": notes,
        }
    )


def count_where(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    return sum(1 for row in rows if predicate(row))


def build_summary_counts(
    sample_rows: list[dict[str, Any]],
    has_train: bool,
    has_branch: bool,
    branch_by_config: dict[str, list[dict[str, Any]]],
    primary_label: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    total = len(sample_rows)
    summary: list[dict[str, Any]] = []

    add_summary_row(summary, total, "Total samples", "all test samples", total)
    add_summary_row(
        summary,
        total,
        "Missing description",
        "description is empty after string conversion",
        count_where(sample_rows, lambda row: str(row["description"]).strip() == ""),
    )
    add_summary_row(
        summary,
        total,
        "Missing ground-truth APIs",
        "num_gt_apis == 0",
        count_where(sample_rows, lambda row: int(row["num_gt_apis"]) == 0),
    )
    add_summary_row(
        summary,
        total,
        "Missing API category labels",
        "at least one gt API has no API-to-category mapping",
        count_where(sample_rows, lambda row: row["missing_api_categories"] != "[]"),
    )

    for bucket in ["1 API", "2 APIs", "3 APIs", "4 APIs", "5+ APIs"]:
        add_summary_row(
            summary,
            total,
            bucket,
            f"num_gt_apis bin == {bucket}",
            count_where(sample_rows, lambda row, b=bucket: row["num_gt_apis_bin"] == b),
        )

    for bucket in [
        "1 category",
        "2 categories",
        "3 categories",
        "4 categories",
        "5+ categories",
    ]:
        add_summary_row(
            summary,
            total,
            bucket,
            f"num_gt_categories bin == {bucket}",
            count_where(sample_rows, lambda row, b=bucket: row["num_gt_categories_bin"] == b),
        )

    for bucket, definition in [
        ("short", "1 <= word_count <= 20"),
        ("medium", "21 <= word_count <= 50"),
        ("long", "word_count >= 51"),
    ]:
        add_summary_row(
            summary,
            total,
            f"Description {bucket}",
            definition,
            count_where(sample_rows, lambda row, b=bucket: row["word_count_bin"] == b),
        )

    if has_train:
        for status in ["Seen-only", "Partially-unseen", "All-unseen"]:
            add_summary_row(
                summary,
                total,
                status,
                f"seen_status == {status}",
                count_where(sample_rows, lambda row, s=status: row["seen_status"] == s),
            )
        add_summary_row(
            summary,
            total,
            "Rare-including",
            "at least one gt API has train frequency 1-2",
            count_where(
                sample_rows,
                lambda row: intish(row["min_api_train_freq"]) is not None
                and any_freq_in_cell(row, 1, 2),
            ),
        )
        add_summary_row(
            summary,
            total,
            "Medium-including",
            "at least one gt API has train frequency 3-9",
            count_where(sample_rows, lambda row: any_freq_in_cell(row, 3, 9)),
        )
        add_summary_row(
            summary,
            total,
            "Frequent-only",
            "all gt APIs have train frequency >= 10",
            count_where(sample_rows, lambda row: row["api_frequency_status"] == "Frequent-only"),
        )
        for status in [
            "Low-frequency-api-including",
            "Middle-frequency-api-including",
            "High-frequency-api-only",
        ]:
            add_summary_row(
                summary,
                total,
                f"{status} (quantile)",
                f"api_frequency_quantile_status == {status}",
                count_where(
                    sample_rows,
                    lambda row, s=status: row["api_frequency_quantile_status"] == s,
                ),
            )
    else:
        add_summary_row(
            summary,
            total,
            "API train frequency conditions",
            "train data is unavailable",
            NA,
            "Skipped because train data was not available.",
        )

    for status in [
        "Low-frequency-category-including",
        "Middle-frequency-category-including",
        "High-frequency-category-only",
    ]:
        add_summary_row(
            summary,
            total,
            status,
            f"category_frequency_status == {status}",
            count_where(sample_rows, lambda row, s=status: row["category_frequency_status"] == s),
        )

    add_summary_row(
        summary,
        total,
        "Single-category",
        "num_gt_categories == 1",
        count_where(sample_rows, lambda row: row["single_or_multi_category"] == "Single-category"),
    )
    add_summary_row(
        summary,
        total,
        "Multi-category",
        "num_gt_categories >= 2",
        count_where(sample_rows, lambda row: row["single_or_multi_category"] == "Multi-category"),
    )
    for name, predicate in [
        ("Multi-category: 2 categories", lambda row: int(row["num_gt_categories"]) == 2),
        ("Multi-category: 3 categories", lambda row: int(row["num_gt_categories"]) == 3),
        ("Multi-category: 4+ categories", lambda row: int(row["num_gt_categories"]) >= 4),
    ]:
        add_summary_row(summary, total, name, name, count_where(sample_rows, predicate))

    if has_branch:
        for config_label, branch_rows in branch_by_config.items():
            for layer in ("layer1", "layer2"):
                status_key = f"single_or_multi_branch_{layer}"
                add_summary_row(
                    summary,
                    total,
                    f"{config_label}: Single-branch at {layer}",
                    f"{status_key} == Single-branch",
                    count_where(branch_rows, lambda row, k=status_key: row[k] == "Single-branch"),
                )
                add_summary_row(
                    summary,
                    total,
                    f"{config_label}: Multi-branch at {layer}",
                    f"{status_key} == Multi-branch",
                    count_where(branch_rows, lambda row, k=status_key: row[k] == "Multi-branch"),
                )
    else:
        add_summary_row(
            summary,
            total,
            "Branch spread conditions",
            "hierarchy data is unavailable",
            NA,
            "Skipped because branch information was not available.",
        )

    composite_counts: dict[str, int] = {}
    composites: list[tuple[str, str, Callable[[dict[str, Any]], bool], str]] = [
        (
            "Condition A: Representative-friendly",
            "1 <= num_gt_apis <= 3 and 1 <= num_gt_categories <= 2",
            lambda row: 1 <= int(row["num_gt_apis"]) <= 3
            and 1 <= int(row["num_gt_categories"]) <= 2,
            "",
        ),
        (
            "Condition B: Multi-category",
            "num_gt_categories >= 2",
            lambda row: int(row["num_gt_categories"]) >= 2,
            "",
        ),
    ]
    if has_train:
        composites.append(
            (
                "Condition C: Difficult rare multi-category",
                "num_gt_categories >= 2 and Rare-including",
                lambda row: int(row["num_gt_categories"]) >= 2
                and row["api_frequency_status"] == "Rare-including",
                "",
            )
        )
    else:
        composites.append(
            (
                "Condition C: Difficult rare multi-category",
                "num_gt_categories >= 2 and Rare-including",
                lambda row: False,
                "Skipped because train data was not available.",
            )
        )

    if has_branch:
        composites.append(
            (
                "Condition D: Hierarchy-sensitive",
                f"num_gt_categories >= 2 and multi-branch in {primary_label}",
                lambda row: int(row["num_gt_categories"]) >= 2
                and (
                    row["single_or_multi_branch_layer1"] == "Multi-branch"
                    or row["single_or_multi_branch_layer2"] == "Multi-branch"
                ),
                "",
            )
        )
    else:
        composites.append(
            (
                "Condition D: Hierarchy-sensitive fallback",
                "num_gt_categories >= 2",
                lambda row: int(row["num_gt_categories"]) >= 2,
                "Branch information was unavailable, so this uses the category fallback.",
            )
        )

    if has_train:
        composites.append(
            (
                "Condition E: Stable evaluation",
                "Seen-only and 1 <= num_gt_apis <= 3 and 1 <= num_gt_categories <= 2",
                lambda row: row["seen_status"] == "Seen-only"
                and 1 <= int(row["num_gt_apis"]) <= 3
                and 1 <= int(row["num_gt_categories"]) <= 2,
                "",
            )
        )
    else:
        composites.append(
            (
                "Condition E: Stable evaluation fallback",
                "1 <= num_gt_apis <= 3 and 1 <= num_gt_categories <= 2",
                lambda row: 1 <= int(row["num_gt_apis"]) <= 3
                and 1 <= int(row["num_gt_categories"]) <= 2,
                "Train data was unavailable, so Seen-only was omitted.",
            )
        )

    for name, definition, predicate, notes in composites:
        count = count_where(sample_rows, predicate)
        composite_counts[name] = count
        add_summary_row(summary, total, name, definition, count, notes)

    return summary, composite_counts


def intish(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def any_freq_in_cell(row: dict[str, Any], lower: int, upper: int) -> bool:
    cell = row.get("_api_freqs")
    if not isinstance(cell, list):
        return False
    return any(lower <= int(freq) <= upper for freq in cell)


def attach_internal_frequency_lists(
    sample_rows: list[dict[str, Any]],
    records: list[MashupRecord],
    train_api_count: Counter[str] | None,
) -> None:
    if train_api_count is None:
        return
    for row, record in zip(sample_rows, records):
        row["_api_freqs"] = [int(train_api_count.get(api, 0)) for api in record.ground_truth_apis]


def strip_internal_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def build_branch_rows(
    sample_rows: list[dict[str, Any]],
    hierarchies: list[HierarchySpec],
) -> dict[str, list[dict[str, Any]]]:
    branch_by_config: dict[str, list[dict[str, Any]]] = {}
    for hierarchy in hierarchies:
        rows: list[dict[str, Any]] = []
        for sample_row in sample_rows:
            categories = json.loads(sample_row["ground_truth_categories"])
            info = branch_info_for_categories(categories, hierarchy)
            rows.append(
                {
                    "sample_id": sample_row["sample_id"],
                    "num_branches_layer1": info["num_branches_layer1"],
                    "num_branches_layer2": info["num_branches_layer2"],
                    "single_or_multi_branch_layer1": info["single_or_multi_branch_layer1"],
                    "single_or_multi_branch_layer2": info["single_or_multi_branch_layer2"],
                }
            )
        branch_by_config[hierarchy.label] = rows
    return branch_by_config


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_label_count_distribution(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["num_gt_apis_bin"] for row in sample_rows)
    values = [int(row["num_gt_apis"]) for row in sample_rows]
    stats = numeric_summary(values)
    order = ["1 API", "2 APIs", "3 APIs", "4 APIs", "5+ APIs"]
    total = len(sample_rows)
    return [
        {
            "metric": "num_gt_apis",
            "bucket": bucket,
            "count": counts.get(bucket, 0),
            "ratio": ratio(counts.get(bucket, 0), total),
            **stats,
        }
        for bucket in order
    ]


def build_category_spread_distribution(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["num_gt_categories_bin"] for row in sample_rows)
    single_multi = Counter(row["single_or_multi_category"] for row in sample_rows)
    values = [int(row["num_gt_categories"]) for row in sample_rows]
    stats = numeric_summary(values)
    total = len(sample_rows)
    rows: list[dict[str, Any]] = []
    for bucket in ["1 category", "2 categories", "3 categories", "4 categories", "5+ categories"]:
        rows.append(
            {
                "metric": "num_gt_categories",
                "bucket": bucket,
                "count": counts.get(bucket, 0),
                "ratio": ratio(counts.get(bucket, 0), total),
                **stats,
            }
        )
    for bucket in ["Single-category", "Multi-category"]:
        rows.append(
            {
                "metric": "single_or_multi_category",
                "bucket": bucket,
                "count": single_multi.get(bucket, 0),
                "ratio": ratio(single_multi.get(bucket, 0), total),
                **stats,
            }
        )
    return rows


def build_branch_spread_distribution(
    branch_by_config: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_label, branch_rows in branch_by_config.items():
        total = len(branch_rows)
        for layer in ("layer1", "layer2"):
            count_key = f"num_branches_{layer}"
            status_key = f"single_or_multi_branch_{layer}"
            branch_counts = Counter(str(row[count_key]) for row in branch_rows)
            status_counts = Counter(str(row[status_key]) for row in branch_rows)
            for bucket, count in sorted(branch_counts.items(), key=lambda item: natural_branch_sort(item[0])):
                rows.append(
                    {
                        "hierarchy_config": config_label,
                        "layer": layer,
                        "metric": "num_branches",
                        "bucket": bucket,
                        "count": count,
                        "ratio": ratio(count, total),
                    }
                )
            for bucket in ["Single-branch", "Multi-branch", NA]:
                rows.append(
                    {
                        "hierarchy_config": config_label,
                        "layer": layer,
                        "metric": "single_or_multi_branch",
                        "bucket": bucket,
                        "count": status_counts.get(bucket, 0),
                        "ratio": ratio(status_counts.get(bucket, 0), total),
                    }
                )
    return rows


def natural_branch_sort(value: str) -> tuple[int, str]:
    if value == NA:
        return (10**9, value)
    try:
        return (int(value), value)
    except ValueError:
        return (10**8, value)


def build_category_distribution(
    test_category_count: Counter[str],
    source_category_count: Counter[str],
    train_category_count: Counter[str] | None,
    category_q25: float,
    category_q75: float,
    frequency_source: str,
) -> list[dict[str, Any]]:
    categories = sorted(set(test_category_count) | set(source_category_count) | set(train_category_count or {}))
    total_test = sum(test_category_count.values())
    rows: list[dict[str, Any]] = []
    for category in categories:
        source_count = int(source_category_count.get(category, 0))
        rows.append(
            {
                "category": category,
                "test_count": int(test_category_count.get(category, 0)),
                "test_ratio": ratio(int(test_category_count.get(category, 0)), total_test),
                "train_count": int(train_category_count.get(category, 0)) if train_category_count else NA,
                "frequency_source": frequency_source,
                "frequency_source_count": source_count,
                "frequency_bucket": quantile_bucket(source_count, category_q25, category_q75),
            }
        )
    return rows


def build_api_frequency_distribution(
    api_to_category: Mapping[str, str],
    test_api_count: Counter[str],
    train_api_count: Counter[str] | None,
    api_q25: float,
    api_q75: float,
) -> list[dict[str, Any]]:
    apis = sorted(set(api_to_category) | set(test_api_count) | set(train_api_count or {}))
    rows: list[dict[str, Any]] = []
    for api in apis:
        train_count = int(train_api_count.get(api, 0)) if train_api_count is not None else NA
        fixed_bucket = NA
        quantile = NA
        if train_api_count is not None:
            if train_count == 0:
                fixed_bucket = "unseen"
            elif train_count <= 2:
                fixed_bucket = "rare"
            elif train_count <= 9:
                fixed_bucket = "medium"
            else:
                fixed_bucket = "frequent"
            quantile = quantile_bucket(train_count, api_q25, api_q75)
        rows.append(
            {
                "api": api,
                "category": api_to_category.get(api, NA),
                "test_count": int(test_api_count.get(api, 0)),
                "train_count": train_count,
                "fixed_frequency_bucket": fixed_bucket,
                "quantile_frequency_bucket": quantile,
            }
        )
    return rows


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    counts = Counter(str(row.get(key, NA)) for row in rows)
    total = len(rows)
    if total == 0:
        return {}
    return {bucket: count / total for bucket, count in counts.items()}


def max_ratio_delta(full_rows: list[dict[str, Any]], subset_rows: list[dict[str, Any]], key: str) -> float:
    full = distribution(full_rows, key)
    subset = distribution(subset_rows, key)
    buckets = set(full) | set(subset)
    if not buckets:
        return 0.0
    return max(abs(full.get(bucket, 0.0) - subset.get(bucket, 0.0)) for bucket in buckets)


def representative_subset(sample_rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size >= len(sample_rows):
        return list(sample_rows)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        key = (
            str(row["num_gt_apis_bin"]),
            str(row["num_gt_categories_bin"]),
            str(row["api_frequency_status"]),
            str(row["single_or_multi_category"]),
        )
        groups[key].append(row)

    allocation: dict[tuple[str, str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str, str]]] = []
    for key, group_rows in groups.items():
        exact = len(group_rows) * size / len(sample_rows)
        base = min(int(math.floor(exact)), len(group_rows))
        allocation[key] = base
        remainders.append((exact - base, key))

    selected: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        selected.extend(group_rows[: allocation[key]])

    leftover = size - len(selected)
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        if allocation[key] < len(groups[key]):
            selected.append(groups[key][allocation[key]])
            allocation[key] += 1
            leftover -= 1

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


def write_subset_candidates(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    sizes: list[int],
) -> list[dict[str, Any]]:
    candidate_dir = output_dir / "subset_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    feasibility_rows: list[dict[str, Any]] = []
    total = len(sample_rows)
    for size in sizes:
        if size > total:
            feasibility_rows.append(
                {
                    "size": size,
                    "feasible": "no",
                    "available": total,
                    "candidate_file": NA,
                    "max_delta_num_gt_apis": NA,
                    "max_delta_num_gt_categories": NA,
                    "max_delta_api_frequency_status": NA,
                    "max_delta_single_or_multi_category": NA,
                }
            )
            continue
        subset = representative_subset(sample_rows, size)
        path = candidate_dir / f"representative_{size}_ids.txt"
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(f"{row['sample_id']}\n")
        feasibility_rows.append(
            {
                "size": size,
                "feasible": "yes",
                "available": total,
                "candidate_file": str(path.relative_to(output_dir)),
                "max_delta_num_gt_apis": f"{max_ratio_delta(sample_rows, subset, 'num_gt_apis_bin'):.3f}",
                "max_delta_num_gt_categories": f"{max_ratio_delta(sample_rows, subset, 'num_gt_categories_bin'):.3f}",
                "max_delta_api_frequency_status": f"{max_ratio_delta(sample_rows, subset, 'api_frequency_status'):.3f}",
                "max_delta_single_or_multi_category": f"{max_ratio_delta(sample_rows, subset, 'single_or_multi_category'):.3f}",
            }
        )
    return feasibility_rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [str(row.get(column, NA)).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    composite_counts: dict[str, int],
    feasibility_rows: list[dict[str, Any]],
    has_train: bool,
    has_branch: bool,
    train_path: Path | None,
    hierarchy_labels: list[str],
    primary_label: str,
    api_thresholds: tuple[float, float],
    category_thresholds: tuple[float, float],
    category_frequency_source: str,
) -> str:
    total = len(sample_rows)
    api_stats = numeric_summary([int(row["num_gt_apis"]) for row in sample_rows])
    category_stats = numeric_summary([int(row["num_gt_categories"]) for row in sample_rows])
    word_stats = numeric_summary([int(row["word_count"]) for row in sample_rows])

    enough_50 = [name for name, count in composite_counts.items() if count >= 50]
    short_conditions = [name for name, count in composite_counts.items() if count < 50]

    lines = [
        "# Test Dataset Subset Condition Analysis",
        "",
        "## Dataset Overview",
        f"- total samples: {total}",
        f"- average number of ground-truth APIs: {api_stats['mean']}",
        f"- median/min/max ground-truth APIs: {api_stats['median']} / {api_stats['min']} / {api_stats['max']}",
        f"- average number of ground-truth categories: {category_stats['mean']}",
        f"- median/min/max ground-truth categories: {category_stats['median']} / {category_stats['min']} / {category_stats['max']}",
        f"- average word count: {word_stats['mean']}",
        f"- median/min/max word count: {word_stats['median']} / {word_stats['min']} / {word_stats['max']}",
        "",
        "## Condition Counts",
        markdown_table(summary_rows, ["condition_name", "condition_definition", "count", "ratio", "notes"]),
        "",
        "## Subset Size Feasibility",
        markdown_table(
            feasibility_rows,
            [
                "size",
                "feasible",
                "available",
                "candidate_file",
                "max_delta_num_gt_apis",
                "max_delta_num_gt_categories",
                "max_delta_api_frequency_status",
                "max_delta_single_or_multi_category",
            ],
        ),
        "",
        "## Important Findings",
        f"- Conditions with at least 50 samples: {', '.join(enough_50) if enough_50 else 'none'}",
        f"- Conditions below 50 samples: {', '.join(short_conditions) if short_conditions else 'none'}",
        "- Candidate ID lists were generated deterministically with stratification over API-count, category-count, API-frequency status, and single/multi-category labels.",
        "",
        "## Recommended Subset Conditions",
        "- Representative subset: Condition A when the target size fits; otherwise use the generated representative ID lists.",
        "- Rare / long-tail subset: Condition C when train frequency is available.",
        "- Multi-category subset: Condition B.",
        "- Stable evaluation subset: Condition E when train frequency is available; otherwise Condition A.",
        "",
        "## Notes",
        f"- train data: {train_path if has_train and train_path is not None else 'not available'}",
        f"- API frequency quantile thresholds: q25={api_thresholds[0]:.3f}, q75={api_thresholds[1]:.3f}" if has_train else "- API train-frequency conditions were skipped because train data was not available.",
        f"- category frequency source: {category_frequency_source}",
        f"- category frequency quantile thresholds: q25={category_thresholds[0]:.3f}, q75={category_thresholds[1]:.3f}",
    ]
    if has_branch:
        lines.append(f"- hierarchy configurations: {', '.join(hierarchy_labels)}")
        lines.append(f"- primary hierarchy for sample_condition_flags and Condition D: {primary_label}")
    else:
        lines.append("- branch information was not available, so branch spread was not calculated.")
    lines.append(f"- output directory: {output_dir}")
    return "\n".join(lines) + "\n"


def parse_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count test subset conditions for E/F1 analysis without model training."
    )
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH), help="336-sample test JSON path.")
    parser.add_argument(
        "--api-category-path",
        default=str(DEFAULT_API_CATEGORY_PATH),
        help="API-to-category JSON path. Missing categories are filled from --api-catalog-path.",
    )
    parser.add_argument(
        "--api-catalog-path",
        default=str(DEFAULT_API_CATALOG_PATH),
        help="API catalog / metadata JSON path.",
    )
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_TRAIN_PATH),
        help="Optional train mashup JSON path. Use an empty string to skip train-frequency analysis.",
    )
    parser.add_argument(
        "--hierarchy-paths",
        nargs="*",
        default=None,
        help="Optional category_hierarchy JSON paths. Defaults to representative existing hierarchies.",
    )
    parser.add_argument(
        "--hierarchy-glob",
        nargs="*",
        default=None,
        help="Optional glob patterns for hierarchy JSON files.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--candidate-sizes", default="50,100,150,200,250,336")
    parser.add_argument("--expected-test-size", type=int, default=336)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test_path)
    api_category_path = Path(args.api_category_path) if args.api_category_path else None
    api_catalog_path = Path(args.api_catalog_path) if args.api_catalog_path else None
    train_path = Path(args.train_path) if args.train_path else None
    output_dir = Path(args.output_dir)

    records = load_mashup_records(test_path, limit=args.limit)
    if args.expected_test_size and args.limit is None and len(records) != args.expected_test_size:
        print(
            f"[WARN] expected {args.expected_test_size} test samples, but loaded {len(records)} from {test_path}"
        )

    api_to_category = load_api_to_category(api_category_path, api_catalog_path)
    if not api_to_category:
        raise ValueError("No API-to-category metadata could be loaded.")

    train_records: list[MashupRecord] = []
    train_api_count: Counter[str] | None = None
    train_category_count: Counter[str] | None = None
    has_train = train_path is not None and train_path.exists()
    if has_train:
        train_records = load_mashup_records(train_path)
        train_api_count = count_apis(train_records)
        train_category_count = count_categories(train_records, api_to_category)

    test_api_count = count_apis(records)
    test_category_count = count_categories(records, api_to_category)

    api_universe = sorted(set(api_to_category) | set(test_api_count) | set(train_api_count or {}))
    if train_api_count is not None:
        api_frequency_values = [int(train_api_count.get(api, 0)) for api in api_universe]
        api_q25 = percentile(api_frequency_values, 0.25)
        api_q75 = percentile(api_frequency_values, 0.75)
    else:
        api_q25 = 0.0
        api_q75 = 0.0

    if train_category_count:
        source_category_count = train_category_count
        category_frequency_source = "train"
    else:
        source_category_count = test_category_count
        category_frequency_source = "test"
    category_universe = sorted(
        set(source_category_count) | set(test_category_count) | set(api_to_category.values())
    )
    category_frequency_values = [int(source_category_count.get(category, 0)) for category in category_universe]
    category_q25 = percentile(category_frequency_values, 0.25)
    category_q75 = percentile(category_frequency_values, 0.75)

    hierarchy_paths = resolve_hierarchy_paths(args)
    hierarchies = [hierarchy for path in hierarchy_paths if (hierarchy := load_hierarchy(path))]
    primary = primary_hierarchy(hierarchies)

    sample_rows, missing_api_category_counter = build_sample_rows(
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
    attach_internal_frequency_lists(sample_rows, records, train_api_count)

    branch_by_config = build_branch_rows(sample_rows, hierarchies) if hierarchies else {}
    summary_rows, composite_counts = build_summary_counts(
        sample_rows=sample_rows,
        has_train=has_train,
        has_branch=bool(hierarchies),
        branch_by_config=branch_by_config,
        primary_label=primary.label if primary else NA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_fieldnames = [
        "sample_id",
        "description",
        "ground_truth_apis",
        "num_gt_apis",
        "ground_truth_categories",
        "num_gt_categories",
        "word_count",
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
        "missing_api_categories",
        "missing_branch_categories",
        "num_gt_apis_bin",
        "num_gt_categories_bin",
        "word_count_bin",
    ]
    write_csv(output_dir / "sample_condition_flags.csv", sample_fieldnames, strip_internal_fields(sample_rows))
    write_csv(
        output_dir / "summary_counts.csv",
        ["condition_name", "condition_definition", "count", "ratio", "notes"],
        summary_rows,
    )
    write_csv(
        output_dir / "category_distribution.csv",
        [
            "category",
            "test_count",
            "test_ratio",
            "train_count",
            "frequency_source",
            "frequency_source_count",
            "frequency_bucket",
        ],
        build_category_distribution(
            test_category_count,
            source_category_count,
            train_category_count,
            category_q25,
            category_q75,
            category_frequency_source,
        ),
    )
    write_csv(
        output_dir / "api_frequency_distribution.csv",
        [
            "api",
            "category",
            "test_count",
            "train_count",
            "fixed_frequency_bucket",
            "quantile_frequency_bucket",
        ],
        build_api_frequency_distribution(api_to_category, test_api_count, train_api_count, api_q25, api_q75),
    )
    write_csv(
        output_dir / "label_count_distribution.csv",
        ["metric", "bucket", "count", "ratio", "mean", "median", "min", "max"],
        build_label_count_distribution(sample_rows),
    )
    write_csv(
        output_dir / "category_spread_distribution.csv",
        ["metric", "bucket", "count", "ratio", "mean", "median", "min", "max"],
        build_category_spread_distribution(sample_rows),
    )
    if branch_by_config:
        write_csv(
            output_dir / "branch_spread_distribution.csv",
            ["hierarchy_config", "layer", "metric", "bucket", "count", "ratio"],
            build_branch_spread_distribution(branch_by_config),
        )

    candidate_sizes = parse_sizes(args.candidate_sizes)
    feasibility_rows = write_subset_candidates(output_dir, sample_rows, candidate_sizes)
    write_csv(
        output_dir / "subset_feasibility.csv",
        [
            "size",
            "feasible",
            "available",
            "candidate_file",
            "max_delta_num_gt_apis",
            "max_delta_num_gt_categories",
            "max_delta_api_frequency_status",
            "max_delta_single_or_multi_category",
        ],
        feasibility_rows,
    )

    report = build_report(
        output_dir=output_dir,
        sample_rows=sample_rows,
        summary_rows=summary_rows,
        composite_counts=composite_counts,
        feasibility_rows=feasibility_rows,
        has_train=has_train,
        has_branch=bool(hierarchies),
        train_path=train_path,
        hierarchy_labels=[hierarchy.label for hierarchy in hierarchies],
        primary_label=primary.label if primary else NA,
        api_thresholds=(api_q25, api_q75),
        category_thresholds=(category_q25, category_q75),
        category_frequency_source=category_frequency_source,
    )
    (output_dir / "subset_feasibility_report.md").write_text(report, encoding="utf-8")

    metadata = {
        "test_path": str(test_path),
        "train_path": str(train_path) if has_train else None,
        "api_category_path": str(api_category_path) if api_category_path else None,
        "api_catalog_path": str(api_catalog_path) if api_catalog_path else None,
        "hierarchy_paths": [str(hierarchy.path) for hierarchy in hierarchies],
        "primary_hierarchy": primary.label if primary else None,
        "total_samples": len(records),
        "missing_api_category_counts": dict(missing_api_category_counter.most_common()),
        "api_frequency_quantiles": {"q25": api_q25, "q75": api_q75} if has_train else None,
        "category_frequency_source": category_frequency_source,
        "category_frequency_quantiles": {"q25": category_q25, "q75": category_q75},
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded test samples: {len(records)}")
    print(f"Loaded API categories: {len(api_to_category)}")
    print(f"Train frequency analysis: {'yes' if has_train else 'no'}")
    print(f"Hierarchy analysis: {', '.join(h.label for h in hierarchies) if hierarchies else 'no'}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
