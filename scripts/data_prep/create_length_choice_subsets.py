#!/usr/bin/env python3
"""Create length x choice subset tables and ID lists from sample_condition_flags.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_test_subset_conditions as base

NA = base.NA
LENGTH_ORDER = ["short", "medium", "long"]
CHOICE_ORDER = ["small", "medium", "large"]

SUMMARY_FIELDS = [
    "subset_name",
    "condition_definition",
    "count",
    "ratio",
    "avg_word_count",
    "median_word_count",
    "min_word_count",
    "max_word_count",
    "avg_sum_oracle_candidate_size",
    "median_sum_oracle_candidate_size",
    "min_sum_oracle_candidate_size",
    "max_sum_oracle_candidate_size",
    "avg_num_gt_apis",
    "avg_num_gt_categories",
    "seen_only_count",
    "seen_only_ratio",
    "partially_unseen_count",
    "partially_unseen_ratio",
    "all_unseen_count",
    "all_unseen_ratio",
    "single_category_count",
    "single_category_ratio",
    "multi_category_count",
    "multi_category_ratio",
    "clear_requirement_count",
    "clear_requirement_ratio",
    "middle_requirement_count",
    "middle_requirement_ratio",
    "ambiguous_requirement_count",
    "ambiguous_requirement_ratio",
    "seen_api_combination_count",
    "seen_api_combination_ratio",
    "novel_api_combination_count",
    "novel_api_combination_ratio",
    "seen_category_combination_count",
    "seen_category_combination_ratio",
    "novel_category_combination_count",
    "novel_category_combination_ratio",
    "feasibility_label",
    "feasibility_note",
]


def read_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row.get('sample_id', '')}\n")


def ratio(count: int, total: int) -> str:
    return base.ratio(count, total)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == NA:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_int(value: Any) -> int | None:
    numeric = as_float(value)
    if numeric is None:
        return None
    return int(numeric)


def normalize_length_bin(row: dict[str, Any], warnings: list[str]) -> str:
    value = str(row.get("word_count_bin", "")).strip()
    if value in LENGTH_ORDER:
        return value
    word_count = as_int(row.get("word_count"))
    if word_count is not None:
        return base.word_count_bin(word_count)
    warnings.append(f"sample {row.get('sample_id', NA)} has no usable word_count_bin")
    return NA


def normalize_choice_bin(row: dict[str, Any], warnings: list[str]) -> str:
    value = str(row.get("oracle_category_candidate_size_bin", "")).strip()
    mapping = {
        "small": "small",
        "medium": "medium",
        "large": "large",
        "small_candidate": "small",
        "medium_candidate": "medium",
        "large_candidate": "large",
    }
    if value in mapping:
        return mapping[value]
    candidate_size = as_int(row.get("sum_oracle_category_candidate_size"))
    if candidate_size is not None:
        if candidate_size <= 20:
            return "small"
        if candidate_size <= 100:
            return "medium"
        return "large"
    warnings.append(f"sample {row.get('sample_id', NA)} has no usable oracle candidate size bin")
    return NA


def prepare_rows(rows: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out = dict(row)
        out["_original_index"] = index
        out["length_bin_normalized"] = normalize_length_bin(out, warnings)
        out["choice_bin_normalized"] = normalize_choice_bin(out, warnings)
        prepared.append(out)
    return prepared


def subset_name(length_bin: str, choice_bin: str) -> str:
    return f"{length_bin}_{choice_bin}"


def condition_text_for_cell(length_bin: str, choice_bin: str) -> str:
    return f"word_count_bin == {length_bin} and oracle_candidate_size_bin == {choice_bin}"


def cell_rows(rows: list[dict[str, Any]], length_bin: str, choice_bin: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("length_bin_normalized") == length_bin and row.get("choice_bin_normalized") == choice_bin
    ]


def build_cross_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    output: list[dict[str, Any]] = []
    for length_bin in LENGTH_ORDER:
        for choice_bin in CHOICE_ORDER:
            count = len(cell_rows(rows, length_bin, choice_bin))
            output.append(
                {
                    "word_count_bin": length_bin,
                    "oracle_candidate_size_bin": choice_bin,
                    "count": count,
                    "ratio": ratio(count, total),
                }
            )
    return output


def build_pivot(cross_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {(row["word_count_bin"], row["oracle_candidate_size_bin"]): row["count"] for row in cross_rows}
    output: list[dict[str, Any]] = []
    for length_bin in LENGTH_ORDER:
        row = {"word_count_bin": length_bin}
        for choice_bin in CHOICE_ORDER:
            row[choice_bin] = counts.get((length_bin, choice_bin), 0)
        row["total"] = sum(int(row[choice_bin]) for choice_bin in CHOICE_ORDER)
        output.append(row)
    return output


def feasibility(count: int) -> tuple[str, str]:
    if count >= 100:
        return "main_or_comparison_ready", "enough samples for comparative evaluation"
    if count >= 50:
        return "auxiliary_ready", "usable for auxiliary evaluation, but variance should be noted"
    if count >= 30:
        return "diagnostic_only", "usable for diagnostic analysis, not strong enough for main claims"
    return "case_study_only", "too small for stable F1 comparison; use for qualitative case analysis"


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def stat(values: list[float], kind: str) -> str:
    if not values:
        return NA
    if kind == "avg":
        return f"{mean(values):.3f}"
    if kind == "median":
        return f"{median(values):.3f}"
    if kind == "min":
        return f"{min(values):.3f}"
    if kind == "max":
        return f"{max(values):.3f}"
    raise ValueError(kind)


def count_status(rows: list[dict[str, Any]], key: str, expected: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "")).strip() == expected)


def count_not_status(rows: list[dict[str, Any]], key: str, expected: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "")).strip() not in {expected, "", NA})


def subset_summary_row(
    name: str,
    definition: str,
    subset: list[dict[str, Any]],
    total: int,
) -> dict[str, Any]:
    count = len(subset)
    word_values = numeric_values(subset, "word_count")
    candidate_values = numeric_values(subset, "sum_oracle_category_candidate_size")
    api_values = numeric_values(subset, "num_gt_apis")
    category_values = numeric_values(subset, "num_gt_categories")
    seen_only = count_status(subset, "seen_status", "Seen-only")
    partially_unseen = count_status(subset, "seen_status", "Partially-unseen")
    all_unseen = count_status(subset, "seen_status", "All-unseen")
    single_category = count_status(subset, "single_or_multi_category", "Single-category")
    multi_category = count_status(subset, "single_or_multi_category", "Multi-category")
    clear = count_status(subset, "requirement_clarity_status", "Clear-requirement")
    middle = count_status(subset, "requirement_clarity_status", "Middle-requirement")
    ambiguous = count_status(subset, "requirement_clarity_status", "Ambiguous-requirement")
    seen_api = count_status(subset, "api_combination_status", "api_combination_exact_seen")
    novel_api = count_not_status(subset, "api_combination_status", "api_combination_exact_seen")
    seen_category = count_status(subset, "category_combination_status", "category_combination_exact_seen")
    novel_category = count_not_status(subset, "category_combination_status", "category_combination_exact_seen")
    label, note = feasibility(count)

    return {
        "subset_name": name,
        "condition_definition": definition,
        "count": count,
        "ratio": ratio(count, total),
        "avg_word_count": stat(word_values, "avg"),
        "median_word_count": stat(word_values, "median"),
        "min_word_count": stat(word_values, "min"),
        "max_word_count": stat(word_values, "max"),
        "avg_sum_oracle_candidate_size": stat(candidate_values, "avg"),
        "median_sum_oracle_candidate_size": stat(candidate_values, "median"),
        "min_sum_oracle_candidate_size": stat(candidate_values, "min"),
        "max_sum_oracle_candidate_size": stat(candidate_values, "max"),
        "avg_num_gt_apis": stat(api_values, "avg"),
        "avg_num_gt_categories": stat(category_values, "avg"),
        "seen_only_count": seen_only,
        "seen_only_ratio": ratio(seen_only, count),
        "partially_unseen_count": partially_unseen,
        "partially_unseen_ratio": ratio(partially_unseen, count),
        "all_unseen_count": all_unseen,
        "all_unseen_ratio": ratio(all_unseen, count),
        "single_category_count": single_category,
        "single_category_ratio": ratio(single_category, count),
        "multi_category_count": multi_category,
        "multi_category_ratio": ratio(multi_category, count),
        "clear_requirement_count": clear,
        "clear_requirement_ratio": ratio(clear, count),
        "middle_requirement_count": middle,
        "middle_requirement_ratio": ratio(middle, count),
        "ambiguous_requirement_count": ambiguous,
        "ambiguous_requirement_ratio": ratio(ambiguous, count),
        "seen_api_combination_count": seen_api,
        "seen_api_combination_ratio": ratio(seen_api, count),
        "novel_api_combination_count": novel_api,
        "novel_api_combination_ratio": ratio(novel_api, count),
        "seen_category_combination_count": seen_category,
        "seen_category_combination_ratio": ratio(seen_category, count),
        "novel_category_combination_count": novel_category,
        "novel_category_combination_ratio": ratio(novel_category, count),
        "feasibility_label": label,
        "feasibility_note": note,
    }


def integrated_subsets(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    definitions = [
        (
            "InfoLimited_SmallChoice",
            "word_count_bin == short and oracle_candidate_size_bin == small",
            lambda row: row.get("length_bin_normalized") == "short" and row.get("choice_bin_normalized") == "small",
        ),
        (
            "InfoLimited_LargeChoice",
            "word_count_bin == short and oracle_candidate_size_bin in {medium, large}",
            lambda row: row.get("length_bin_normalized") == "short" and row.get("choice_bin_normalized") in {"medium", "large"},
        ),
        (
            "InfoRich_SmallChoice",
            "word_count_bin in {medium, long} and oracle_candidate_size_bin == small",
            lambda row: row.get("length_bin_normalized") in {"medium", "long"} and row.get("choice_bin_normalized") == "small",
        ),
        (
            "InfoRich_LargeChoice",
            "word_count_bin in {medium, long} and oracle_candidate_size_bin in {medium, large}",
            lambda row: row.get("length_bin_normalized") in {"medium", "long"} and row.get("choice_bin_normalized") in {"medium", "large"},
        ),
    ]
    return [(name, definition, [row for row in rows if predicate(row)]) for name, definition, predicate in definitions]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, NA)).replace("\n", " ") for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    total: int,
    cross_rows: list[dict[str, Any]],
    pivot_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    integrated_names = {
        "InfoLimited_SmallChoice",
        "InfoLimited_LargeChoice",
        "InfoRich_SmallChoice",
        "InfoRich_LargeChoice",
    }
    integrated = [row for row in summary_rows if row["subset_name"] in integrated_names]
    nine_way = [row for row in summary_rows if row["subset_name"] not in integrated_names]
    ready = [row["subset_name"] for row in summary_rows if row["feasibility_label"] == "main_or_comparison_ready"]
    auxiliary = [row["subset_name"] for row in summary_rows if row["feasibility_label"] == "auxiliary_ready"]
    diagnostic = [row["subset_name"] for row in summary_rows if row["feasibility_label"] == "diagnostic_only"]
    case_study = [row["subset_name"] for row in summary_rows if row["feasibility_label"] == "case_study_only"]

    info_limited_large = next((row for row in integrated if row["subset_name"] == "InfoLimited_LargeChoice"), None)
    info_rich_small = next((row for row in integrated if row["subset_name"] == "InfoRich_SmallChoice"), None)

    lines = [
        "# Length x Choice Test Subset Analysis",
        "",
        "## Overview",
        f"- total samples: {total}",
        "- document length bins: short=1-20 words, medium=21-50 words, long=51+ words.",
        "- oracle candidate size bins: small=1-20 APIs, medium=21-100 APIs, large=101+ APIs.",
        "- These axes separate input information volume from API-layer choice-set size after a correct category decision.",
        "",
        "## 3x3 Cross Table",
        markdown_table(cross_rows, ["word_count_bin", "oracle_candidate_size_bin", "count", "ratio"]),
        "",
        "### Pivot Counts",
        markdown_table(pivot_rows, ["word_count_bin", "small", "medium", "large", "total"]),
        "",
        "## 2x2 Integrated Subsets",
        markdown_table(integrated, ["subset_name", "condition_definition", "count", "ratio", "feasibility_label"]),
        "",
        "## Subset Statistics",
        markdown_table(
            summary_rows,
            [
                "subset_name",
                "count",
                "ratio",
                "avg_word_count",
                "avg_sum_oracle_candidate_size",
                "avg_num_gt_apis",
                "avg_num_gt_categories",
                "seen_only_ratio",
                "multi_category_ratio",
                "ambiguous_requirement_ratio",
                "feasibility_label",
            ],
        ),
        "",
        "## Feasibility",
        f"- main_or_comparison_ready: {', '.join(ready) if ready else 'none'}",
        f"- auxiliary_ready: {', '.join(auxiliary) if auxiliary else 'none'}",
        f"- diagnostic_only: {', '.join(diagnostic) if diagnostic else 'none'}",
        f"- case_study_only: {', '.join(case_study) if case_study else 'none'}",
        "",
        "## Recommended Usage",
        "- main experiment: keep using representative_with_length for balanced global evaluation.",
        "- auxiliary experiment: use 2x2 integrated subsets whose feasibility label is main_or_comparison_ready or auxiliary_ready.",
        "- diagnostic analysis: use 3x3 cells with at least 30 samples.",
        "- case study: use cells below 30 samples qualitatively instead of making stable F1 claims.",
        "",
        "## Hypotheses",
        "- Hypothesis 1: short documents with large choice sets may have lower F1 because little input information is paired with many possible APIs.",
        "- Hypothesis 2: rich documents with small choice sets may have higher F1 because more input information is paired with fewer possible APIs.",
        "- Hypothesis 3: length and choice-set size may change the relationship between E and F1; InfoLimited_LargeChoice may expose early commitment effects in the hierarchy.",
        "- These are hypotheses only; they should not be treated as conclusions until model evaluation is run.",
        "",
        "## Final Judgement Checks",
        f"- InfoLimited_LargeChoice count: {info_limited_large['count'] if info_limited_large else NA}",
        f"- InfoRich_SmallChoice count: {info_rich_small['count'] if info_rich_small else NA}",
        "- The 2x2 subsets are the practical comparison units when 3x3 cells are sparse.",
        "",
        "## Notes",
        f"- warnings: {len(warnings)}",
    ]
    if warnings:
        for warning in warnings[:20]:
            lines.append(f"- {warning}")
        if len(warnings) > 20:
            lines.append(f"- ... {len(warnings) - 20} more warnings omitted")
    else:
        lines.append("- no missing required columns were encountered.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create length x oracle-choice subset ID lists and summaries.")
    parser.add_argument("--analysis-dir", default=str(base.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-flags", default=None, help="Optional sample_condition_flags.csv path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    sample_flags = Path(args.sample_flags) if args.sample_flags else analysis_dir / "sample_condition_flags.csv"
    rows, fieldnames = read_csv_rows(sample_flags)
    warnings: list[str] = []
    required = [
        "sample_id",
        "word_count",
        "word_count_bin",
        "sum_oracle_category_candidate_size",
        "oracle_category_candidate_size_bin",
        "seen_status",
        "num_gt_apis",
        "num_gt_categories",
        "single_or_multi_category",
        "requirement_clarity_status",
        "api_combination_status",
        "category_combination_status",
    ]
    for column in required:
        if column not in fieldnames:
            warnings.append(f"missing column: {column}")

    prepared = prepare_rows(rows, warnings)
    total = len(prepared)
    cross_rows = build_cross_table(prepared)
    pivot_rows = build_pivot(cross_rows)
    write_csv(analysis_dir / "length_choice_cross_table.csv", ["word_count_bin", "oracle_candidate_size_bin", "count", "ratio"], cross_rows)
    write_csv(analysis_dir / "length_choice_cross_table_pivot.csv", ["word_count_bin", "small", "medium", "large", "total"], pivot_rows)

    subset_dir = analysis_dir / "subset_candidates" / "length_choice"
    summary_rows: list[dict[str, Any]] = []
    for length_bin in LENGTH_ORDER:
        for choice_bin in CHOICE_ORDER:
            name = subset_name(length_bin, choice_bin)
            subset = cell_rows(prepared, length_bin, choice_bin)
            write_ids(subset_dir / f"{name}_ids.txt", subset)
            summary_rows.append(subset_summary_row(name, condition_text_for_cell(length_bin, choice_bin), subset, total))

    integrated_filenames = {
        "InfoLimited_SmallChoice": "info_limited_small_choice_ids.txt",
        "InfoLimited_LargeChoice": "info_limited_large_choice_ids.txt",
        "InfoRich_SmallChoice": "info_rich_small_choice_ids.txt",
        "InfoRich_LargeChoice": "info_rich_large_choice_ids.txt",
    }
    for name, definition, subset in integrated_subsets(prepared):
        filename = integrated_filenames[name]
        write_ids(subset_dir / filename, subset)
        summary_rows.append(subset_summary_row(name, definition, subset, total))

    write_csv(analysis_dir / "length_choice_subset_summary.csv", SUMMARY_FIELDS, summary_rows)
    report = build_report(total, cross_rows, pivot_rows, summary_rows, warnings)
    (analysis_dir / "length_choice_subset_report.md").write_text(report, encoding="utf-8")

    metadata_path = analysis_dir / "analysis_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata["length_choice_subset_analysis"] = {
        "script": str(Path(__file__).resolve()),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_flags": str(sample_flags),
        "total_samples": total,
        "warnings": warnings,
        "outputs": [
            "length_choice_cross_table.csv",
            "length_choice_cross_table_pivot.csv",
            "length_choice_subset_summary.csv",
            "length_choice_subset_report.md",
            "subset_candidates/length_choice/",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded samples: {total}")
    print(f"Cross table: {analysis_dir / 'length_choice_cross_table.csv'}")
    print(f"Subset summary: {analysis_dir / 'length_choice_subset_summary.csv'}")
    print(f"Report: {analysis_dir / 'length_choice_subset_report.md'}")
    print(f"ID lists: {subset_dir}")


if __name__ == "__main__":
    main()
