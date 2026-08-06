#!/usr/bin/env python3
"""Create concrete test-data JSON subsets by document length and oracle choice size."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
import create_length_choice_subsets as lc

NA = base.NA
DEFAULT_ANALYSIS_DIR = base.DEFAULT_OUTPUT_DIR
DEFAULT_TEST_PATH = base.DEFAULT_TEST_PATH

SUMMARY_FIELDS = [
    "subset_group",
    "subset_name",
    "condition_definition",
    "count",
    "ratio",
    "output_data_file",
    "output_id_file",
    "feasibility_label",
    "notes",
]

STAT_FIELDS = [
    "subset_group",
    "subset_name",
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
    "seen_only_ratio",
    "multi_category_ratio",
    "ambiguous_requirement_ratio",
    "novel_api_combination_ratio",
    "novel_category_combination_ratio",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_if_exists(path: Path, backups: list[str]) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.name}.bak_{timestamp()}")
    shutil.copy2(path, backup)
    backups.append(f"{path} -> {backup}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]], backups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path, backups)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ids(path: Path, rows: list[dict[str, Any]], backups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path, backups)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row.get('sample_id', '')}\n")


def load_sample_flags(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows, fieldnames = lc.read_csv_rows(path)
    warnings: list[str] = []
    required = [
        "sample_id",
        "word_count",
        "word_count_bin",
        "sum_oracle_category_candidate_size",
        "oracle_category_candidate_size_bin",
    ]
    for column in required:
        if column not in fieldnames:
            warnings.append(f"missing column in sample_condition_flags.csv: {column}")
    prepared = lc.prepare_rows(rows, warnings)
    return prepared, fieldnames, warnings


def load_original_data(path: Path) -> tuple[Any, str, dict[str, Any], list[str]]:
    data = base.load_json(path)
    warnings: list[str] = []
    if isinstance(data, dict):
        return data, "dict", {str(key): value for key, value in data.items()}, list(str(key) for key in data.keys())
    if isinstance(data, list):
        records = base.load_mashup_records(path)
        mapping: dict[str, Any] = {}
        order: list[str] = []
        for record, raw in zip(records, data):
            mapping[record.sample_id] = raw
            order.append(record.sample_id)
        return data, "list", mapping, order
    warnings.append(f"unsupported original JSON root type: {type(data).__name__}")
    return data, type(data).__name__, {}, []


def subset_original_data(rows: list[dict[str, Any]], original_format: str, original_map: dict[str, Any]) -> tuple[Any, list[str]]:
    missing: list[str] = []
    if original_format == "dict":
        out: dict[str, Any] = {}
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            if sample_id in original_map:
                out[sample_id] = original_map[sample_id]
            else:
                missing.append(sample_id)
        return out, missing
    if original_format == "list":
        out_list: list[Any] = []
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            if sample_id in original_map:
                out_list.append(original_map[sample_id])
            else:
                missing.append(sample_id)
        return out_list, missing
    return [], [str(row.get("sample_id", "")) for row in rows]


def write_json(path: Path, payload: Any, backups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path, backups)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def as_float(value: Any) -> float | None:
    return lc.as_float(value)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return lc.numeric_values(rows, key)


def stat(values: list[float], kind: str) -> str:
    return lc.stat(values, kind)


def ratio(count: int, total: int) -> str:
    return base.ratio(count, total)


def count_status(rows: list[dict[str, Any]], key: str, expected: str) -> int:
    return lc.count_status(rows, key, expected)


def count_not_status(rows: list[dict[str, Any]], key: str, expected: str) -> int:
    return lc.count_not_status(rows, key, expected)


def statistics_row(group: str, name: str, rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    count = len(rows)
    word_values = numeric_values(rows, "word_count")
    candidate_values = numeric_values(rows, "sum_oracle_category_candidate_size")
    api_values = numeric_values(rows, "num_gt_apis")
    category_values = numeric_values(rows, "num_gt_categories")
    seen_only = count_status(rows, "seen_status", "Seen-only")
    multi_category = count_status(rows, "single_or_multi_category", "Multi-category")
    ambiguous = count_status(rows, "requirement_clarity_status", "Ambiguous-requirement")
    seen_api = count_status(rows, "api_combination_status", "api_combination_exact_seen")
    seen_category = count_status(rows, "category_combination_status", "category_combination_exact_seen")
    novel_api = count - seen_api if count else 0
    novel_category = count - seen_category if count else 0
    return {
        "subset_group": group,
        "subset_name": name,
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
        "seen_only_ratio": ratio(seen_only, count),
        "multi_category_ratio": ratio(multi_category, count),
        "ambiguous_requirement_ratio": ratio(ambiguous, count),
        "novel_api_combination_ratio": ratio(novel_api, count),
        "novel_category_combination_ratio": ratio(novel_category, count),
    }


def subset_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for length_bin in lc.LENGTH_ORDER:
        specs.append(
            {
                "group": "by_length",
                "name": length_bin,
                "definition": f"word_count_bin == {length_bin}",
                "directory": "by_length",
                "data_file": f"{length_bin}_test.json",
                "id_file": f"{length_bin}_ids.txt",
                "predicate": lambda row, b=length_bin: row.get("length_bin_normalized") == b,
            }
        )
    for choice_bin in lc.CHOICE_ORDER:
        specs.append(
            {
                "group": "by_choice",
                "name": f"{choice_bin}_choice",
                "definition": f"oracle_candidate_size_bin == {choice_bin}",
                "directory": "by_choice",
                "data_file": f"{choice_bin}_choice_test.json",
                "id_file": f"{choice_bin}_choice_ids.txt",
                "predicate": lambda row, b=choice_bin: row.get("choice_bin_normalized") == b,
            }
        )
    for length_bin in lc.LENGTH_ORDER:
        for choice_bin in lc.CHOICE_ORDER:
            name = f"{length_bin}_{choice_bin}"
            specs.append(
                {
                    "group": "by_length_choice_3x3",
                    "name": name,
                    "definition": f"word_count_bin == {length_bin} and oracle_candidate_size_bin == {choice_bin}",
                    "directory": "by_length_choice_3x3",
                    "data_file": f"{name}_test.json",
                    "id_file": f"{name}_ids.txt",
                    "predicate": lambda row, lb=length_bin, cb=choice_bin: row.get("length_bin_normalized") == lb and row.get("choice_bin_normalized") == cb,
                }
            )
    specs.extend(
        [
            {
                "group": "by_length_choice_2x2",
                "name": "info_limited_small_choice",
                "definition": "word_count_bin == short and oracle_candidate_size_bin == small",
                "directory": "by_length_choice_2x2",
                "data_file": "info_limited_small_choice_test.json",
                "id_file": "info_limited_small_choice_ids.txt",
                "predicate": lambda row: row.get("length_bin_normalized") == "short" and row.get("choice_bin_normalized") == "small",
            },
            {
                "group": "by_length_choice_2x2",
                "name": "info_limited_large_choice",
                "definition": "word_count_bin == short and oracle_candidate_size_bin in {medium, large}",
                "directory": "by_length_choice_2x2",
                "data_file": "info_limited_large_choice_test.json",
                "id_file": "info_limited_large_choice_ids.txt",
                "predicate": lambda row: row.get("length_bin_normalized") == "short" and row.get("choice_bin_normalized") in {"medium", "large"},
            },
            {
                "group": "by_length_choice_2x2",
                "name": "info_rich_small_choice",
                "definition": "word_count_bin in {medium, long} and oracle_candidate_size_bin == small",
                "directory": "by_length_choice_2x2",
                "data_file": "info_rich_small_choice_test.json",
                "id_file": "info_rich_small_choice_ids.txt",
                "predicate": lambda row: row.get("length_bin_normalized") in {"medium", "long"} and row.get("choice_bin_normalized") == "small",
            },
            {
                "group": "by_length_choice_2x2",
                "name": "info_rich_large_choice",
                "definition": "word_count_bin in {medium, long} and oracle_candidate_size_bin in {medium, large}",
                "directory": "by_length_choice_2x2",
                "data_file": "info_rich_large_choice_test.json",
                "id_file": "info_rich_large_choice_ids.txt",
                "predicate": lambda row: row.get("length_bin_normalized") in {"medium", "long"} and row.get("choice_bin_normalized") in {"medium", "large"},
            },
        ]
    )
    return specs


def feasibility(count: int) -> tuple[str, str]:
    return lc.feasibility(count)


def expected_counts() -> dict[str, dict[str, int]]:
    return {
        "by_length": {"short": 128, "medium": 177, "long": 31},
        "by_choice": {"small_choice": 49, "medium_choice": 253, "large_choice": 34},
        "by_length_choice_3x3": {
            "short_small": 17,
            "short_medium": 100,
            "short_large": 11,
            "medium_small": 26,
            "medium_medium": 132,
            "medium_large": 19,
            "long_small": 6,
            "long_medium": 21,
            "long_large": 4,
        },
        "by_length_choice_2x2": {
            "info_limited_small_choice": 17,
            "info_limited_large_choice": 111,
            "info_rich_small_choice": 32,
            "info_rich_large_choice": 176,
        },
    }


def validate_group(summary_rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        rows_by_group.setdefault(str(row["subset_group"]), []).append(row)
    expected = expected_counts()
    validation: list[dict[str, Any]] = []
    for group, rows in rows_by_group.items():
        count_sum = sum(int(row["count"]) for row in rows)
        expected_total = total
        sum_ok = count_sum == expected_total
        expected_for_group = expected.get(group, {})
        expected_ok = all(int(row["count"]) == expected_for_group.get(str(row["subset_name"]), int(row["count"])) for row in rows)
        validation.append(
            {
                "group": group,
                "count_sum": count_sum,
                "expected_total": expected_total,
                "sum_ok": "yes" if sum_ok else "no",
                "expected_counts_ok": "yes" if expected_ok else "no",
            }
        )
    return validation


def duplicate_check(summary_rows: list[dict[str, Any]], rows_by_subset: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in ["by_length", "by_choice", "by_length_choice_3x3", "by_length_choice_2x2"]:
        group_rows = [row for row in summary_rows if row["subset_group"] == group]
        seen: Counter[str] = Counter()
        for row in group_rows:
            key = f"{group}/{row['subset_name']}"
            seen.update(str(item.get("sample_id", "")) for item in rows_by_subset.get(key, []))
        duplicate_ids = sorted(sample_id for sample_id, count in seen.items() if count > 1)
        output.append(
            {
                "group": group,
                "unique_count": len(seen),
                "duplicate_count": len(duplicate_ids),
                "duplicates_ok": "yes" if not duplicate_ids else "no",
            }
        )
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, NA)).replace("\n", " ") for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    total: int,
    summary_rows: list[dict[str, Any]],
    stat_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    missing_ids: list[str],
    warnings: list[str],
    backups: list[str],
    original_format: str,
) -> str:
    def group_rows(group: str) -> list[dict[str, Any]]:
        return [row for row in summary_rows if row["subset_group"] == group]

    ready = [row for row in summary_rows if row["feasibility_label"] == "main_or_comparison_ready"]
    auxiliary = [row for row in summary_rows if row["feasibility_label"] == "auxiliary_ready"]
    diagnostic = [row for row in summary_rows if row["feasibility_label"] == "diagnostic_only"]
    case_study = [row for row in summary_rows if row["feasibility_label"] == "case_study_only"]

    lines = [
        "# Test Subset Creation Report",
        "",
        "## Overview",
        f"- total samples: {total}",
        "- Created concrete test-data files for document length, choice size, 3x3 length x choice, and 2x2 integrated length x choice splits.",
        "- No model training, inference, or F1 calculation was performed.",
        f"- original data format: {original_format}",
        "",
        "## Bin Definitions",
        "- Document length: short = 1-20 words, medium = 21-50 words, long = 51+ words.",
        "- Choice size: small = 1-20 candidate APIs, medium = 21-100 candidate APIs, large = 101+ candidate APIs.",
        "- Choice size is based on sum_oracle_category_candidate_size, meaning the number of APIs remaining after assuming the correct category path.",
        "- 2x2 length groups: InfoLimited = short; InfoRich = medium or long.",
        "- 2x2 choice groups: SmallChoice = small; LargeChoice = medium or large.",
        "",
        "## By Document Length",
        markdown_table(group_rows("by_length"), ["subset_name", "count", "ratio", "output_data_file", "feasibility_label"]),
        "",
        "## By Choice Size",
        markdown_table(group_rows("by_choice"), ["subset_name", "count", "ratio", "output_data_file", "feasibility_label"]),
        "",
        "## By Length x Choice, 3x3",
        markdown_table(group_rows("by_length_choice_3x3"), ["subset_name", "count", "ratio", "output_data_file", "feasibility_label"]),
        "",
        "## By Length x Choice, 2x2",
        markdown_table(group_rows("by_length_choice_2x2"), ["subset_name", "count", "ratio", "output_data_file", "feasibility_label"]),
        "",
        "## Validation",
        markdown_table(validation_rows, ["group", "count_sum", "expected_total", "sum_ok", "expected_counts_ok"]),
        "",
        "### Duplicate Checks",
        markdown_table(duplicate_rows, ["group", "unique_count", "duplicate_count", "duplicates_ok"]),
        f"- missing IDs in original data: {len(missing_ids)}",
        "",
        "## Recommended Usage",
        "- Main experiment: use globally representative subsets such as representative_with_length; these condition splits are mainly for auxiliary and diagnostic analysis.",
        "- By document length: short_test.json and medium_test.json are comparison-ready; long_test.json is diagnostic/case-study oriented.",
        "- By choice size: medium_choice_test.json is comparison-ready; small_choice_test.json and large_choice_test.json are diagnostic-oriented.",
        "- 3x3 split: short_medium_test.json and medium_medium_test.json are the most comparison-ready; sparse cells should be used diagnostically or qualitatively.",
        "- 2x2 split: info_limited_large_choice_test.json and info_rich_large_choice_test.json are comparison-ready; info_limited_small_choice_test.json is case-study only and info_rich_small_choice_test.json is diagnostic-only.",
        "",
        "## Feasibility Summary",
        f"- main_or_comparison_ready: {', '.join(row['subset_name'] for row in ready) if ready else 'none'}",
        f"- auxiliary_ready: {', '.join(row['subset_name'] for row in auxiliary) if auxiliary else 'none'}",
        f"- diagnostic_only: {', '.join(row['subset_name'] for row in diagnostic) if diagnostic else 'none'}",
        f"- case_study_only: {', '.join(row['subset_name'] for row in case_study) if case_study else 'none'}",
        "",
        "## Notes",
        f"- warnings: {len(warnings)}",
        f"- backups created: {len(backups)}",
    ]
    for warning in warnings[:30]:
        lines.append(f"- warning: {warning}")
    if len(warnings) > 30:
        lines.append(f"- ... {len(warnings) - 30} more warnings omitted")
    if missing_ids:
        lines.append(f"- missing sample IDs: {', '.join(missing_ids[:30])}")
        if len(missing_ids) > 30:
            lines.append(f"- ... {len(missing_ids) - 30} more missing IDs omitted")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create concrete test JSON subsets by length and oracle choice size.")
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--sample-flags", default=None)
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    sample_flags = Path(args.sample_flags) if args.sample_flags else analysis_dir / "sample_condition_flags.csv"
    test_path = Path(args.test_path)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "test_subsets"
    reports_dir = output_dir / "reports"
    warnings: list[str] = []
    backups: list[str] = []

    rows, _, flag_warnings = load_sample_flags(sample_flags)
    warnings.extend(flag_warnings)
    total = len(rows)
    _, original_format, original_map, original_order = load_original_data(test_path)
    if not original_map:
        warnings.append(f"no original data mapping could be built from {test_path}")

    original_ids = set(original_map)
    flag_ids = [str(row.get("sample_id", "")) for row in rows]
    missing_ids = [sample_id for sample_id in flag_ids if sample_id not in original_ids]
    for sample_id in missing_ids:
        warnings.append(f"sample_id not found in original test data: {sample_id}")

    summary_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    rows_by_subset: dict[str, list[dict[str, Any]]] = {}

    for spec in subset_specs():
        subset = [row for row in rows if spec["predicate"](row)]
        subset.sort(key=lambda row: int(row.get("_original_index", 0)))
        subset_key = f"{spec['group']}/{spec['name']}"
        rows_by_subset[subset_key] = subset
        data_payload, subset_missing = subset_original_data(subset, original_format, original_map)
        if subset_missing:
            for sample_id in subset_missing:
                warnings.append(f"{subset_key}: sample_id not found in original test data: {sample_id}")
        subdir = output_dir / str(spec["directory"])
        data_path = subdir / str(spec["data_file"])
        id_path = subdir / str(spec["id_file"])
        write_json(data_path, data_payload, backups)
        write_ids(id_path, subset, backups)
        label, note = feasibility(len(subset))
        summary_rows.append(
            {
                "subset_group": spec["group"],
                "subset_name": spec["name"],
                "condition_definition": spec["definition"],
                "count": len(subset),
                "ratio": ratio(len(subset), total),
                "output_data_file": str(data_path.relative_to(output_dir)),
                "output_id_file": str(id_path.relative_to(output_dir)),
                "feasibility_label": label,
                "notes": note,
            }
        )
        stat_rows.append(statistics_row(spec["group"], spec["name"], subset, total))

    validation_rows = validate_group(summary_rows, total)
    duplicate_rows = duplicate_check(summary_rows, rows_by_subset)

    write_csv(reports_dir / "test_subset_creation_summary.csv", SUMMARY_FIELDS, summary_rows, backups)
    write_csv(reports_dir / "test_subset_statistics.csv", STAT_FIELDS, stat_rows, backups)
    report = build_report(total, summary_rows, stat_rows, validation_rows, duplicate_rows, missing_ids, warnings, backups, original_format)
    report_path = reports_dir / "test_subset_creation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(report_path, backups)
    report_path.write_text(report, encoding="utf-8")

    metadata_path = analysis_dir / "analysis_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata["condition_test_subset_creation"] = {
        "script": str(Path(__file__).resolve()),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_flags": str(sample_flags),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "original_format": original_format,
        "total_samples": total,
        "missing_id_count": len(missing_ids),
        "warnings": warnings,
        "backups_created": backups,
    }
    backup_if_exists(metadata_path, backups)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded sample flags: {total}")
    print(f"Original test data format: {original_format}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {reports_dir / 'test_subset_creation_summary.csv'}")
    print(f"Statistics: {reports_dir / 'test_subset_statistics.csv'}")
    print(f"Report: {report_path}")
    if warnings:
        print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
