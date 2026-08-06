#!/usr/bin/env python3
"""Create condition-analysis test subsets from the existing 336 test samples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FLAGS = PROJECT_ROOT / "analysis_test_subset_conditions" / "sample_condition_flags.csv"
DEFAULT_TEST_DATA = (
    PROJECT_ROOT
    / "output"
    / "layer1_direct_api"
    / "test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_subsets" / "condition_analysis"

EXPECTED_COUNTS = {
    "overall": 336,
    "shorter_desc": 171,
    "longer_desc": 165,
    "fewer_candidates": 195,
    "more_candidates": 141,
    "shorter_desc_fewer_candidates": 107,
    "shorter_desc_more_candidates": 64,
    "longer_desc_fewer_candidates": 88,
    "longer_desc_more_candidates": 77,
}

REQUIRED_COLUMNS = {
    "sample_id",
    "word_count",
    "sum_oracle_category_candidate_size",
}


@dataclass(frozen=True)
class SubsetSpec:
    group: str
    name: str
    relative_path: Path
    condition: str
    predicate: Callable[[dict[str, Any]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the median-based condition-analysis JSON subsets."
    )
    parser.add_argument("--sample-flags", default=str(DEFAULT_SAMPLE_FLAGS))
    parser.add_argument("--test-data", default=str(DEFAULT_TEST_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak_{timestamp()}")
    shutil.copy2(path, backup)
    return f"{path} -> {backup}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def to_int(value: Any, *, field: str, sample_id: str) -> int:
    text = str(value).strip()
    if not text or text.upper() == "NA":
        raise ValueError(f"sample_id={sample_id}: missing {field}")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"sample_id={sample_id}: invalid {field}: {text}") from exc
    if not number.is_integer():
        raise ValueError(f"sample_id={sample_id}: non-integer {field}: {text}")
    return int(number)


def prepare_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, dict[str, Any]]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing_columns:
        raise ValueError(f"missing required columns in sample_condition_flags.csv: {missing_columns}")

    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(row.get("sample_id", "") for row in rows).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate sample_id values in sample_condition_flags.csv: {duplicate_ids[:20]}")

    prepared: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"row {index + 2}: missing sample_id")
        out: dict[str, Any] = dict(row)
        out["_csv_index"] = index
        out["_word_count_int"] = to_int(row.get("word_count"), field="word_count", sample_id=sample_id)
        out["_candidate_size_int"] = to_int(
            row.get("sum_oracle_category_candidate_size"),
            field="sum_oracle_category_candidate_size",
            sample_id=sample_id,
        )
        prepared[sample_id] = out
    return prepared


def original_mapping(data: Any) -> tuple[str, list[str], dict[str, Any]]:
    if isinstance(data, dict):
        ids = [str(key) for key in data.keys()]
        return "dict", ids, {str(key): value for key, value in data.items()}
    raise ValueError(
        "unsupported original JSON root type. This script expects the existing test data to be a JSON object keyed by sample_id."
    )


def subset_specs() -> list[SubsetSpec]:
    return [
        SubsetSpec(
            "overall",
            "overall",
            Path("overall.json"),
            "all 336 existing test samples",
            lambda row: True,
        ),
        SubsetSpec(
            "by_length",
            "shorter_desc",
            Path("by_length") / "shorter_desc.json",
            "word_count <= 24",
            lambda row: row["_word_count_int"] <= 24,
        ),
        SubsetSpec(
            "by_length",
            "longer_desc",
            Path("by_length") / "longer_desc.json",
            "word_count >= 25",
            lambda row: row["_word_count_int"] >= 25,
        ),
        SubsetSpec(
            "by_candidate_size",
            "fewer_candidates",
            Path("by_candidate_size") / "fewer_candidates.json",
            "sum_oracle_category_candidate_size <= 52",
            lambda row: row["_candidate_size_int"] <= 52,
        ),
        SubsetSpec(
            "by_candidate_size",
            "more_candidates",
            Path("by_candidate_size") / "more_candidates.json",
            "sum_oracle_category_candidate_size >= 53",
            lambda row: row["_candidate_size_int"] >= 53,
        ),
        SubsetSpec(
            "by_length_candidate_2x2",
            "shorter_desc_fewer_candidates",
            Path("by_length_candidate_2x2") / "shorter_desc_fewer_candidates.json",
            "word_count <= 24 and sum_oracle_category_candidate_size <= 52",
            lambda row: row["_word_count_int"] <= 24 and row["_candidate_size_int"] <= 52,
        ),
        SubsetSpec(
            "by_length_candidate_2x2",
            "shorter_desc_more_candidates",
            Path("by_length_candidate_2x2") / "shorter_desc_more_candidates.json",
            "word_count <= 24 and sum_oracle_category_candidate_size >= 53",
            lambda row: row["_word_count_int"] <= 24 and row["_candidate_size_int"] >= 53,
        ),
        SubsetSpec(
            "by_length_candidate_2x2",
            "longer_desc_fewer_candidates",
            Path("by_length_candidate_2x2") / "longer_desc_fewer_candidates.json",
            "word_count >= 25 and sum_oracle_category_candidate_size <= 52",
            lambda row: row["_word_count_int"] >= 25 and row["_candidate_size_int"] <= 52,
        ),
        SubsetSpec(
            "by_length_candidate_2x2",
            "longer_desc_more_candidates",
            Path("by_length_candidate_2x2") / "longer_desc_more_candidates.json",
            "word_count >= 25 and sum_oracle_category_candidate_size >= 53",
            lambda row: row["_word_count_int"] >= 25 and row["_candidate_size_int"] >= 53,
        ),
    ]


def verify_input_ids(original_ids: list[str], rows_by_id: dict[str, dict[str, Any]]) -> None:
    original_set = set(original_ids)
    csv_set = set(rows_by_id)
    missing_in_json = sorted(csv_set - original_set)
    missing_in_csv = sorted(original_set - csv_set)
    if missing_in_json or missing_in_csv:
        details = [
            "sample_id mismatch between original JSON and sample_condition_flags.csv.",
            f"sample IDs in CSV but not JSON: {len(missing_in_json)}",
            f"sample IDs in JSON but not CSV: {len(missing_in_csv)}",
        ]
        if missing_in_json:
            details.append(f"first CSV-only IDs: {missing_in_json[:20]}")
        if missing_in_csv:
            details.append(f"first JSON-only IDs: {missing_in_csv[:20]}")
        raise ValueError("\n".join(details))


def selected_ids_for(
    spec: SubsetSpec,
    original_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    return [sample_id for sample_id in original_ids if spec.predicate(rows_by_id[sample_id])]


def validate_counts(selected_ids: dict[str, list[str]]) -> None:
    mismatches = []
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(selected_ids.get(name, []))
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, actual {actual}")
    if mismatches:
        raise ValueError("subset count validation failed:\n" + "\n".join(mismatches))


def validate_axis(
    group: str,
    names: list[str],
    selected_ids: dict[str, list[str]],
    original_ids: list[str],
) -> dict[str, Any]:
    total = len(original_ids)
    original_set = set(original_ids)
    seen: Counter[str] = Counter()
    for name in names:
        seen.update(selected_ids[name])
    duplicate_ids = sorted(sample_id for sample_id, count in seen.items() if count > 1)
    missing_ids = sorted(original_set - set(seen))
    outside_ids = sorted(set(seen) - original_set)
    return {
        "group": group,
        "count_sum": sum(len(selected_ids[name]) for name in names),
        "expected_total": total,
        "unique_count": len(seen),
        "duplicate_count": len(duplicate_ids),
        "missing_count": len(missing_ids),
        "outside_original_count": len(outside_ids),
        "ok": not duplicate_ids
        and not missing_ids
        and not outside_ids
        and sum(len(selected_ids[name]) for name in names) == total,
    }


def validate_axes(selected_ids: dict[str, list[str]], original_ids: list[str]) -> list[dict[str, Any]]:
    return [
        validate_axis("by_length", ["shorter_desc", "longer_desc"], selected_ids, original_ids),
        validate_axis(
            "by_candidate_size",
            ["fewer_candidates", "more_candidates"],
            selected_ids,
            original_ids,
        ),
        validate_axis(
            "by_length_candidate_2x2",
            [
                "shorter_desc_fewer_candidates",
                "shorter_desc_more_candidates",
                "longer_desc_fewer_candidates",
                "longer_desc_more_candidates",
            ],
            selected_ids,
            original_ids,
        ),
    ]


def validate_relationships(selected_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    checks = [
        (
            "shorter_desc = shorter_desc_fewer_candidates + shorter_desc_more_candidates",
            len(selected_ids["shorter_desc"]),
            len(selected_ids["shorter_desc_fewer_candidates"]) + len(selected_ids["shorter_desc_more_candidates"]),
        ),
        (
            "longer_desc = longer_desc_fewer_candidates + longer_desc_more_candidates",
            len(selected_ids["longer_desc"]),
            len(selected_ids["longer_desc_fewer_candidates"]) + len(selected_ids["longer_desc_more_candidates"]),
        ),
        (
            "fewer_candidates = shorter_desc_fewer_candidates + longer_desc_fewer_candidates",
            len(selected_ids["fewer_candidates"]),
            len(selected_ids["shorter_desc_fewer_candidates"]) + len(selected_ids["longer_desc_fewer_candidates"]),
        ),
        (
            "more_candidates = shorter_desc_more_candidates + longer_desc_more_candidates",
            len(selected_ids["more_candidates"]),
            len(selected_ids["shorter_desc_more_candidates"]) + len(selected_ids["longer_desc_more_candidates"]),
        ),
        (
            "overall = shorter_desc + longer_desc",
            len(selected_ids["overall"]),
            len(selected_ids["shorter_desc"]) + len(selected_ids["longer_desc"]),
        ),
        (
            "overall = fewer_candidates + more_candidates",
            len(selected_ids["overall"]),
            len(selected_ids["fewer_candidates"]) + len(selected_ids["more_candidates"]),
        ),
    ]
    return [
        {
            "relationship": relationship,
            "left_count": left,
            "right_count": right,
            "ok": left == right,
        }
        for relationship, left, right in checks
    ]


def ensure_valid_axes(axis_rows: list[dict[str, Any]], relationship_rows: list[dict[str, Any]]) -> None:
    failures = [row["group"] for row in axis_rows if not row["ok"]]
    failures.extend(row["relationship"] for row in relationship_rows if not row["ok"])
    if failures:
        raise ValueError("axis validation failed:\n" + "\n".join(str(item) for item in failures))


def subset_payload(ids: list[str], original_map: dict[str, Any]) -> dict[str, Any]:
    return {sample_id: original_map[sample_id] for sample_id in ids}


def write_json(path: Path, payload: Any, backups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_if_exists(path)
    if backup:
        backups.append(backup)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    *,
    sample_flags: Path,
    test_data: Path,
    output_dir: Path,
    original_format: str,
    selected_ids: dict[str, list[str]],
    specs: list[SubsetSpec],
    axis_rows: list[dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
    backups: list[str],
) -> str:
    subset_rows = []
    specs_by_name = {spec.name: spec for spec in specs}
    for name, expected in EXPECTED_COUNTS.items():
        spec = specs_by_name[name]
        count = len(selected_ids[name])
        subset_rows.append(
            {
                "subset": name,
                "condition": spec.condition,
                "count": count,
                "expected": expected,
                "match": "yes" if count == expected else "no",
                "file": str(spec.relative_path),
            }
        )

    axis_display_rows = [
        {
            **row,
            "ok": "yes" if row["ok"] else "no",
        }
        for row in axis_rows
    ]
    relationship_display_rows = [
        {
            **row,
            "ok": "yes" if row["ok"] else "no",
        }
        for row in relationship_rows
    ]

    lines = [
        "# Condition Analysis Test Subset Creation Report",
        "",
        f"作成日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 使用した入力ファイル",
        "",
        f"- 元テストデータ: `{test_data}`",
        f"- 分析CSV: `{sample_flags}`",
        f"- 元JSON形式: `{original_format}`",
        "",
        "## 2. 使用した指標",
        "",
        "- 説明文長: `word_count`",
        "- API候補数: `sum_oracle_category_candidate_size`",
        "",
        "## 3. 使用した閾値",
        "",
        "- `shorter_desc`: `word_count <= 24`",
        "- `longer_desc`: `word_count >= 25`",
        "- `fewer_candidates`: `sum_oracle_category_candidate_size <= 52`",
        "- `more_candidates`: `sum_oracle_category_candidate_size >= 53`",
        "",
        "## 4. 作成したサブセット一覧",
        "",
        markdown_table(subset_rows, ["subset", "condition", "count", "expected", "match", "file"]),
        "",
        "## 5. 件数検証",
        "",
        "- 期待件数との一致: yes",
        "- `overall`: 336",
        "- `shorter_desc + longer_desc = 336`",
        "- `fewer_candidates + more_candidates = 336`",
        "- 2x2 の4セル合計 = 336",
        "",
        "## 6. 分類軸ごとの重複・欠落・元データ外ID確認",
        "",
        markdown_table(
            axis_display_rows,
            [
                "group",
                "count_sum",
                "expected_total",
                "unique_count",
                "duplicate_count",
                "missing_count",
                "outside_original_count",
                "ok",
            ],
        ),
        "",
        "## 7. 合算関係の確認",
        "",
        markdown_table(relationship_display_rows, ["relationship", "left_count", "right_count", "ok"]),
        "",
        "## 8. 注意事項",
        "",
        "- 今回のサブセットは、新しいtrain/test splitではなく、既存テストデータ336件を条件別に再集計するためのものである。",
        "- `shorter_desc` / `longer_desc` は、テストデータ内での相対的な説明文長を表す。",
        "- `fewer_candidates` / `more_candidates` は、テストデータ内での相対的なAPI候補数を表す。",
        "- `sum_oracle_category_candidate_size` は、正解カテゴリパスが分かっていると仮定した後に残るAPI候補数であり、カテゴリ予測自体の難しさを直接表すものではない。",
        "- 単体サブセットと2x2サブセットは独立したデータではなく、同じ336件を異なる観点で集計したものである。",
        "",
        "## 9. 出力ディレクトリ",
        "",
        f"- `{output_dir}`",
        "",
        "## 10. 実行コマンドの記録",
        "",
        "```bash",
        "python3 scripts/create_condition_analysis_subsets.py",
        "```",
        "",
        f"- スクリプト保存場所: `{Path(__file__).resolve()}`",
        f"- バックアップ作成数: {len(backups)}",
    ]
    if backups:
        lines.extend(["", "## 11. バックアップ", ""])
        lines.extend(f"- `{backup}`" for backup in backups)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    sample_flags = Path(args.sample_flags).resolve()
    test_data = Path(args.test_data).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "reports" / "subset_creation_report.md"

    try:
        csv_rows, fieldnames = read_csv_rows(sample_flags)
        rows_by_id = prepare_rows(csv_rows, fieldnames)
        original_data = load_json(test_data)
        original_format, original_ids, original_map = original_mapping(original_data)
        verify_input_ids(original_ids, rows_by_id)

        specs = subset_specs()
        selected_ids = {
            spec.name: selected_ids_for(spec, original_ids, rows_by_id)
            for spec in specs
        }
        validate_counts(selected_ids)
        axis_rows = validate_axes(selected_ids, original_ids)
        relationship_rows = validate_relationships(selected_ids)
        ensure_valid_axes(axis_rows, relationship_rows)

        backups: list[str] = []
        for spec in specs:
            write_json(output_dir / spec.relative_path, subset_payload(selected_ids[spec.name], original_map), backups)

        report = build_report(
            sample_flags=sample_flags,
            test_data=test_data,
            output_dir=output_dir,
            original_format=original_format,
            selected_ids=selected_ids,
            specs=specs,
            axis_rows=axis_rows,
            relationship_rows=relationship_rows,
            backups=backups,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        backup = backup_if_exists(report_path)
        if backup:
            backups.append(backup)
            report = build_report(
                sample_flags=sample_flags,
                test_data=test_data,
                output_dir=output_dir,
                original_format=original_format,
                selected_ids=selected_ids,
                specs=specs,
                axis_rows=axis_rows,
                relationship_rows=relationship_rows,
                backups=backups,
            )
        report_path.write_text(report, encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Created condition-analysis subsets in: {output_dir}")
    for name in EXPECTED_COUNTS:
        print(f"{name}: {len(selected_ids[name])}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
