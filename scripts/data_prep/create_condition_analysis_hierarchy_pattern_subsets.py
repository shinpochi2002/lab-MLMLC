#!/usr/bin/env python3
"""Apply condition-analysis sample splits to hierarchy-pattern test data."""

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
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDITION_DIR = PROJECT_ROOT / "test_subsets" / "condition_analysis"
DEFAULT_PATTERN_DIR = PROJECT_ROOT / "output" / "by_pattern"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_subsets" / "condition_analysis_by_pattern"

SUBSETS = [
    ("overall", Path("overall.json"), Path("overall.json")),
    ("shorter_desc", Path("by_length") / "shorter_desc.json", Path("by_length") / "shorter_desc.json"),
    ("longer_desc", Path("by_length") / "longer_desc.json", Path("by_length") / "longer_desc.json"),
    (
        "fewer_candidates",
        Path("by_candidate_size") / "fewer_candidates.json",
        Path("by_candidate_size") / "fewer_candidates.json",
    ),
    (
        "more_candidates",
        Path("by_candidate_size") / "more_candidates.json",
        Path("by_candidate_size") / "more_candidates.json",
    ),
    (
        "shorter_desc_fewer_candidates",
        Path("by_length_candidate_2x2") / "shorter_desc_fewer_candidates.json",
        Path("by_length_candidate_2x2") / "shorter_desc_fewer_candidates.json",
    ),
    (
        "shorter_desc_more_candidates",
        Path("by_length_candidate_2x2") / "shorter_desc_more_candidates.json",
        Path("by_length_candidate_2x2") / "shorter_desc_more_candidates.json",
    ),
    (
        "longer_desc_fewer_candidates",
        Path("by_length_candidate_2x2") / "longer_desc_fewer_candidates.json",
        Path("by_length_candidate_2x2") / "longer_desc_fewer_candidates.json",
    ),
    (
        "longer_desc_more_candidates",
        Path("by_length_candidate_2x2") / "longer_desc_more_candidates.json",
        Path("by_length_candidate_2x2") / "longer_desc_more_candidates.json",
    ),
]


@dataclass(frozen=True)
class SourceData:
    pattern: str
    source_path: Path
    source_stem: str
    data: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create condition-analysis subsets for every hierarchy-pattern test-data JSON."
    )
    parser.add_argument("--condition-dir", default=str(DEFAULT_CONDITION_DIR))
    parser.add_argument("--pattern-dir", default=str(DEFAULT_PATTERN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--source-glob",
        default="*/*_drop_unknown_samples.json",
        help="Glob relative to --pattern-dir. Defaults to all pattern drop-unknown test-data JSON files.",
    )
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


def write_json(path: Path, payload: Any, backups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_if_exists(path)
    if backup:
        backups.append(backup)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], backups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_if_exists(path)
    if backup:
        backups.append(backup)
    fieldnames = [
        "pattern",
        "source_file",
        "source_path",
        "subset",
        "count",
        "expected_count",
        "match",
        "output_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_split_ids(condition_dir: Path) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for subset_name, split_rel_path, _ in SUBSETS:
        split_path = condition_dir / split_rel_path
        payload = load_json(split_path)
        if not isinstance(payload, dict):
            raise ValueError(f"condition split is not a JSON object keyed by sample_id: {split_path}")
        split_ids[subset_name] = [str(sample_id) for sample_id in payload.keys()]
    return split_ids


def validate_split_ids(split_ids: dict[str, list[str]]) -> None:
    overall_ids = set(split_ids["overall"])
    for name, ids in split_ids.items():
        duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate sample_id values in condition split {name}: {duplicates[:20]}")
        outside = sorted(set(ids) - overall_ids)
        if outside:
            raise ValueError(f"condition split {name} contains IDs outside overall: {outside[:20]}")

    axis_checks = {
        "by_length": ["shorter_desc", "longer_desc"],
        "by_candidate_size": ["fewer_candidates", "more_candidates"],
        "by_length_candidate_2x2": [
            "shorter_desc_fewer_candidates",
            "shorter_desc_more_candidates",
            "longer_desc_fewer_candidates",
            "longer_desc_more_candidates",
        ],
    }
    for axis_name, names in axis_checks.items():
        combined: Counter[str] = Counter()
        for name in names:
            combined.update(split_ids[name])
        duplicates = sorted(sample_id for sample_id, count in combined.items() if count > 1)
        missing = sorted(overall_ids - set(combined))
        if duplicates or missing:
            raise ValueError(
                f"condition split axis {axis_name} is not a partition: "
                f"duplicate_count={len(duplicates)}, missing_count={len(missing)}"
            )


def load_sources(pattern_dir: Path, source_glob: str) -> list[SourceData]:
    paths = sorted(pattern_dir.glob(source_glob), key=lambda p: (p.parent.name, p.name))
    if not paths:
        raise ValueError(f"no source files matched {pattern_dir / source_glob}")

    sources: list[SourceData] = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"source JSON is not a dict keyed by sample_id: {path}")
        sources.append(
            SourceData(
                pattern=path.parent.name,
                source_path=path,
                source_stem=path.stem,
                data={str(sample_id): value for sample_id, value in payload.items()},
            )
        )
    return sources


def validate_source_ids(source: SourceData, overall_ids: set[str]) -> None:
    source_ids = set(source.data)
    missing = sorted(overall_ids - source_ids)
    extra = sorted(source_ids - overall_ids)
    if missing or extra:
        details = [
            f"sample_id mismatch for {source.source_path}",
            f"missing_count={len(missing)}",
            f"extra_count={len(extra)}",
        ]
        if missing:
            details.append(f"first missing IDs: {missing[:20]}")
        if extra:
            details.append(f"first extra IDs: {extra[:20]}")
        raise ValueError("\n".join(details))


def subset_payload(source: SourceData, ids: list[str]) -> dict[str, Any]:
    return {sample_id: source.data[sample_id] for sample_id in ids}


def output_root_for_source(output_dir: Path, source: SourceData) -> Path:
    return output_dir / source.pattern / source.source_stem


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
    condition_dir: Path,
    pattern_dir: Path,
    output_dir: Path,
    source_glob: str,
    sources: list[SourceData],
    summary_rows: list[dict[str, Any]],
    backups: list[str],
) -> str:
    source_rows = [
        {
            "pattern": source.pattern,
            "source_file": source.source_path.name,
            "count": len(source.data),
            "output_root": str(output_root_for_source(output_dir, source).relative_to(output_dir)),
        }
        for source in sources
    ]
    subset_counts: dict[str, int] = {}
    for row in summary_rows:
        subset_counts[str(row["subset"])] = int(row["expected_count"])
    subset_rows = [
        {
            "subset": subset,
            "count": subset_counts[subset],
        }
        for subset, _, _ in SUBSETS
    ]

    lines = [
        "# Hierarchy Pattern Condition-Analysis Subset Creation Report",
        "",
        f"作成日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "- 既存の `condition_analysis` 9分割を、階層構造パターン別テストデータへ同じ `sample_id` で投影した。",
        "- 新しいtrain/test splitではなく、既存テストデータ336件を条件別に再集計するための派生ファイルである。",
        "- サンプル内容は変更せず、各パターンJSON内の値をそのまま保持した。",
        "",
        "## Inputs",
        "",
        f"- condition split dir: `{condition_dir}`",
        f"- hierarchy pattern dir: `{pattern_dir}`",
        f"- source glob: `{source_glob}`",
        "",
        "## Output",
        "",
        f"- output dir: `{output_dir}`",
        f"- source files processed: {len(sources)}",
        f"- JSON files created: {len(summary_rows)}",
        "",
        "## Subset Counts",
        "",
        markdown_table(subset_rows, ["subset", "count"]),
        "",
        "## Processed Sources",
        "",
        markdown_table(source_rows, ["pattern", "source_file", "count", "output_root"]),
        "",
        "## Validation",
        "",
        "- 各source JSONの `sample_id` 集合は `condition_analysis/overall.json` と完全一致した。",
        "- 各source x subset の件数は元の9分割件数と一致した。",
        "- 各分類軸の重複・欠落は、入力分割側で検証済みのpartitionを使用した。",
        "",
        "## Notes",
        "",
        "- `shorter_desc` / `longer_desc` は、テストデータ内での相対的な説明文長を表す。",
        "- `fewer_candidates` / `more_candidates` は、テストデータ内での相対的なAPI候補数を表す。",
        "- `sum_oracle_category_candidate_size` は、正解カテゴリパスが分かっていると仮定した後に残るAPI候補数であり、カテゴリ予測自体の難しさを直接表すものではない。",
        "",
        "## Execution Command",
        "",
        "```bash",
        "python3 scripts/create_condition_analysis_hierarchy_pattern_subsets.py",
        "```",
        "",
        f"- script: `{Path(__file__).resolve()}`",
        f"- backups created: {len(backups)}",
    ]
    if backups:
        lines.extend(["", "## Backups", ""])
        lines.extend(f"- `{backup}`" for backup in backups)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    condition_dir = Path(args.condition_dir).resolve()
    pattern_dir = Path(args.pattern_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_glob = str(args.source_glob)
    report_path = output_dir / "reports" / "hierarchy_pattern_subset_creation_report.md"
    summary_path = output_dir / "reports" / "hierarchy_pattern_subset_summary.csv"

    try:
        split_ids = load_split_ids(condition_dir)
        validate_split_ids(split_ids)
        overall_ids = set(split_ids["overall"])
        sources = load_sources(pattern_dir, source_glob)

        backups: list[str] = []
        summary_rows: list[dict[str, Any]] = []
        for source in sources:
            validate_source_ids(source, overall_ids)
            source_output_root = output_root_for_source(output_dir, source)
            for subset_name, _, output_rel_path in SUBSETS:
                ids = split_ids[subset_name]
                payload = subset_payload(source, ids)
                output_path = source_output_root / output_rel_path
                write_json(output_path, payload, backups)
                summary_rows.append(
                    {
                        "pattern": source.pattern,
                        "source_file": source.source_path.name,
                        "source_path": str(source.source_path),
                        "subset": subset_name,
                        "count": len(payload),
                        "expected_count": len(ids),
                        "match": "yes" if len(payload) == len(ids) else "no",
                        "output_file": str(output_path),
                    }
                )

        write_csv(summary_path, summary_rows, backups)
        report = build_report(
            condition_dir=condition_dir,
            pattern_dir=pattern_dir,
            output_dir=output_dir,
            source_glob=source_glob,
            sources=sources,
            summary_rows=summary_rows,
            backups=backups,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        backup = backup_if_exists(report_path)
        if backup:
            backups.append(backup)
            report = build_report(
                condition_dir=condition_dir,
                pattern_dir=pattern_dir,
                output_dir=output_dir,
                source_glob=source_glob,
                sources=sources,
                summary_rows=summary_rows,
                backups=backups,
            )
        report_path.write_text(report, encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Processed source files: {len(sources)}")
    print(f"Created JSON files: {len(summary_rows)}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
