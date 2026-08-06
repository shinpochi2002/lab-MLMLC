import argparse
import ast
import csv
import json
import re
from pathlib import Path


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

DEFAULT_RESULTS_DIR = Path(
    legacy_path("/home/nakagawa/lab-MLMLC/codex/results/hierarchy_existing_models_condition_2x2")
)
DEFAULT_OUTPUT_DIR_NAME = "metric_summaries_by_test_data"
METRIC_COLUMNS = ("precision", "recall", "f1")
LIST_COLUMNS = {
    "pred_apis": "avg_pred_apis_count",
    "true_apis": "avg_true_apis_count",
}
FIELDNAMES = [
    "test_data",
    "configuration",
    "layer",
    "target",
    "layer1_category_num",
    "sample_count",
    "source_csv",
    "evaluation_data_path",
    "status",
    "seconds",
    "avg_precision",
    "avg_recall",
    "avg_f1",
    "avg_pred_apis_count",
    "avg_true_apis_count",
]


def list_length(cell):
    """
    セルの中身が list 表記文字列なら、その長さを返す。
    それ以外（空欄・不正形式）は 0。
    """
    if isinstance(cell, list):
        return len(cell)
    if cell in (None, ""):
        return 0
    try:
        value = ast.literal_eval(cell)
        return len(value) if isinstance(value, list) else 0
    except (ValueError, SyntaxError):
        return 0


def add_list_length_column(rows, source_col, new_col):
    """
    指定した列(source_col)からリスト長を計算し、
    新しい列(new_col)として各行に追加する。
    """
    for row in rows:
        row[new_col] = list_length(row.get(source_col))
    return rows


def average_list_length(rows, source_col):
    """
    list 表記文字列が入った列の平均長を返す。
    """
    if not rows:
        return None
    return sum(list_length(row.get(source_col)) for row in rows) / len(rows)


def numeric_mean(rows, column):
    values = []
    for row in rows:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    if not values:
        return None
    return sum(values) / len(values)


def read_csv_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def load_run_summary(results_dir):
    summary_path = results_dir / "run_summary.json"
    if not summary_path.exists():
        return []
    with summary_path.open(encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"run_summary.json must contain a list: {summary_path}")
    return data


def metadata_by_csv_path(run_summary):
    metadata = {}
    for item in run_summary:
        out_csv_path = item.get("out_csv_path")
        if not out_csv_path:
            continue
        metadata[str(Path(out_csv_path))] = item
    return metadata


def parse_csv_name(csv_path):
    stem = csv_path.stem
    config = stem.removesuffix("_results")
    layer = None
    target = None

    if config == "layer1_direct_api":
        layer = 1
        target = "direct_api"
    else:
        match = re.match(r"layer(?P<layer>\d+)_(?P<target>.+)", config)
        if match:
            layer = int(match.group("layer"))
            target = match.group("target").replace("_", "-")

    return {
        "configuration": config,
        "layer": layer,
        "target": target,
    }


def discover_csvs(results_dir, run_summary):
    seen = set()
    csv_paths = []

    for item in run_summary:
        out_csv_path = item.get("out_csv_path")
        if not out_csv_path:
            continue
        csv_path = Path(out_csv_path)
        if csv_path.exists() and csv_path.suffix == ".csv":
            key = str(csv_path)
            if key not in seen:
                seen.add(key)
                csv_paths.append(csv_path)

    for csv_path in sorted(results_dir.glob("**/*_results.csv")):
        if DEFAULT_OUTPUT_DIR_NAME in csv_path.parts:
            continue
        key = str(csv_path)
        if key not in seen:
            seen.add(key)
            csv_paths.append(csv_path)

    return csv_paths


def summarize_result_csv(csv_path, metadata):
    rows = read_csv_rows(csv_path)
    parsed = parse_csv_name(csv_path)
    subset_name = metadata.get("subset_name") or csv_path.parent.name

    row = {
        "test_data": subset_name,
        "configuration": parsed["configuration"],
        "layer": metadata.get("layer", parsed["layer"]),
        "target": metadata.get("target", parsed["target"]),
        "layer1_category_num": metadata.get("layer1_category_num"),
        "sample_count": len(rows),
        "source_csv": str(csv_path),
        "evaluation_data_path": metadata.get("evaluation_data_path"),
        "status": metadata.get("status"),
        "seconds": metadata.get("seconds"),
    }

    for metric in METRIC_COLUMNS:
        row[f"avg_{metric}"] = numeric_mean(rows, metric)

    for source_col, output_col in LIST_COLUMNS.items():
        row[output_col] = (
            average_list_length(rows, source_col)
            if rows and source_col in rows[0]
            else None
        )

    return row


def build_summary(results_dir):
    run_summary = load_run_summary(results_dir)
    metadata_lookup = metadata_by_csv_path(run_summary)
    summary_rows = []

    for csv_path in discover_csvs(results_dir, run_summary):
        metadata = metadata_lookup.get(str(csv_path), {})
        summary_rows.append(summarize_result_csv(csv_path, metadata))

    if not summary_rows:
        raise FileNotFoundError(f"No *_results.csv files found under {results_dir}")

    return sort_summary(summary_rows)


def sort_summary(summary_rows):
    return sorted(
        summary_rows,
        key=lambda row: (
            str(row.get("test_data") or ""),
            int(row.get("layer") or 0),
            str(row.get("target") or ""),
            str(row.get("configuration") or ""),
        ),
    )


def write_csv(rows, out_path):
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csvs(summary_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "all_test_data_summary.csv"
    write_csv(summary_rows, all_path)

    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row.get("test_data"), []).append(row)

    per_test_data_paths = []
    for test_data in sorted(grouped, key=lambda value: str(value or "")):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(test_data)).strip("_")
        if not safe_name:
            safe_name = "unknown_test_data"
        out_path = output_dir / f"{safe_name}.csv"
        write_csv(sort_summary(grouped[test_data]), out_path)
        per_test_data_paths.append(out_path)

    return all_path, per_test_data_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate precision/recall/f1 means from result CSV files by test data."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory that contains run_summary.json and result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write summary CSV files. Defaults to results-dir/metric_summaries_by_test_data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / DEFAULT_OUTPUT_DIR_NAME

    summary_rows = build_summary(results_dir)
    all_path, per_test_data_paths = write_summary_csvs(summary_rows, output_dir)

    print(f"Wrote all summary: {all_path}")
    print(f"Wrote {len(per_test_data_paths)} per-test-data summaries to: {output_dir}")


if __name__ == "__main__":
    main()
