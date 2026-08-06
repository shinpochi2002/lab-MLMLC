#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
CODEX_ROOT = Path(__file__).resolve().parent
BY_PATTERN_DIR = CODEX_ROOT / "output" / "by_pattern"
DEFAULT_RESULTS_DIR = CODEX_ROOT / "results" / "hierarchy_existing_models"
DEFAULT_MASKED_RESULTS_DIR = CODEX_ROOT / "results" / "hierarchy_existing_models_api_name_masked"
DEFAULT_MODEL_PYTHON = Path("/home/nakagawa/venvs/swallow-py312/bin/python")
DEFAULT_CONDITION_2X2_RESULTS_DIR = CODEX_ROOT / "results" / "hierarchy_existing_models_condition_2x2"
MASKED_DATASET = "masked"
UNMASKED_DATASET = "unmasked"
MASKED_SUFFIX = "_drop_unknown_samples_categories_from_collaboration_api_name_masked.json"
UNMASKED_SUFFIX = "_drop_unknown_samples.json"
MASK_PLACEHOLDERS = ("<API_NAME>", "[API_NAME]")
MASKED_LAYER1_FILE = (
    "test_data_mashup_apis_layer1_drop_unknown_samples_"
    "categories_from_collaboration_api_name_masked.json"
)
UNMASKED_LAYER1_FILE = "test_data_mashup_apis_layer1_drop_unknown_samples.json"
CONDITION_SUBSETS = (
    ("by_length", "shorter_desc"),
    ("by_length", "longer_desc"),
    ("by_candidate_size", "fewer_candidates"),
    ("by_candidate_size", "more_candidates"),
    ("by_length_candidate_2x2", "shorter_desc_fewer_candidates"),
    ("by_length_candidate_2x2", "shorter_desc_more_candidates"),
    ("by_length_candidate_2x2", "longer_desc_fewer_candidates"),
    ("by_length_candidate_2x2", "longer_desc_more_candidates"),
)


@dataclass(frozen=True)
class EvalTask:
    name: str
    layer: int
    target: str
    subset_name: str | None
    layer1_category_num: str | None
    script_path: str
    evaluation_data_path: str
    out_csv_path: str
    log_path: str


def natural_key(text: str) -> list[int | str]:
    parts: list[int | str] = []
    buf = ""
    is_digit = None
    for ch in text:
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            buf += ch
            is_digit = ch_is_digit
        else:
            parts.append(int(buf) if is_digit else buf)
            buf = ch
            is_digit = ch_is_digit
    if buf:
        parts.append(int(buf) if is_digit else buf)
    return parts


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def make_task(
    *,
    layer: int,
    target: str,
    layer1_category_num: str | None,
    script_name: str,
    eval_path: Path,
    results_dir: Path,
    subset_name: str | None = None,
) -> EvalTask:
    base_task_name = "layer1_direct_api" if layer == 1 else f"layer{layer}_{safe_name(target)}"
    if subset_name:
        subset_dir = safe_name(subset_name)
        task_name = f"{base_task_name}__{subset_dir}"
        out_csv_path = results_dir / subset_dir / f"{base_task_name}_results.csv"
        log_path = results_dir / "logs" / subset_dir / f"{base_task_name}.log"
    else:
        task_name = base_task_name
        out_csv_path = results_dir / f"{base_task_name}_results.csv"
        log_path = results_dir / "logs" / f"{base_task_name}.log"
    return EvalTask(
        name=task_name,
        layer=layer,
        target=target,
        subset_name=subset_name,
        layer1_category_num=layer1_category_num,
        script_path=str(LAB_ROOT / script_name),
        evaluation_data_path=str(eval_path),
        out_csv_path=str(out_csv_path),
        log_path=str(log_path),
    )


def first_match(pattern: str) -> Path | None:
    matches = sorted(BY_PATTERN_DIR.glob(pattern), key=lambda p: natural_key(str(p)))
    return matches[0] if matches else None


def condition_subset_root_for_task(task: EvalTask) -> Path:
    if task.layer == 1:
        return CODEX_ROOT / "test_subsets" / "condition_analysis"

    source_path = Path(task.evaluation_data_path)
    pattern_name = source_path.parent.name
    source_stem = source_path.stem
    return (
        CODEX_ROOT
        / "test_subsets"
        / "condition_analysis_by_pattern"
        / pattern_name
        / source_stem
    )


def expand_condition_2x2_tasks(tasks: list[EvalTask], results_dir: Path) -> list[EvalTask]:
    expanded: list[EvalTask] = []
    for task in tasks:
        subset_root = condition_subset_root_for_task(task)
        missing = []
        for subset_group, subset_name in CONDITION_SUBSETS:
            subset_path = subset_root / subset_group / f"{subset_name}.json"
            if not subset_path.exists():
                missing.append(str(subset_path))
                continue
            expanded.append(
                make_task(
                    layer=task.layer,
                    target=task.target,
                    layer1_category_num=task.layer1_category_num,
                    script_name=Path(task.script_path).name,
                    eval_path=subset_path,
                    results_dir=results_dir,
                    subset_name=subset_name,
                )
            )
        if missing:
            print(f"[WARN] missing condition subset(s) for {task.name}:")
            for path in missing:
                print(f"  - {path}")
    return expanded


def discover_tasks(results_dir: Path, dataset: str = MASKED_DATASET) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    eval_suffix = MASKED_SUFFIX if dataset == MASKED_DATASET else UNMASKED_SUFFIX
    layer1_file = MASKED_LAYER1_FILE if dataset == MASKED_DATASET else UNMASKED_LAYER1_FILE

    layer1_eval = CODEX_ROOT / "output" / "layer1_direct_api" / layer1_file
    if (LAB_ROOT / "layer1").is_dir() and layer1_eval.exists():
        tasks.append(
            make_task(
                layer=1,
                target="direct_api",
                layer1_category_num=None,
                script_name="mashup_dis_nosp_1layer_api.py",
                eval_path=layer1_eval,
                results_dir=results_dir,
            )
        )

    for model_dir in sorted(LAB_ROOT.glob("layer2_*"), key=lambda p: natural_key(p.name)):
        layer1_num = model_dir.name.removeprefix("layer2_")
        eval_path = first_match(
            f"*/test_data_mashup_apis_with_hierarchy_layer2_{layer1_num}{eval_suffix}"
        )
        if not eval_path:
            continue
        tasks.append(
            make_task(
                layer=2,
                target=eval_path.parent.name,
                layer1_category_num=layer1_num,
                script_name="mashup_dis_nosp_2layer_api.py",
                eval_path=eval_path,
                results_dir=results_dir,
            )
        )

    for model_dir in sorted(LAB_ROOT.glob("layer3_*"), key=lambda p: natural_key(p.name)):
        target = model_dir.name.removeprefix("layer3_")
        eval_path = first_match(
            f"*/test_data_mashup_apis_with_hierarchy_layer3_{target}{eval_suffix}"
        )
        if not eval_path:
            continue
        layer1_num = target.replace("-", "_").split("_")[0]
        tasks.append(
            make_task(
                layer=3,
                target=target,
                layer1_category_num=layer1_num,
                script_name="mashup_dis_nosp_3layer_api.py",
                eval_path=eval_path,
                results_dir=results_dir,
            )
        )

    for model_dir in sorted(LAB_ROOT.glob("layer4_*"), key=lambda p: natural_key(p.name)):
        target = model_dir.name.removeprefix("layer4_")
        eval_path = first_match(
            f"*/test_data_mashup_apis_with_hierarchy_layer4_{target}{eval_suffix}"
        )
        if not eval_path:
            continue
        layer1_num = target.replace("-", "_").split("_")[0]
        tasks.append(
            make_task(
                layer=4,
                target=target,
                layer1_category_num=layer1_num,
                script_name="mashup_dis_nosp_4layer_api.py",
                eval_path=eval_path,
                results_dir=results_dir,
            )
        )

    return tasks


def description_from_entry(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("description", "") or "")
    if isinstance(entry, list) and len(entry) > 1:
        return str(entry[1] or "")
    return ""


def validate_evaluation_data(
    tasks: list[EvalTask],
    dataset: str,
) -> dict[str, tuple[int, int]]:
    """Validate every resolved JSON before starting expensive model runs."""
    stats: dict[str, tuple[int, int]] = {}
    errors: list[str] = []

    for task in tasks:
        path = Path(task.evaluation_data_path)
        if not path.is_file():
            errors.append(f"{task.name}: missing evaluation data: {path}")
            continue
        try:
            with path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{task.name}: cannot load {path}: {exc}")
            continue
        if not isinstance(data, dict) or not data:
            errors.append(f"{task.name}: evaluation data must be a non-empty JSON object: {path}")
            continue

        masked_count = sum(
            any(placeholder in description_from_entry(entry) for placeholder in MASK_PLACEHOLDERS)
            for entry in data.values()
        )
        if dataset == MASKED_DATASET:
            if "api_name_masked" not in path.name:
                errors.append(f"{task.name}: resolved file is not marked as masked: {path}")
            if masked_count == 0:
                errors.append(
                    f"{task.name}: no API-name mask placeholders "
                    f"{MASK_PLACEHOLDERS} found: {path}"
                )
        stats[str(path)] = (len(data), masked_count)

    if errors:
        raise ValueError("Evaluation-data validation failed:\n- " + "\n- ".join(errors))
    return stats


def parse_layers(value: str) -> set[int]:
    if value == "all":
        return {1, 2, 3, 4}
    layers = {int(part.strip()) for part in value.split(",") if part.strip()}
    unknown = layers - {1, 2, 3, 4}
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown layer(s): {sorted(unknown)}")
    return layers


def write_manifest(tasks: list[EvalTask], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "test_data_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(task) for task in tasks], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_task(task: EvalTask, python_cmd: str, skip_existing: bool) -> dict[str, object]:
    out_csv = Path(task.out_csv_path)
    log_path = Path(task.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_csv.exists():
        return {**asdict(task), "status": "skipped", "returncode": 0, "seconds": 0.0}

    env = os.environ.copy()
    env["EVALUATION_DATA_PATH"] = task.evaluation_data_path
    env["OUT_CSV_PATH"] = task.out_csv_path
    env["PYTHONUNBUFFERED"] = "1"
    if task.layer > 1:
        env["TARGET"] = task.target
    if task.layer1_category_num:
        env["LAYER1_CATEGORY_NUM"] = task.layer1_category_num

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"task={task.name}\n")
        log_file.write(f"script={task.script_path}\n")
        log_file.write(f"evaluation_data={task.evaluation_data_path}\n")
        log_file.write(f"out_csv={task.out_csv_path}\n\n")
        log_file.flush()
        proc = subprocess.run(
            [python_cmd, task.script_path],
            cwd=str(LAB_ROOT.parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - started
    status = "ok" if proc.returncode == 0 else "failed"
    return {**asdict(task), "status": status, "returncode": proc.returncode, "seconds": round(elapsed, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run existing hierarchy evaluation scripts for all patterns with trained models.")
    parser.add_argument("--layers", default="all", type=parse_layers, help="Layer filter: all, 1, 2, 3, 4, or comma-separated values like 2,3.")
    parser.add_argument(
        "--dataset",
        choices=[MASKED_DATASET, UNMASKED_DATASET],
        default=MASKED_DATASET,
        help="Evaluation dataset variant. Default: masked collaboration-category data.",
    )
    parser.add_argument("--results-dir", default=None, help="Directory for CSVs, logs, and manifests.")
    parser.add_argument("--condition-2x2", action="store_true", help="Run codex/test_subsets condition_analysis subsets, excluding overall/all.")
    parser.add_argument(
        "--python",
        default=os.getenv(
            "MODEL_PYTHON",
            str(DEFAULT_MODEL_PYTHON if DEFAULT_MODEL_PYTHON.exists() else sys.executable),
        ),
        help="Python executable used to run the existing scripts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List tasks and test data without running models.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a task when its output CSV already exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running remaining tasks after a failure.")
    args = parser.parse_args()

    if args.condition_2x2 and args.dataset == MASKED_DATASET:
        parser.error(
            "--condition-2x2 currently has only unmasked subset files; "
            "use --dataset unmasked or generate masked subsets first"
        )

    if args.condition_2x2:
        default_results_dir = DEFAULT_CONDITION_2X2_RESULTS_DIR
    elif args.dataset == MASKED_DATASET:
        default_results_dir = DEFAULT_MASKED_RESULTS_DIR
    else:
        default_results_dir = DEFAULT_RESULTS_DIR
    results_dir = Path(args.results_dir or default_results_dir).resolve()
    tasks = [
        task
        for task in discover_tasks(results_dir, dataset=args.dataset)
        if task.layer in args.layers
    ]
    if args.condition_2x2:
        tasks = expand_condition_2x2_tasks(tasks, results_dir)

    if not tasks:
        print("No runnable tasks found.")
        return 1

    try:
        data_stats = validate_evaluation_data(tasks, args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Found {len(tasks)} runnable task(s).")
    print(f"Dataset variant: {args.dataset}")
    print(f"Results directory: {results_dir}")
    print(f"Model Python: {args.python}")
    if args.condition_2x2:
        print("Dataset scope: condition_analysis subsets, excluding overall/all")
    for task in tasks:
        subset = f" [{task.subset_name}]" if task.subset_name else ""
        sample_count, masked_count = data_stats[task.evaluation_data_path]
        print(
            f"- {task.name}{subset}: {task.evaluation_data_path} "
            f"(samples={sample_count}, masked_descriptions={masked_count})"
        )

    if args.dry_run:
        return 0

    write_manifest(tasks, results_dir)
    summary: list[dict[str, object]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] running {task.name}")
        record = run_task(task, args.python, args.skip_existing)
        summary.append(record)
        print(f"[{index}/{len(tasks)}] {record['status']} rc={record['returncode']} log={task.log_path}")
        if record["returncode"] != 0 and not args.continue_on_error:
            break

    summary_path = results_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [record for record in summary if record["returncode"] != 0]
    if failed:
        print(f"Failed task(s): {', '.join(str(record['name']) for record in failed)}")
        print(f"Summary: {summary_path}")
        return 1

    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
