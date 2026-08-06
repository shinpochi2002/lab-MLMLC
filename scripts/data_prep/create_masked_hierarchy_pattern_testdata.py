#!/usr/bin/env python3
"""Create API-name-masked test data for every hierarchy pattern.

The source is the collaboration-category layer-1 test set.  Existing full-depth
``drop_unknown_samples`` files are used only as schema/order templates; all
sample data and hierarchical ground-truth labels are rebuilt from the source
and each pattern's ``category_hierarchy_*.json`` definition.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CODEX_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    CODEX_ROOT
    / "output/layer1_direct_api/"
    "test_data_mashup_apis_layer1_drop_unknown_samples_"
    "categories_from_collaboration_api_name_masked.json"
)
DEFAULT_PATTERN_DIR = CODEX_ROOT / "output/by_pattern"
OUTPUT_SUFFIX = "_categories_from_collaboration_api_name_masked.json"
GT_LAYER_RE = re.compile(r"^gt_layer(\d+)$")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def choose_hierarchy(pattern_dir: Path) -> Path:
    exact = pattern_dir / f"category_hierarchy_{pattern_dir.name}.json"
    if exact.is_file():
        return exact

    candidates = sorted(pattern_dir.glob("category_hierarchy_*.json"))
    if not candidates:
        raise ValueError(f"No category hierarchy found in {pattern_dir}")
    if len(candidates) == 1:
        return candidates[0]

    definitions = [load_json(path) for path in candidates]
    signatures = {
        json.dumps(
            {
                "max_depth": item.get("max_depth"),
                "category_paths": item.get("category_paths"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in definitions
    }
    if len(signatures) != 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Ambiguous hierarchy definitions in {pattern_dir}: {names}")
    return candidates[0]


def max_gt_layer(data: dict[str, Any]) -> int:
    if not data:
        return 0
    first = next(iter(data.values()))
    if not isinstance(first, dict):
        return 0
    layers = [
        int(match.group(1))
        for key in first
        if (match := GT_LAYER_RE.match(key)) is not None
    ]
    return max(layers, default=0)


def choose_template(pattern_dir: Path, expected_layers: int) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(pattern_dir.glob("*_drop_unknown_samples.json")):
        if path.name.endswith(OUTPUT_SUFFIX):
            continue
        data = load_json(path)
        if isinstance(data, dict) and max_gt_layer(data) == expected_layers:
            candidates.append((path, data))

    if len(candidates) != 1:
        names = ", ".join(path.name for path, _ in candidates) or "none"
        raise ValueError(
            f"Expected one full-depth template with {expected_layers} gt layers "
            f"in {pattern_dir}, found: {names}"
        )
    return candidates[0]


def parse_source_entry(name: str, entry: Any) -> tuple[list[str], str, list[str], int]:
    if not isinstance(entry, list) or len(entry) < 4:
        raise ValueError(f"Invalid source entry for {name!r}")
    categories, description, apis, numeric_id = entry[:4]
    if not isinstance(categories, list) or not all(isinstance(x, str) for x in categories):
        raise ValueError(f"Invalid categories for {name!r}")
    if not isinstance(description, str):
        raise ValueError(f"Invalid description for {name!r}")
    if not isinstance(apis, list) or not all(isinstance(x, str) for x in apis):
        raise ValueError(f"Invalid APIs for {name!r}")
    if not isinstance(numeric_id, int):
        raise ValueError(f"Invalid id for {name!r}")
    return categories, description, apis, numeric_id


def build_labels(
    categories: list[str],
    category_paths: dict[str, list[int]],
    max_depth: int,
) -> dict[str, list[str]]:
    unknown = sorted(set(categories) - set(category_paths))
    if unknown:
        raise ValueError(f"Categories missing from hierarchy: {unknown}")

    paths = [tuple(category_paths[category]) for category in categories]
    incomplete = sorted(
        category
        for category in categories
        if len(category_paths[category]) != max_depth
    )
    if incomplete:
        raise ValueError(f"Categories with incomplete hierarchy paths: {incomplete}")

    labels: dict[str, list[str]] = {}
    for level in range(1, max_depth + 1):
        prefixes = sorted({path[:level] for path in paths})
        labels[f"gt_layer{level}"] = [
            "-".join(str(label) for label in prefix) for prefix in prefixes
        ]
    labels[f"gt_layer{max_depth + 1}"] = sorted(set(categories))
    return labels


def build_pattern_data(
    source: dict[str, Any],
    template: dict[str, Any],
    category_paths: dict[str, list[int]],
    max_depth: int,
) -> dict[str, Any]:
    if set(source) != set(template):
        missing = sorted(set(source) - set(template))
        extra = sorted(set(template) - set(source))
        raise ValueError(f"Source/template key mismatch; missing={missing}, extra={extra}")

    output: dict[str, Any] = {}
    for name, source_entry in source.items():
        categories, description, apis, numeric_id = parse_source_entry(name, source_entry)
        labels = build_labels(categories, category_paths, max_depth)
        template_entry = template[name]
        if not isinstance(template_entry, dict):
            raise ValueError(f"Invalid template entry for {name!r}")

        values = {
            "categories": categories,
            "description": description,
            "apis": apis,
            "id": numeric_id,
            **labels,
        }
        expected_keys = set(values)
        if set(template_entry) != expected_keys:
            raise ValueError(
                f"Unexpected template fields for {name!r}: "
                f"expected={sorted(expected_keys)}, actual={sorted(template_entry)}"
            )

        # Preserve each existing pattern's field order for minimal downstream churn.
        output[name] = {key: values[key] for key in template_entry}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create masked collaboration-category test data for all hierarchy patterns."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-generated outputs.",
    )
    args = parser.parse_args()

    source = load_json(args.source)
    if not isinstance(source, dict):
        raise ValueError("Source JSON must be an object keyed by mashup name")

    pattern_dirs = sorted(path for path in args.pattern_dir.iterdir() if path.is_dir())
    if not pattern_dirs:
        raise ValueError(f"No pattern directories found in {args.pattern_dir}")

    generated: list[Path] = []
    for pattern_dir in pattern_dirs:
        hierarchy_path = choose_hierarchy(pattern_dir)
        hierarchy = load_json(hierarchy_path)
        max_depth = hierarchy.get("max_depth")
        category_paths = hierarchy.get("category_paths")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError(f"Invalid max_depth in {hierarchy_path}")
        if not isinstance(category_paths, dict):
            raise ValueError(f"Invalid category_paths in {hierarchy_path}")

        template_path, template = choose_template(pattern_dir, max_depth + 1)
        output_path = template_path.with_name(template_path.stem + OUTPUT_SUFFIX)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")

        output = build_pattern_data(source, template, category_paths, max_depth)
        write_json(output_path, output)
        generated.append(output_path)
        print(f"{pattern_dir.name}: {output_path.name} ({len(output)} samples)")

    print(f"Generated {len(generated)} hierarchy-pattern datasets.")


if __name__ == "__main__":
    main()
