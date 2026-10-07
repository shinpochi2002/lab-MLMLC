#!/usr/bin/env python3
"""Split the current evaluation pool into fixed validation and test samples.

The same mashup keys are used in every hierarchy and masking variant. Source
files and training data are retained; existing split artifacts are immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation_split import BASELINE_SOURCE, DEFAULT_SPLIT_DIR, SOURCE_ROOT
from common.project_paths import PROJECT_ROOT, ROOT_JSON_DIR


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_dataset(path: Path) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    data = json.loads(content)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty object keyed by mashup name: {path}")
    return data, digest(content)


def sample_identity(entry: Any) -> tuple[int, frozenset[str]]:
    if isinstance(entry, dict):
        numeric_id, apis = entry["id"], entry["apis"]
    elif isinstance(entry, list) and len(entry) >= 4:
        numeric_id, apis = entry[3], entry[2]
    else:
        raise ValueError("Unsupported mashup record")
    if not isinstance(numeric_id, int) or not isinstance(apis, list):
        raise ValueError("Invalid mashup ID or API labels")
    return numeric_id, frozenset(apis)


def split_keys(data: dict[str, Any], validation_size: int, seed: int) -> dict[str, list[str]]:
    if not 0 < validation_size < len(data):
        raise ValueError("validation-size must be positive and leave at least one test sample")
    # Sorting before sampling makes membership independent of JSON insertion order.
    validation = set(random.Random(seed).sample(sorted(data), validation_size))
    return {
        "validation": sorted(validation),
        "test": sorted(set(data) - validation),
    }


def create_split(
    source_root: Path = SOURCE_ROOT,
    source_path: Path = BASELINE_SOURCE,
    train_path: Path = ROOT_JSON_DIR / "study_data_mashup_apis.json",
    output_dir: Path = DEFAULT_SPLIT_DIR,
    validation_size: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    source_root, source_path = source_root.resolve(), source_path.resolve()
    output_dir = output_dir.resolve()
    source, source_hash = load_dataset(source_path)
    train, train_hash = load_dataset(train_path)
    overlap = set(source) & set(train)
    if overlap:
        raise ValueError(f"Evaluation samples overlap training data: {sorted(overlap)[:10]}")
    identities = {key: sample_identity(entry) for key, entry in source.items()}
    if len({identity[0] for identity in identities.values()}) != len(source):
        raise ValueError("Source contains duplicate numeric mashup IDs")
    splits = split_keys(source, validation_size, seed)
    membership = {name: set(keys) for name, keys in splits.items()}
    paths = sorted(set(
        list((source_root / "layer1_direct_api").glob("*_drop_unknown_samples*.json"))
        + list((source_root / "by_pattern").glob("*/*_drop_unknown_samples*.json"))
        + [source_path]
    ))
    artifacts: dict[Path, bytes] = {}
    datasets = {}
    for path in paths:
        data, source_sha256 = load_dataset(path)
        if set(data) != set(source):
            raise ValueError(f"Source key mismatch: {path}")
        for key, entry in data.items():
            if sample_identity(entry) != identities[key]:
                raise ValueError(f"Mashup ID or API label mismatch for {key!r}: {path}")
        relative = path.relative_to(source_root)
        record: dict[str, Any] = {"source_sha256": source_sha256, "sha256": {}}
        for name, keys in membership.items():
            subset = {key: entry for key, entry in data.items() if key in keys}
            content = json_bytes(subset)
            artifacts[output_dir / name / relative] = content
            record["sha256"][name] = digest(content)
        datasets[relative.as_posix()] = record
    manifest = {
        "schema_version": 1,
        "purpose": "Select the hierarchy using validation only; evaluate the fixed choice on test.",
        "sampling": "random.Random(seed).sample(sorted(mashup_keys), validation_size)",
        "seed": seed,
        "source": source_path.relative_to(source_root).as_posix(),
        "source_sha256": source_hash,
        "source_count": len(source),
        "training_data": {
            "path": str(train_path.resolve().relative_to(PROJECT_ROOT))
            if train_path.resolve().is_relative_to(PROJECT_ROOT) else str(train_path.resolve()),
            "sha256": train_hash,
            "count": len(train),
            "overlap_count": 0,
        },
        "splits": {
            name: {"count": len(keys), "keys": keys, "ids": [identities[key][0] for key in keys]}
            for name, keys in splits.items()
        },
        "datasets": datasets,
    }
    artifacts[output_dir / "manifest.json"] = json_bytes(manifest)
    # Preflight every destination before writing anything. A rerun with identical
    # inputs is safe; a different seed/source must use a different output directory.
    for path, content in artifacts.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to change a fixed split: {path}; use a new --output-dir")
        if path.resolve() in paths or path.resolve() == train_path.resolve():
            raise ValueError(f"Output would overwrite input data: {path}")
    for path, content in artifacts.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASELINE_SOURCE)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--train-path", type=Path, default=ROOT_JSON_DIR / "study_data_mashup_apis.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = create_split(args.source_root, args.source, args.train_path, args.output_dir, args.validation_size, args.seed)
    print(f"Source: {manifest['source_count']} samples; seed={args.seed}")
    for name, split in manifest["splits"].items():
        print(f"{name}: {split['count']} samples")
    print(f"Datasets: {len(manifest['datasets'])}; train overlap: 0")
    print(f"Manifest: {args.output_dir.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
