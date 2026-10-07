"""Shared paths and integrity checks for the fixed model-selection split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common.project_paths import HIERARCHY_RESULTS_DIR, SPLITS_DIR


SOURCE_ROOT = HIERARCHY_RESULTS_DIR / "output"
DEFAULT_SPLIT_DIR = SPLITS_DIR / "model_selection"
BASELINE_SOURCE = SOURCE_ROOT / "layer1_direct_api" / (
    "test_data_mashup_apis_layer1_drop_unknown_samples_"
    "categories_from_collaboration.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(split_dir: Path = DEFAULT_SPLIT_DIR) -> dict[str, Any]:
    path = split_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model-selection split: {path}. Run "
            "python scripts/data_prep/create_validation_test_split.py first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported split manifest: {path}")
    validation = manifest["splits"]["validation"]["keys"]
    test = manifest["splits"]["test"]["keys"]
    for name, keys in (("validation", validation), ("test", test)):
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"Empty or duplicate keys in {name}: {path}")
        if len(keys) != manifest["splits"][name]["count"]:
            raise ValueError(f"Incorrect {name} count: {path}")
    if set(validation) & set(test):
        raise ValueError(f"Validation and test overlap: {path}")
    if len(validation) + len(test) != manifest["source_count"]:
        raise ValueError(f"Split counts do not cover the source: {path}")
    return manifest


def resolve_split_path(
    source_path: Path,
    split: str,
    split_dir: Path = DEFAULT_SPLIT_DIR,
) -> Path:
    """Resolve a generated subset, rejecting missing or modified artifacts."""
    source_path = Path(source_path).resolve()
    if split == "full":
        return source_path
    if split not in {"validation", "test"}:
        raise ValueError(f"Unknown evaluation split: {split}")
    split_dir = Path(split_dir)
    manifest = load_manifest(split_dir)
    relative = source_path.relative_to(SOURCE_ROOT.resolve()).as_posix()
    if relative not in manifest["datasets"]:
        raise ValueError(f"Dataset is not registered in the split manifest: {source_path}")
    record = manifest["datasets"][relative]
    if sha256_file(source_path) != record["source_sha256"]:
        raise ValueError(f"Source has changed since the split was created: {source_path}")
    subset_path = split_dir / split / relative
    if sha256_file(subset_path) != record["sha256"][split]:
        raise ValueError(f"Split data has changed: {subset_path}")
    data = json.loads(subset_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != set(manifest["splits"][split]["keys"]):
        raise ValueError(f"Split membership mismatch: {subset_path}")
    return subset_path.resolve()
