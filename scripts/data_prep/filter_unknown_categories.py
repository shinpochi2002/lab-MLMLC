import argparse
import json
from typing import Any, Dict, List, Tuple


def filter_legacy_entry(entry: List[Any], drop_samples_with_unknown: bool) -> Tuple[bool, List[Any]]:
    if len(entry) < 5 or not isinstance(entry[0], list) or not isinstance(entry[4], dict):
        return True, entry

    categories = list(entry[0])
    meta = dict(entry[4])
    unknown = set(meta.get("unknown_categories", []) or [])
    category_hierarchy_paths = meta.get("category_hierarchy_paths", {}) or {}
    resolved_category_map = meta.get("resolved_category_map", {}) or {}

    kept_categories = [c for c in categories if c not in unknown]
    removed = [c for c in categories if c in unknown]

    if drop_samples_with_unknown and removed:
        return False, entry

    # Keep only known categories and rebuild metadata.
    meta["unknown_categories"] = []
    meta["category_hierarchy_paths"] = {
        c: path for c, path in category_hierarchy_paths.items() if c in kept_categories
    }
    meta["resolved_category_map"] = {
        c: rc for c, rc in resolved_category_map.items() if c in kept_categories
    }

    level_labels: Dict[str, set] = {}
    for path in meta["category_hierarchy_paths"].values():
        if not isinstance(path, list):
            continue
        for i, label in enumerate(path, start=1):
            level = f"level{i}"
            level_labels.setdefault(level, set()).add(label)
    meta["answer_labels_by_level"] = {
        level: sorted(labels)
        for level, labels in sorted(level_labels.items(), key=lambda x: int(x[0].replace("level", "")))
    }

    out = list(entry)
    out[0] = kept_categories
    out[4] = meta
    return True, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create test data with unknown categories removed.")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--drop-samples-with-unknown",
        action="store_true",
        help="Drop entire samples if they originally contained unknown categories.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input must be dict keyed by mashup name.")

    out: Dict[str, Any] = {}
    total = len(data)
    kept = 0
    dropped = 0
    changed = 0

    for name, entry in data.items():
        if isinstance(entry, list):
            keep, new_entry = filter_legacy_entry(entry, args.drop_samples_with_unknown)
            if not keep:
                dropped += 1
                continue
            if new_entry != entry:
                changed += 1
            out[name] = new_entry
            kept += 1
        else:
            out[name] = entry
            kept += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("input:", args.input)
    print("output:", args.output)
    print("total:", total)
    print("kept:", kept)
    print("dropped:", dropped)
    print("changed:", changed)


if __name__ == "__main__":
    main()
