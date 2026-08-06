import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import HIERARCHY_RESULTS_DIR, legacy_path

def extract_token_tuple(path: str) -> Optional[Tuple[str, ...]]:
    name = os.path.basename(path)
    m = re.search(r"_results_(.+?)(?:_layer|ver\.json$)", name)
    if not m:
        return None
    token_str = m.group(1)
    if not token_str:
        return None
    return tuple(token_str.split("_"))


def load_cluster_maps(cluster_dir: str) -> Dict[Tuple[str, ...], Dict[str, List[str]]]:
    token_to_clusters: Dict[Tuple[str, ...], Dict[str, List[str]]] = {}
    for path in sorted(glob.glob(os.path.join(cluster_dir, "*.json"))):
        token_tuple = extract_token_tuple(path)
        if token_tuple is None:
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        clusters = data.get("clusters")
        if not isinstance(clusters, dict):
            continue

        normalized: Dict[str, List[str]] = {}
        for cid, cats in clusters.items():
            if isinstance(cats, list):
                normalized[str(cid)] = [str(c) for c in cats]

        token_to_clusters[token_tuple] = normalized

    return token_to_clusters


def load_single_cluster_file(path: str) -> Optional[Tuple[Tuple[str, ...], Dict[str, List[str]]]]:
    token_tuple = extract_token_tuple(path)
    if token_tuple is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data.get("clusters")
    if not isinstance(clusters, dict):
        return None

    normalized: Dict[str, List[str]] = {}
    for cid, cats in clusters.items():
        if isinstance(cats, list):
            normalized[str(cid)] = [str(c) for c in cats]

    return token_tuple, normalized


def find_category_cluster_id(clusters: Dict[str, List[str]], category: str) -> Optional[str]:
    for cluster_id, categories in clusters.items():
        if category in categories:
            return cluster_id
    return None


def build_category_paths(token_to_clusters: Dict[Tuple[str, ...], Dict[str, List[str]]]) -> Tuple[Tuple[str, ...], Dict[str, List[int]]]:
    if not token_to_clusters:
        raise ValueError("No valid cluster JSON files were found.")

    root_candidates = [t for t in token_to_clusters.keys() if len(t) == 1]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one root file (len==1), found: {root_candidates}")
    root = root_candidates[0]

    root_clusters = token_to_clusters[root]
    all_categories: List[str] = []
    for cats in root_clusters.values():
        all_categories.extend(cats)
    all_categories = sorted(set(all_categories))

    category_to_path: Dict[str, List[int]] = {}

    for category in all_categories:
        current = root
        path_labels: List[int] = []

        while current in token_to_clusters:
            clusters = token_to_clusters[current]
            cid = find_category_cluster_id(clusters, category)
            if cid is None:
                break

            path_labels.append(int(cid))
            next_tuple = current + (cid,)
            if next_tuple not in token_to_clusters:
                break
            current = next_tuple

        category_to_path[category] = path_labels

    return root, category_to_path


def build_tree_from_paths(category_to_path: Dict[str, List[int]]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for category, path in category_to_path.items():
        node = tree
        for label in path:
            key = str(label)
            if key not in node:
                node[key] = {}
            node = node[key]
        node.setdefault("_categories", []).append(category)
    return tree


def build_layer_connections(category_to_path: Dict[str, List[int]]) -> Dict[str, Dict[str, List[int]]]:
    l1_to_l2: Dict[int, set] = defaultdict(set)
    l2_to_l3: Dict[str, set] = defaultdict(set)

    for path in category_to_path.values():
        if len(path) >= 2:
            l1_to_l2[path[0]].add(path[1])
        if len(path) >= 3:
            l2_to_l3[f"{path[0]}-{path[1]}"].add(path[2])

    return {
        "layer1_to_layer2": {str(k): sorted(v) for k, v in sorted(l1_to_l2.items())},
        "layer2_to_layer3": {k: sorted(v) for k, v in sorted(l2_to_l3.items())},
    }


def build_category_routes(category_to_path: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    for category, path in sorted(category_to_path.items()):
        route: Dict[str, Any] = {"category": category}
        if len(path) > 0:
            route["layer1"] = path[0]
        if len(path) > 1:
            route["layer2"] = path[1]
        if len(path) > 2:
            route["layer3"] = path[2]
        routes.append(route)
    return routes


def build_simple_hierarchy(category_to_path: Dict[str, List[int]]) -> Dict[str, Any]:
    layer1_set = set()
    layer1_to_layer2: Dict[str, set] = defaultdict(set)
    layer2_to_categories: Dict[str, List[str]] = defaultdict(list)

    for category, path in sorted(category_to_path.items()):
        if len(path) < 3:
            continue
        l1 = str(path[0])
        l2 = f"{path[0]}-{path[1]}"
        layer1_set.add(l1)
        layer1_to_layer2[l1].add(l2)
        layer2_to_categories[l2].append(category)

    return {
        "layer1": sorted(layer1_set, key=int),
        "layer1_to_layer2": {k: sorted(v) for k, v in sorted(layer1_to_layer2.items(), key=lambda x: int(x[0]))},
        "layer2_to_categories": {k: v for k, v in sorted(layer2_to_categories.items())},
    }


def save_hierarchy_json(
    out_path: str,
    cluster_dir: str,
    root: Tuple[str, ...],
    category_to_path: Dict[str, List[int]],
) -> None:
    max_depth = max((len(v) for v in category_to_path.values()), default=0)
    labels_by_level: Dict[str, List[int]] = {}
    for level in range(max_depth):
        labels = sorted({path[level] for path in category_to_path.values() if len(path) > level})
        labels_by_level[f"level{level + 1}"] = labels

    payload = {
        "cluster_dir": cluster_dir,
        "root_token": list(root),
        "num_categories": len(category_to_path),
        "max_depth": max_depth,
        "labels_by_level": labels_by_level,
        "layer_connections": build_layer_connections(category_to_path),
        "category_routes": build_category_routes(category_to_path),
        "category_paths": category_to_path,
        "tree": build_tree_from_paths(category_to_path),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_mashup_entry(entry: Any) -> Tuple[List[str], str, List[str], Optional[int]]:
    if not isinstance(entry, list):
        return [], "", [], None

    categories = [str(x) for x in entry[0]] if len(entry) > 0 and isinstance(entry[0], list) else []
    description = str(entry[1]) if len(entry) > 1 and isinstance(entry[1], str) else ""
    apis = [str(x) for x in entry[2]] if len(entry) > 2 and isinstance(entry[2], list) else []
    numeric_id = entry[3] if len(entry) > 3 and isinstance(entry[3], int) else None
    return categories, description, apis, numeric_id


def canonicalize_category(category: str) -> str:
    s = category.strip().lower().replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def build_canonical_lookup(category_to_path: Dict[str, List[int]]) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for cat in category_to_path.keys():
        grouped[canonicalize_category(cat)].append(cat)

    lookup: Dict[str, str] = {}
    for key, values in grouped.items():
        if len(values) == 1:
            lookup[key] = values[0]
    return lookup


def resolve_category(
    category: str,
    category_to_path: Dict[str, List[int]],
    alias_map: Dict[str, str],
    canonical_lookup: Dict[str, str],
) -> Tuple[Optional[str], str]:
    if category in category_to_path:
        return category, "exact"

    if category in alias_map:
        target = alias_map[category]
        if target in category_to_path:
            return target, "alias"

    canon = canonicalize_category(category)
    target = canonical_lookup.get(canon)
    if target:
        return target, "canonical"

    if canon.endswith("s"):
        target = canonical_lookup.get(canon[:-1])
        if target:
            return target, "canonical-singular"
    else:
        target = canonical_lookup.get(canon + "s")
        if target:
            return target, "canonical-plural"

    return None, "unknown"


def format_entry_legacy(
    categories: List[str],
    description: str,
    apis: List[str],
    numeric_id: Optional[int],
    category_hierarchy_paths: Dict[str, List[int]],
    answer_labels_by_level: Dict[str, List[int]],
    unknown_categories: List[str],
    resolved_category_map: Dict[str, str],
) -> List[Any]:
    payload = [
        categories,
        description,
        apis,
        numeric_id,
        {
            "category_hierarchy_paths": category_hierarchy_paths,
            "answer_labels_by_level": answer_labels_by_level,
            "unknown_categories": unknown_categories,
            "resolved_category_map": resolved_category_map,
        },
    ]
    return payload


def enrich_test_data(
    mashup_data: Dict[str, Any],
    category_to_path: Dict[str, List[int]],
    alias_map: Dict[str, str],
    output_format: str,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, int]]:
    enriched: Dict[str, Any] = {}

    known_categories = 0
    unknown_categories = 0
    resolve_method_counter: Counter = Counter()
    unknown_counter: Counter = Counter()

    canonical_lookup = build_canonical_lookup(category_to_path)

    for mashup_name, entry in mashup_data.items():
        categories, description, apis, numeric_id = parse_mashup_entry(entry)

        category_hierarchy_paths: Dict[str, List[int]] = {}
        unknown: List[str] = []
        level_labels: Dict[str, set] = defaultdict(set)
        resolved_category_map: Dict[str, str] = {}

        for category in categories:
            resolved, method = resolve_category(category, category_to_path, alias_map, canonical_lookup)
            resolve_method_counter[method] += 1
            if resolved is None:
                unknown.append(category)
                unknown_categories += 1
                unknown_counter[category] += 1
                continue

            known_categories += 1
            resolved_category_map[category] = resolved
            path = category_to_path[resolved]
            category_hierarchy_paths[category] = path
            for idx, label in enumerate(path, start=1):
                level_labels[f"level{idx}"].add(label)

        answer_labels_by_level = {
            level: sorted(list(labels))
            for level, labels in sorted(level_labels.items(), key=lambda x: int(x[0].replace("level", "")))
        }

        if output_format == "legacy":
            enriched[mashup_name] = format_entry_legacy(
                categories=categories,
                description=description,
                apis=apis,
                numeric_id=numeric_id,
                category_hierarchy_paths=category_hierarchy_paths,
                answer_labels_by_level=answer_labels_by_level,
                unknown_categories=sorted(set(unknown)),
                resolved_category_map=resolved_category_map,
            )
        else:
            enriched[mashup_name] = {
                "categories": categories,
                "description": description,
                "apis": apis,
                "id": numeric_id,
                "category_hierarchy_paths": category_hierarchy_paths,
                "answer_labels_by_level": answer_labels_by_level,
                "unknown_categories": sorted(set(unknown)),
                "resolved_category_map": resolved_category_map,
            }

    stats = {
        "num_mashups": len(mashup_data),
        "known_categories": known_categories,
        "unknown_categories": unknown_categories,
    }
    return enriched, stats, dict(resolve_method_counter), dict(unknown_counter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build category hierarchy JSON and enrich mashup test data with hierarchical labels.")
    parser.add_argument("--cluster-dir", required=True, help="Directory containing clustering result JSON files.")
    parser.add_argument("--mashup-json", required=True, help="Input mashup test JSON path.")
    parser.add_argument("--out-base", default=str(HIERARCHY_RESULTS_DIR / "output"), help="Base output directory.")
    parser.add_argument("--out-hierarchy", default=None, help="Output hierarchy JSON path.")
    parser.add_argument("--out-test", default=None, help="Output enriched test JSON path.")
    parser.add_argument("--out-simple-hierarchy", default=None, help="Output simple hierarchy JSON path.")
    parser.add_argument("--output-format", choices=["legacy", "structured"], default="legacy", help="legacy keeps original list schema and appends hierarchy info at index 4.")
    parser.add_argument("--alias-json", default=None, help="Optional JSON file: {\"unknown_category\": \"known_cluster_category\"}")
    parser.add_argument("--out-unknown-report", default=None, help="Optional JSON file to dump unresolved category counts.")
    parser.add_argument("--root-file", default=None, help="Optional root cluster JSON (e.g., ..._results_12ver.json) to force root loading.")
    args = parser.parse_args()

    cluster_tag = os.path.basename(os.path.normpath(args.cluster_dir))
    out_dir = os.path.join(args.out_base, cluster_tag)
    os.makedirs(out_dir, exist_ok=True)
    mashup_stem = os.path.splitext(os.path.basename(args.mashup_json))[0]

    out_hierarchy = args.out_hierarchy or os.path.join(out_dir, f"category_hierarchy_{cluster_tag}.json")
    out_test = args.out_test or os.path.join(out_dir, f"{mashup_stem}_with_hierarchy_{cluster_tag}.json")
    out_simple_hierarchy = args.out_simple_hierarchy or os.path.join(out_dir, f"simple_hierarchy_{cluster_tag}.json")
    out_unknown_report = args.out_unknown_report or os.path.join(out_dir, f"unknown_categories_{cluster_tag}.json")

    token_to_clusters = load_cluster_maps(args.cluster_dir)
    if args.root_file:
        loaded = load_single_cluster_file(args.root_file)
        if loaded is None:
            raise ValueError(f"Invalid root-file: {args.root_file}")
        token_to_clusters[loaded[0]] = loaded[1]
    root, category_to_path = build_category_paths(token_to_clusters)

    save_hierarchy_json(
        out_path=out_hierarchy,
        cluster_dir=args.cluster_dir,
        root=root,
        category_to_path=category_to_path,
    )

    with open(args.mashup_json, "r", encoding="utf-8") as f:
        mashup_data = json.load(f)

    if not isinstance(mashup_data, dict):
        raise ValueError("mashup_json must be a dict keyed by mashup name.")

    alias_map: Dict[str, str] = {}
    if args.alias_json:
        with open(args.alias_json, "r", encoding="utf-8") as f:
            raw_alias = json.load(f)
        if not isinstance(raw_alias, dict):
            raise ValueError("alias_json must be a dict")
        alias_map = {str(k): str(v) for k, v in raw_alias.items()}

    enriched, stats, resolve_stats, unknown_counter = enrich_test_data(
        mashup_data=mashup_data,
        category_to_path=category_to_path,
        alias_map=alias_map,
        output_format=args.output_format,
    )

    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    if out_simple_hierarchy:
        simple = build_simple_hierarchy(category_to_path)
        with open(out_simple_hierarchy, "w", encoding="utf-8") as f:
            json.dump(simple, f, ensure_ascii=False, indent=2)

    if out_unknown_report:
        with open(out_unknown_report, "w", encoding="utf-8") as f:
            json.dump({
                "unknown_category_counts": dict(sorted(unknown_counter.items(), key=lambda x: (-x[1], x[0]))),
                "resolve_method_stats": resolve_stats,
            }, f, ensure_ascii=False, indent=2)

    print("Output dir:", out_dir)
    print("Hierarchy JSON:", out_hierarchy)
    print("Enriched test data:", out_test)
    if out_simple_hierarchy:
        print("Simple hierarchy JSON:", out_simple_hierarchy)
    if out_unknown_report:
        print("Unknown report:", out_unknown_report)
    print("Root token:", "_".join(root))
    print("Categories in hierarchy:", len(category_to_path))
    print("Mashups:", stats["num_mashups"])
    print("Known category labels:", stats["known_categories"])
    print("Unknown category labels:", stats["unknown_categories"])
    print("Resolve stats:", resolve_stats)


if __name__ == "__main__":
    main()
