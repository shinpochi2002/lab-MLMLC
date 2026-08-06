#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# -*- coding: utf-8 -*-

from pathlib import Path
import json
import csv
from typing import Any, Dict, List, Tuple, Set

# ========= ここだけ書き換えればOK（絶対パス/相対パスどちらでも可） =========
MANIFEST_PATH = Path(legacy_path("/home/nakagawa/softfill_mashup_data_new_api/nocategory/softfill_mashup_new_data.json"))            # 1つ目のJSON（使用APIが index=2）
CATALOG_PATH  = Path(legacy_path("/home/nakagawa/new_api_data/api_data_new.json"))                # 2つ目のJSON（カテゴリ配列が index=0）
OUT_CSV       = Path("./api_category_pairs.csv")   # 出力CSV
OUT_JSON      = Path("./api_category_pairs.json")  # 出力JSON（{api: first_category}）
# 構造が違う場合はインデックスを変更
MANIFEST_USED_APIS_INDEX = 2   # マニフェストの「使用API配列」がある位置
CATALOG_CATEGORIES_INDEX = 0   # カタログの「カテゴリ配列」がある位置
# =========================================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSONファイルが見つかりません: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_used_apis(manifest: Dict[str, Any], used_index: int = 2) -> List[str]:
    """
    マニフェストの各値（list想定）の used_index に使用API名（配列 or 文字列）が入っている前提で抽出
    """
    used: Set[str] = set()
    for _, v in manifest.items():
        if isinstance(v, list) and len(v) > used_index:
            apis = v[used_index]
            if isinstance(apis, list):
                for a in apis:
                    if isinstance(a, str) and a.strip():
                        used.add(a.strip())
            elif isinstance(apis, str) and apis.strip():
                used.add(apis.strip())
    return sorted(used)

def pick_first_category(catalog_entry: Any, categories_index: int = 0) -> str:
    """
    カタログ側の各値（list想定）について、categories_index がカテゴリ配列。
    その先頭カテゴリ（文字列）を返す。取れなければ空文字。
    """
    if not isinstance(catalog_entry, list) or len(catalog_entry) <= categories_index:
        return ""
    categories = catalog_entry[categories_index]
    if isinstance(categories, list) and categories and isinstance(categories[0], str):
        return categories[0].strip()
    return ""

def map_api_to_first_category(
    apis: List[str],
    catalog: Dict[str, Any],
    categories_index: int = 0
) -> Tuple[List[Tuple[str, str]], List[str]]:
    pairs: List[Tuple[str, str]] = []
    missing: List[str] = []
    for api in apis:
        entry = catalog.get(api)
        first_cat = pick_first_category(entry, categories_index) if entry is not None else ""
        if first_cat:
            pairs.append((api, first_cat))
        else:
            missing.append(api)
    return pairs, missing

def save_pairs_csv(pairs: List[Tuple[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["api", "first_category"])
        writer.writerows(pairs)

def save_pairs_json(pairs: List[Tuple[str, str]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {api: cat for api, cat in pairs}
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 入力読込
    manifest = load_json(MANIFEST_PATH)
    catalog  = load_json(CATALOG_PATH)

    # 使用API抽出 → 先頭カテゴリ付与
    used_apis = extract_used_apis(manifest, MANIFEST_USED_APIS_INDEX)
    pairs, missing = map_api_to_first_category(used_apis, catalog, CATALOG_CATEGORIES_INDEX)

    # 出力
    save_pairs_csv(pairs, OUT_CSV)
    save_pairs_json(pairs, OUT_JSON)

    # 結果表示
    print(f"[INFO] used APIs: {len(used_apis)}")
    print(f"[INFO] paired   : {len(pairs)} -> {OUT_CSV} , {OUT_JSON}")
    if missing:
        print("[WARN] カタログ未ヒット or 先頭カテゴリが取れなかったAPI:")
        for api in missing:
            print(f"  - {api}")

if __name__ == "__main__":
    main()
