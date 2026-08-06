#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import csv
from typing import Any, Dict, List, Tuple, Set

# ========= ここだけ書き換えればOK =========
MANIFEST_PATH = Path("/home/nakagawa/softfill_mashup_data_new_api/nocategory/softfill_mashup_new_data.json")
CATALOG_PATH  = Path("/home/nakagawa/new_api_data/api_data_new.json")

OUT_CSV            = Path("./api_category_pairs_with_desc.csv")          # (api, first_category, api_description)
OUT_API_JSON       = Path("./api_dataset_like_manifest.json")            # APIのみ（softfill形式）
OUT_MERGED_JSON    = Path("./mashup_api_merged_softfill.json")           # マニフェスト + API を合体

# 構造が違う場合はインデックスを調整
MANIFEST_USED_APIS_INDEX  = 2   # マニフェスト側：使用API配列の位置
CATALOG_CATEGORIES_INDEX  = 0   # カタログ側　　：カテゴリ配列の位置
CATALOG_DESCRIPTION_INDEX = 1   # カタログ側　　：説明文の位置
# ==========================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSONファイルが見つかりません: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_used_apis(manifest: Dict[str, Any], used_index: int) -> List[str]:
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

def pick_first_category(catalog_entry: Any, categories_index: int) -> str:
    if not isinstance(catalog_entry, list) or len(catalog_entry) <= categories_index:
        return ""
    cats = catalog_entry[categories_index]
    if isinstance(cats, list) and cats and isinstance(cats[0], str):
        return cats[0].strip()
    return ""

def pick_description(catalog_entry: Any, description_index: int) -> str:
    if not isinstance(catalog_entry, list) or len(catalog_entry) <= description_index:
        return ""
    desc = catalog_entry[description_index]
    return desc.strip() if isinstance(desc, str) else ""

def map_api_to_cat_and_desc(
    apis: List[str],
    catalog: Dict[str, Any],
    categories_index: int,
    description_index: int,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    triples: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for api in apis:
        entry = catalog.get(api)
        if entry is None:
            missing.append(api)
            continue
        first_cat = pick_first_category(entry, categories_index)
        desc = pick_description(entry, description_index)
        if first_cat and desc:
            triples.append((api, first_cat, desc))
        else:
            missing.append(api)
    return triples, missing

def save_pairs_csv_with_desc(triples: List[Tuple[str, str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["api", "first_category", "api_description"])
        w.writerows(triples)

def build_api_softfill_items(triples: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """
    APIを softfill の4要素に揃える
      0: [first_category]
      1: api_description
      2: [api]            ← ユーザ要望どおり単一要素でAPI名
      3: 1
    """
    out: Dict[str, Any] = {}
    for api, first_cat, desc in triples:
        out[api] = [
            [first_cat],
            desc,
            [api],
            1
        ]
    return out

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def make_merged_manifest(manifest: Dict[str, Any], api_items: Dict[str, Any]) -> Dict[str, Any]:
    """
    マニフェスト(mashup) と APIデータ(api_items) を同じトップレベルにまとめる。
    キー衝突時は API 側を '<api>__api' にリネーム。
    """
    merged = dict(manifest)  # まずは元マニフェストをそのままコピー
    for api_key, item in api_items.items():
        key = api_key
        if key in merged:
            key = f"{api_key}__api"
        merged[key] = item
    return merged

def main():
    manifest = load_json(MANIFEST_PATH)  # {mashup_name: [cats, desc, used_apis, count], ...}
    catalog  = load_json(CATALOG_PATH)   # {api_name: [categories, api_description, ...], ...}

    used_apis = extract_used_apis(manifest, MANIFEST_USED_APIS_INDEX)

    triples, missing = map_api_to_cat_and_desc(
        used_apis, catalog, CATALOG_CATEGORIES_INDEX, CATALOG_DESCRIPTION_INDEX
    )

    # 1) APIリストの対（カテゴリ・説明）をCSVに
    save_pairs_csv_with_desc(triples, OUT_CSV)

    # 2) APIのみ（softfill互換の4要素）
    api_items = build_api_softfill_items(triples)
    save_json(api_items, OUT_API_JSON)

    # 3) マニフェスト + API を合成
    merged = make_merged_manifest(manifest, api_items)
    save_json(merged, OUT_MERGED_JSON)

    # ログ
    print(f"[INFO] mashups (manifest)   : {len(manifest)}")
    print(f"[INFO] used APIs in manifest: {len(used_apis)}")
    print(f"[INFO] API triples ok       : {len(triples)} -> {OUT_CSV}")
    print(f"[INFO] API softfill JSON    : {len(api_items)} -> {OUT_API_JSON}")
    print(f"[INFO] MERGED (mashup+api)  : {len(merged)} -> {OUT_MERGED_JSON}")
    if missing:
        print("[WARN] 情報不足（カタログ未ヒット/カテゴリor説明なし）のAPI:")
        for api in missing:
            print(f"  - {api}")

if __name__ == "__main__":
    main()
