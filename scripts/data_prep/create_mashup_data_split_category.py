
import json, os, re, sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# ====== 設定（ここを書き換える） =========================================
MASHUPS_JSON = legacy_path("/home/nakagawa/study_data_mashup_nomashup_apis.json")            # 複合サービス（マッシュアップ）JSON
API2CAT_JSON  = legacy_path("/home/nakagawa/api_category_pairs.json")   # API→カテゴリ対応表JSON
OUTDIR        = legacy_path("./out_by_category_nomashup")               # 出力ディレクトリ
MIN_APIS_PER_ITEM = 1                             # そのカテゴリに属するAPIがこの数未満の項目は除外
PRETTY_JSON   = True                              # 出力をインデント整形するか
# =======================================================================

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-一-龯ぁ-んァ-ン]", "_", s)
    return s[:100] or "unknown"

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any, pretty: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

def normalize_api(name: str) -> str:
    return (name or "").strip().lower()

def parse_mashup_entry(entry: Any) -> Tuple[List[str], str, List[str]]:
    """(original_categories, description, used_apis) を取り出す（配列/辞書どちらでもOK）"""
    cats, desc, apis = [], "", []
    if isinstance(entry, list):
        if len(entry) >= 1 and isinstance(entry[0], list):
            cats = [str(x) for x in entry[0]]
        if len(entry) >= 2 and isinstance(entry[1], str):
            desc = entry[1]
        if len(entry) >= 3 and isinstance(entry[2], list):
            apis = [str(x) for x in entry[2]]
    elif isinstance(entry, dict):
        cats = [str(x) for x in entry.get("categories", []) if isinstance(x, (str,int,float))]
        desc = str(entry.get("description", ""))
        raw = entry.get("apis") or entry.get("used_apis") or []
        if isinstance(raw, list):
            apis = [str(x) for x in raw]
    return cats, desc, apis

def main():
    mashups: Dict[str, Any] = load_json(MASHUPS_JSON)
    api2cat_raw: Dict[str, Any] = load_json(API2CAT_JSON)

    # API名の正規化 → カテゴリ（複数可）マップ
    api2cats = {}
    for api, cat in api2cat_raw.items():
        k = normalize_api(api)
        cats = [str(cat)] if not isinstance(cat, list) else [str(c) for c in cat]
        api2cats[k] = sorted(set(cats))

    buckets = defaultdict(list)  # category -> [items...]
    count = {}
    missing, total_refs = 0, 0

    for mashup_key, entry in mashups.items():
        orig_cats, desc, used_apis = parse_mashup_entry(entry)

        # そのマッシュアップ中で、カテゴリごとに属するAPIだけを振り分ける
        per_cat = defaultdict(list)
        for api in used_apis:
            total_refs += 1
            cats = api2cats.get(normalize_api(api))
            if not cats:
                missing += 1
                continue
            for c in cats:
                per_cat[c].append(api)

        # 各カテゴリに出力（MIN_APIS_PER_ITEM 未満ならスキップ）
        for c, apis_in_c in per_cat.items():
            apis_in_c = sorted(set(apis_in_c))
            if len(apis_in_c) < MIN_APIS_PER_ITEM:
                continue
            buckets[c].append({
                "mashup": mashup_key,
                "description": desc,
                "original_categories": orig_cats,
                "used_apis": apis_in_c,  # ← このカテゴリに属するAPIのみ
            })

    # 書き出し
    for c, items in buckets.items():
        out_path = os.path.join(OUTDIR, f"study_mashup_nomashup_data_category_split_{slugify(c)}.json")
        save_json(out_path, {"category": c, "count": len(items), "items": items}, pretty=PRETTY_JSON)
        print(c,len(items))
    
 

    # 簡易ログ
    print(f"[INFO] Categories written: {len(buckets)}", file=sys.stderr)
    print(f"[INFO] Total mashups: {len(mashups)}", file=sys.stderr)
    print(f"[INFO] Total API refs: {total_refs}", file=sys.stderr)
    print(f"[INFO] Missing API mappings: {missing}", file=sys.stderr)
    if missing:
        print("[HINT] API名の表記ゆれ（大文字小文字・ハイフン/空白等）に注意。必要なら API→カテゴリ 対応表を整備してください。", file=sys.stderr)

if __name__ == "__main__":
    main()