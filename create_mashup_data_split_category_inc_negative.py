import json, os, re, sys, random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# ====== 設定（ここを書き換える） =========================================
MASHUPS_JSON = "/home/nakagawa/study_data_mashup_apis.json"  # 複合サービス（マッシュアップ）JSON
API2CAT_JSON  = "/home/nakagawa/api_category_pairs.json"              # API→カテゴリ対応表JSON
OUTDIR        = "./out_by_category_inc_negative"                          # 出力ディレクトリ

MIN_APIS_PER_ITEM = 1     # 正例: そのカテゴリに属するAPIがこの数未満の項目は除外
NEG_PER_POS       = 3     # 負例: 正例1に対して追加する負例の数（比率 1:NEG_PER_POS）
PRETTY_JSON       = True  # 出力をインデント整形するか
RANDOM_SEED       = 42    # サンプリングの再現性用シード
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
        cats = [str(x) for x in entry.get("categories", []) if isinstance(x, (str,int,float,str))]
        desc = str(entry.get("description", ""))
        raw = entry.get("apis") or entry.get("used_apis") or []
        if isinstance(raw, list):
            apis = [str(x) for x in raw]
    return cats, desc, apis

def main():
    rng = random.Random(RANDOM_SEED)

    mashups: Dict[str, Any] = load_json(MASHUPS_JSON)
    api2cat_raw: Dict[str, Any] = load_json(API2CAT_JSON)

    # API名の正規化 → カテゴリ（複数可）マップ
    api2cats: Dict[str, List[str]] = {}
    for api, cat in api2cat_raw.items():
        k = normalize_api(api)
        cats = [str(cat)] if not isinstance(cat, list) else [str(c) for c in cat]
        api2cats[k] = sorted(set(cats))

    # 1件ずつパースして、各マッシュアップにおける「カテゴリ→そのカテゴリに属する使用API」を作っておく
    per_mashup_per_cat: Dict[str, Dict[str, List[str]]] = {}
    parsed_cache: Dict[str, Tuple[List[str], str, List[str]]] = {}
    total_refs = 0
    missing = 0

    for mashup_key, entry in mashups.items():
        orig_cats, desc, used_apis = parse_mashup_entry(entry)
        parsed_cache[mashup_key] = (orig_cats, desc, used_apis)

        cat_map: Dict[str, List[str]] = defaultdict(list)
        for api in used_apis:
            total_refs += 1
            cats = api2cats.get(normalize_api(api))
            if not cats:
                missing += 1
                continue
            for c in cats:
                cat_map[c].append(api)
        # 重複を削る
        for c in list(cat_map.keys()):
            cat_map[c] = sorted(set(cat_map[c]))
        per_mashup_per_cat[mashup_key] = cat_map

    # 正例バケットと、正例に採用されたマッシュアップ集合
    pos_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pos_keys_by_cat: Dict[str, set] = defaultdict(set)

    # 正例の構築（従来どおり MIN_APIS_PER_ITEM でフィルタ）
    for mashup_key, cat_map in per_mashup_per_cat.items():
        orig_cats, desc, _ = parsed_cache[mashup_key]
        for c, apis_in_c in cat_map.items():
            if len(apis_in_c) < MIN_APIS_PER_ITEM:
                continue
            item = {
                "mashup": mashup_key,
                "description": desc,
                "original_categories": orig_cats,
                "used_apis": apis_in_c,  # このカテゴリに属するAPIのみ
                "label": 1               # 正例
            }
            pos_buckets[c].append(item)
            pos_keys_by_cat[c].add(mashup_key)

    # すべてのマッシュアップキー（負例候補の母集団）
    all_mashup_keys = set(mashups.keys())

    # 書き出し（カテゴリごとに、正例 + 負例を所定比率で結合）
    for c, pos_items in pos_buckets.items():
        pos_count = len(pos_items)
        need_negs = pos_count * NEG_PER_POS

        # 負例候補: そのカテゴリ c に属するAPIが「1つも無い」マッシュアップ
        neg_candidates = []
        for key in all_mashup_keys:
            # 正例に入っているものは負例には使わない
            if key in pos_keys_by_cat[c]:
                continue
            cat_map = per_mashup_per_cat.get(key, {})
            has_c = (c in cat_map) and len(cat_map[c]) > 0
            if not has_c:
                neg_candidates.append(key)

        # シャッフルして必要数だけサンプル
        rng.shuffle(neg_candidates)
        chosen_negs = neg_candidates[:need_negs]

        neg_items: List[Dict[str, Any]] = []
        for key in chosen_negs:
            orig_cats, desc, _used_apis_all = parsed_cache[key]
            neg_items.append({
                "mashup": key,
                "description": desc,
                "original_categories": orig_cats,
                "used_apis": [],  # ラベル無し = 当該カテゴリのAPIが無い
                "label": 0        # 負例
            })

        # 出力オブジェクト
        items = pos_items + neg_items
        out_obj = {
            "category": c,
            "count": len(items),           # 合計件数（正+負）
            "positive_count": pos_count,   # 参考: 正例件数
            "negative_count": len(neg_items),  # 参考: 負例件数
            "pos_neg_ratio": f"1:{NEG_PER_POS}",
            "items": items
        }

        out_path = os.path.join(OUTDIR, f"study_mashup_data_category_split_inc_negative_{slugify(c)}.json")
        save_json(out_path, out_obj, pretty=PRETTY_JSON)
        print(f"{c}  pos={pos_count}  neg={len(neg_items)}")

    # ログ
    print(f"[INFO] Categories written: {len(pos_buckets)}", file=sys.stderr)
    print(f"[INFO] Total mashups: {len(mashups)}", file=sys.stderr)
    print(f"[INFO] Total API refs: {total_refs}", file=sys.stderr)
    print(f"[INFO] Missing API mappings: {missing}", file=sys.stderr)
    if missing:
        print("[HINT] API名の表記ゆれ（大文字小文字・ハイフン/空白等）に注意。必要なら API→カテゴリ 対応表を整備してください。", file=sys.stderr)

if __name__ == "__main__":
    main()
