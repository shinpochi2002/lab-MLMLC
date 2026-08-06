import json
import random
from pathlib import Path

# ===== 設定（必要に応じて変更） =====
ROOT_DIR = Path("/home/nakagawa/out_by_category")  # 対象ディレクトリ
GLOB_PATTERN = "*.json"         # 対象ファイルパターン（再帰なら "**/*.json"）
TEST_RATIO = 0.20               # テスト比率
SEED = 42                       # 再現性のためのシード（不要なら None に）
ID_KEY = "mashup"               # 一意IDとして使うフィールド
# ====================================

if SEED is not None:
    random.seed(SEED)

def is_already_split(path: Path) -> bool:
    return path.name.endswith("_test_data.json") or path.name.endswith("_study_data.json")

def split_file(path: Path):
    if is_already_split(path):
        print(f"skip (already split): {path.name}")
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"read error: {path.name} -> {e}")
        return

    # 期待する構造かチェック
    if not isinstance(obj, dict) or "items" not in obj or not isinstance(obj["items"], list):
        print(f"skip (unexpected schema): {path.name}")
        return

    items = [it for it in obj["items"] if isinstance(it, dict) and ID_KEY in it]
    n = len(items)

    # 件数が1件以下はパス
    if n <= 1:
        print(f"pass (items <= 1): {path.name} (n={n})")
        return

    # テスト件数の決定（全部がテストにならないように調整）
    n_test = int(round(n * TEST_RATIO))
    n_test = max(1, min(n_test, n - 1))

    ids = [it[ID_KEY] for it in items]
    test_ids = set(random.sample(ids, n_test))

    test_items = [it for it in items if it[ID_KEY] in test_ids]
    study_items = [it for it in items if it[ID_KEY] not in test_ids]

    # 出力オブジェクト（元の category を引き継ぐ／存在しなければ None）
    category = obj.get("category")

    out_test_obj = {
        "category": category,
        "count": len(test_items),
        "items": test_items,
    }
    out_study_obj = {
        "category": category,
        "count": len(study_items),
        "items": study_items,
    }

    stem = path.stem
    out_test = path.with_name(f"{stem}_test_data.json")
    out_study = path.with_name(f"{stem}_study_data.json")

    try:
        with out_test.open("w", encoding="utf-8") as f:
            json.dump(out_test_obj, f, indent=4, ensure_ascii=False)
        with out_study.open("w", encoding="utf-8") as f:
            json.dump(out_study_obj, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"write error: {path.name} -> {e}")
        return

    print(f"[OK] {path.name}: total={n}, test={len(test_items)}, study={len(study_items)}")
    print(f" -> {out_test.name}, {out_study.name}")

def main():
    files = sorted(ROOT_DIR.glob(GLOB_PATTERN))
    if not files:
        print(f"No files matched: {ROOT_DIR}/{GLOB_PATTERN}")
        return
    for p in files:
        split_file(p)

if __name__ == "__main__":
    main()
