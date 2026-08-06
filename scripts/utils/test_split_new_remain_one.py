import json
import random
from pathlib import Path
from collections import defaultdict

# ===== 設定 =====
SOURCE_DIR = Path("/home/nakagawa/out_by_category_inc_negative_1_10")
TRAIN_DIR = Path("/home/nakagawa/train_by_category_inc_negative_1_10_new")
TEST_DIR = Path("/home/nakagawa/test_by_category_inc_negative_1_10_new")
TEST_RATIO = 0.2
random.seed(42)

# ===== 関数 =====
def extract_items(data):
    """JSONからitems（またはdata）リストを抽出"""
    if isinstance(data, dict):
        # 一般的に 'items' か 'data' を持つ
        if "items" in data and isinstance(data["items"], list):
            return data["items"], "items"
        elif "data" in data and isinstance(data["data"], list):
            return data["data"], "data"
        else:
            # dictの中にlist型valueがあればそれを使う（保険）
            for k, v in data.items():
                if isinstance(v, list):
                    return v, k
    elif isinstance(data, list):
        # 純粋にリストだけのファイルにも対応
        return data, None
    return None, None


def split_and_preserve_labels(file_path: Path):
    """used_apisを1つ以上trainに残すようにデータを分割"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items, list_key = extract_items(data)
    if not items:
        print(f"[WARN] {file_path.name} の形式が不正です。スキップします。")
        return

    # --- 1. used_api出現場所を収集 ---
    api_to_indices = defaultdict(list)
    for idx, item in enumerate(items):
        used_apis = item.get("used_apis", [])
        if not isinstance(used_apis, list):
            continue
        for api in used_apis:
            api_to_indices[api].append(idx)

    # --- 2. 出現1回のみのAPIはtrain固定 ---
    must_train_indices = {idx for idxs in api_to_indices.values() if len(idxs) == 1 for idx in idxs}

    # --- 3. 通常分割 ---
    all_indices = set(range(len(items)))
    remaining_indices = list(all_indices - must_train_indices)
    random.shuffle(remaining_indices)
    test_size = int(len(items) * TEST_RATIO)
    test_indices = set(remaining_indices[:test_size])
    train_indices = all_indices - test_indices

    # --- 4. 全APIがtrainに含まれるよう保証 ---
    train_apis = set()
    for i in train_indices:
        for api in items[i].get("used_apis", []):
            train_apis.add(api)

    missing_apis = []
    for api, idxs in api_to_indices.items():
        if api not in train_apis:
            idx = random.choice(idxs)
            test_indices.discard(idx)
            train_indices.add(idx)
            missing_apis.append(api)

    # --- 5. 出力構築 ---
    train_items = [items[i] for i in sorted(train_indices)]
    test_items = [items[i] for i in sorted(test_indices)]

    train_data = dict(data)
    test_data = dict(data)
    train_data[list_key] = train_items
    test_data[list_key] = test_items

    # --- 6. 保存 ---
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_DIR / file_path.name, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(TEST_DIR / file_path.name, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    # --- 7. ログ ---
    msg = f"[OK] {file_path.name}: {len(items)}件 → train {len(train_items)}件 / test {len(test_items)}件"
    if missing_apis:
        msg += f"（追加でtrainに移動: {len(missing_apis)} API）"
    print(msg)


def main():
    files = list(SOURCE_DIR.glob("*.json"))
    print(f"処理対象ファイル数: {len(files)}")
    for file in files:
        split_and_preserve_labels(file)
    print("✅ 全ファイルで used_apis が少なくとも1件 train に残るように分割完了。")


if __name__ == "__main__":
    main()
