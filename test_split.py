import json
import random
from pathlib import Path

# ===== 設定 =====
SOURCE_DIR = Path("/home/nakagawa/out_by_category_inc_negative_1_10")
TRAIN_DIR = Path("/home/nakagawa/train_by_category_inc_negative_1_10")
TEST_DIR = Path("/home/nakagawa/test_by_category_inc_negative_1_10")
TEST_RATIO = 0.2
random.seed(42)

# ===== 関数 =====
def extract_data(data):
    """元のフォーマットを保ちながら data リストを抽出"""
    if isinstance(data, list):
        # 純粋なリスト形式（まれ）
        return data, None
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                return value, key  # リストとそのキー名（例: "data"）を返す
    return None, None


def split_and_save_json(file_path):
    """ファイルを train/test に分割し、元と同じ構造で保存"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted, list_key = extract_data(data)
    if not extracted:
        print(f"[WARN] {file_path.name} スキップ（データ形式不明）")
        return

    random.shuffle(extracted)
    test_size = max(1, int(len(extracted) * TEST_RATIO))
    test_data = extracted[:test_size]
    train_data = extracted[test_size:]

    # 元構造を維持
    if list_key:
        train_json = dict(data)
        test_json = dict(data)
        train_json[list_key] = train_data
        test_json[list_key] = test_data
    else:
        train_json = train_data
        test_json = test_data

    # 出力フォルダ
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 保存
    with open(TRAIN_DIR / file_path.name, "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)
    with open(TEST_DIR / file_path.name, "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] {file_path.name}: {len(extracted)}件 → train {len(train_data)}件 / test {len(test_data)}件")


def main():
    files = list(SOURCE_DIR.glob("*.json"))
    print(f"処理対象ファイル数: {len(files)}")
    for file in files:
        split_and_save_json(file)
    print("✅ すべてのファイルを out_by_category_inc_negative_1_10/ と同形式で train/test に分割しました。")


if __name__ == "__main__":
    main()
