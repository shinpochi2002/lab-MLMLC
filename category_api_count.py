import os
import json

# ====== パス設定 ======
DIR_PATH = "/home/nakagawa/api_category_list/category_inc_negative_1_10"

# ====== 集計処理 ======
lengths = {}

for filename in sorted(os.listdir(DIR_PATH)):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(DIR_PATH, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        category = filename.replace("_api_labels.json", "")
        if isinstance(data, list):
            lengths[category] = len(data)
        elif isinstance(data, dict):
            # データがdict形式の場合、valueがリストならその長さを取る
            lengths[category] = sum(len(v) for v in data.values() if isinstance(v, list))
        else:
            lengths[category] = 0
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        lengths[filename] = 0

# ====== 結果表示 ======
print("=== Category-wise list lengths ===")
for cat, length in sorted(lengths.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat:35s} : {length}")

# ====== 合計件数 ======
total = sum(lengths.values())
print(f"\nTotal entries across all categories: {total}")
