import json
from collections import defaultdict

# すでに bigcategory が付与された JSON ファイルを読み込む
with open("/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_3layer_12a6a2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# bigcategory ごとに分類
grouped = defaultdict(list)
for item in data:
    bigcat = item.get("bigcategory")
    if bigcat is not None:
        grouped[bigcat].append(item)
    else:
        print(f"[警告] bigcategory が存在しない項目: {item}")

# 各 bigcategory ごとにファイル出力
for bigcat, items in grouped.items():
    filename = f"/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_3layer_12a6a2_{bigcat}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"{filename} を出力しました（{len(items)} 件）")

