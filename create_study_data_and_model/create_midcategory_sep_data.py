import json
from collections import defaultdict

# すでに midcategory が付与された JSON ファイルを読み込む
with open("/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_3layer_12a6a2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# bigcategory ごとに分類
grouped = defaultdict(list)
for item in data:
    midcat = item.get("middlecategory")
    if midcat is not None:
        grouped[midcat].append(item)
    else:
        print(f"[警告] middlecategory が存在しない項目: {item}")

# 各 bigcategory ごとにファイル出力
for midcat, items in grouped.items():
    new_mid = midcat.replace("-", "_")
    filename = f"/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_3layer_12a6a2_{new_mid}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"{filename} を出力しました（{len(items)} 件）")

