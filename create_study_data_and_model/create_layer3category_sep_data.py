import json
from collections import defaultdict

# すでに midcategory が付与された JSON ファイルを読み込む
with open("/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_12a2a2a3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# bigcategory ごとに分類
grouped = defaultdict(list)
for item in data:
    midcat = item.get("layer3category")
    if midcat is not None:
        grouped[midcat].append(item)
    else:
        print(f"[警告] layer3category が存在しない項目: {item}")

# 各 bigcategory ごとにファイル出力
for l3cat, items in grouped.items():
    new_l3 = l3cat.replace("-", "_")
    filename = f"/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_12a2a2a3_{new_l3}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"{filename} を出力しました（{len(items)} 件）")

