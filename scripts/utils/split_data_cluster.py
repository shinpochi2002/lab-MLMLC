import json
from collections import defaultdict

# クラスター情報を category_cluster.json から読み込む
with open('/home/nakagawa/softfill_mashup_data_new_api_category_4category_3_split_new.json', 'r') as f:
    data = json.load(f)

# クラスタごとにデータを分類
clustered_data = defaultdict(list)
for item in data:
    clustered_data[item["middlecluster"]].append(item)

# クラスタごとにJSONファイルとして保存
for cluster, items in clustered_data.items():
    filename = f"softfill_mashup_data_new_api_category_4category_3_split_{cluster}_new.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    print(f"クラスタ {cluster} のデータを {filename} に保存しました。")