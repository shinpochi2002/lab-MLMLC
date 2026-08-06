import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# クラスタ情報を読み込み
with open(legacy_path('/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_inc_8big_7mid.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# クラスタごとにデータを分ける関数
def split_by_cluster(data):
    cluster_dict = {}
    for item in data:
        cluster = item["middlecluster"]
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(item)
    return cluster_dict

# データをクラスタごとに分ける
clustered_data = split_by_cluster(data)

# 結果をJSONファイルに保存
for cluster, items in clustered_data.items():
    filename = f"softfill_mashup_data_new_api_category_8category_mid7_{cluster}_split_new.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    print(f"Cluster {cluster} data saved to {filename}")