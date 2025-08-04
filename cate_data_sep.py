import json
from collections import defaultdict


# クラスタ情報を読み込み
with open('/home/nakagawa/category_soft_clu_balanced_balanced_kmeans+_results_12ver.json', 'r', encoding='utf-8') as f:
    categorized_clusters = json.load(f)

# クラスタ情報を取得
clusters = categorized_clusters.get("clusters", {})

# カテゴリーとクラスタIDの対応関係をマッピング
category_to_cluster = {}
for cluster_id, categories in clusters.items():
    for category in categories:
        category_to_cluster[category] = int(cluster_id)

# 元のデータを読み込み
with open('softfill_mashup_data_new_api_category_inclu_nomash_split_all_inc_8big_test_data.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# クラスタごとにAPI名を収集するための辞書を作成
cluster_to_apis = defaultdict(set)

# 各APIのカテゴリーをクラスタIDに置き換える
filtered_data = []  # 有効なエントリのみを保持するリスト

for api_entry in original_data:
    # 現在のカテゴリー情報を取得
    category = api_entry["category"]
    
    # カテゴリーをクラスタIDに置き換え
    if category in category_to_cluster:
        cluster_id = category_to_cluster[category]
        api_entry["bigcluster"] = cluster_id
        
        # クラスタごとにAPI名を追加 (重複を防止)
        cluster_to_apis[cluster_id].add(api_entry["name"])
        
        # 有効なエントリを保持
        filtered_data.append(api_entry)
    else:
        filtered_data.append(api_entry)

# 辞書をリスト形式に変換
cluster_to_apis = {key: list(value) for key, value in cluster_to_apis.items()}

# 修正後のデータを保存
with open('softfill_mashup_data_new_api_category_inclu_nomash_split_all_inc_8big_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

# # クラスタごとに収集されたAPI名を保存
# with open('8_6_cluster_to_apis_new.json', 'w', encoding='utf-8') as f:
#     json.dump(cluster_to_apis, f, ensure_ascii=False, indent=4)

print("カテゴリーがクラスタIDに置き換えられました。")