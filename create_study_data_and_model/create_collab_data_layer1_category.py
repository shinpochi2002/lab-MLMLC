import json

# ファイル読み込み
with open("/home/nakagawa/collaboration_test_data.json", "r", encoding="utf-8") as f:
    service_data = json.load(f)

with open("/home/nakagawa/category_soft_clu_balanced_balanced_kmeans+_results_15ver.json", "r", encoding="utf-8") as f:
    cluster_data = json.load(f)

# サブカテゴリから大カテゴリ（クラスタ番号）を逆引きする辞書を作成
category_to_cluster = {}
for cluster_id, subcategories in cluster_data["clusters"].items():
    for cat in subcategories:
        category_to_cluster[cat] = int(cluster_id)

# サービスデータにクラスタ番号を追加し、不要キーを削除
updated_data = []
for item in service_data:
    # "category" がリストでなければリスト化
    categories = item["categories"]
    if not isinstance(categories, list):
        categories = [categories]

    # 対応するクラスタ番号を取得
    clusters = [category_to_cluster[cat] for cat in categories if cat in category_to_cluster]
    if clusters:
        item["bigcategory"] = list(set(clusters))  # 重複を除く

        # 不要なキーを削除
        item.pop("bigclusters", None)
        item.pop("middleclusters", None)

        updated_data.append(item)
    else:
        print(f"カテゴリがクラスタに見つかりません: {categories}")

# 結果を保存
with open("collaboration_test_data_layer1_15.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)
