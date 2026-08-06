import json

# ファイルを読み込む
input_file = "softclu_hierarchical_results_5_sorted.txt"
output_file = "hierarchical_5_category_cluster.json"

data = {}

# ファイルの内容を処理
with open(input_file, "r") as file:
    for line in file:
        # 各行のフォーマットが「Category: category_name -> Cluster: cluster_number」と仮定
        if "Category" in line and "Cluster" in line:
            # 行を分割してカテゴリとクラスターを取得
            category_part, cluster_part = line.split("->")
            category = category_part.split(":")[1].strip()
            cluster = cluster_part.split(":")[1].strip()

            # カテゴリー名: クラスター番号の形にする
            data[category] = int(cluster)

# JSONファイルに書き出す
with open(output_file, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSONファイル '{output_file}' に変換されました。")
