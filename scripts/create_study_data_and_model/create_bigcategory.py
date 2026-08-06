import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# ファイル読み込み
with open(legacy_path("/home/nakagawa/lab-MLMLC/studydata/original/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data.json"), "r", encoding="utf-8") as f:
    service_data = json.load(f)

with open(legacy_path("/home/nakagawa/lab-MLMLC/cluster/8cluster/category_soft_clu_balanced_balanced_kmeans+_results_8ver.json"), "r", encoding="utf-8") as f:
    cluster_data = json.load(f)

# サブカテゴリから大カテゴリ（クラスタ番号）を逆引きする辞書を作成
category_to_cluster = {}
for cluster_id, subcategories in cluster_data["clusters"].items():
    for cat in subcategories:
        category_to_cluster[cat] = cluster_id

# サービスデータにクラスタ番号を追加
updated_data = []
for item in service_data:
    category = item["category"]
    cluster = category_to_cluster.get(category)  # クラスタ番号を取得（見つからなければNone）
    if cluster is not None:
        item["bigcategory"] = cluster
        updated_data.append(item)
    else:
        print(f"カテゴリがクラスタに見つかりません: {category}")

# 結果を保存
with open("softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)
