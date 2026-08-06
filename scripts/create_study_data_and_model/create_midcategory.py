import json
import os

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# 入力ファイル（bigcategory付き）
with open(legacy_path("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3.json"), "r", encoding="utf-8") as f:
    service_data = json.load(f)

# middle定義ファイルをキャッシュする辞書（bigcategoryごとに読み込む）
middle_maps = {}

def load_middle_map(bigcategory):
    filename = legacy_path(f"/home/nakagawa/lab-MLMLC/cluster/8_3_2_3/category_soft_clu_balanced_balanced_kmeans+_results_8_{bigcategory}_layer4_8_3_2_3ver.json")
    if not os.path.exists(filename):
        print(f"[警告] middle cluster ファイルが存在しません: {filename}")
        return {}

    with open(filename, "r", encoding="utf-8") as f:
        middle_data = json.load(f)

    cat_to_middle = {}
    for mid_id, cats in middle_data["clusters"].items():
        for cat in cats:
            cat_to_middle[cat] = mid_id
    return cat_to_middle

# middlecategory を追加（形式は "big-middle"）
updated_data = []
for item in service_data:
    category = item["category"]
    bigcategory = item.get("bigcategory")

    if bigcategory is None:
        print(f"[警告] bigcategory が存在しない: {item}")
        item["middlecategory"] = "unknown"
        updated_data.append(item)
        continue

    if bigcategory not in middle_maps:
        middle_maps[bigcategory] = load_middle_map(bigcategory)

    cat_to_mid = middle_maps[bigcategory]
    middle_id = cat_to_mid.get(category)

    if middle_id is not None:
        item["middlecategory"] = f"{bigcategory}-{middle_id}"
    else:
        print(f"[警告] middlecategory が未定義: {category}（bigcategory: {bigcategory}）")
        item["middlecategory"] = f"{bigcategory}-unknown"

    updated_data.append(item)

# 保存ファイル
with open(legacy_path("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3_inc_mid.json"), "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)
