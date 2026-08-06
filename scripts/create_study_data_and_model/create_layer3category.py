import json
import os

# 入力ファイル（midcategory付き）
with open("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3_inc_mid.json", "r", encoding="utf-8") as f:
    service_data = json.load(f)

# middle定義ファイルをキャッシュする辞書（bigcategoryごとに読み込む）
middle_maps = {}

def load_middle_map(midcategory):
    new_mid = midcategory.replace("-", "_")
    filename = f"/home/nakagawa/lab-MLMLC/cluster/8_3_2_3/category_soft_clu_balanced_balanced_kmeans+_results_8_{new_mid}_layer4_8_3_2_3ver.json"
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
    midcategory = item.get("middlecategory")

    if midcategory is None:
        print(f"[警告] bigcategory が存在しない: {item}")
        item["middlecategory"] = "unknown"
        updated_data.append(item)
        continue

    if midcategory not in middle_maps:
        middle_maps[midcategory] = load_middle_map(midcategory)

    cat_to_mid = middle_maps[midcategory]
    layer3_id = cat_to_mid.get(category)

    if layer3_id is not None:
        item["layer3category"] = f"{midcategory}-{layer3_id}"
    else:
        print(f"[警告] middlecategory が未定義: {category}（middlecategory: {midcategory}）")
        item["middlecategory"] = f"{midcategory}-unknown"

    updated_data.append(item)

# 保存ファイル
with open("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3_inc_layer3.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)
