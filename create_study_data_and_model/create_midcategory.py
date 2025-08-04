import json
import os

# 入力ファイル（bigcategory付き）
with open("/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_3layer_12a6a2.json", "r", encoding="utf-8") as f:
    service_data = json.load(f)

# middle定義ファイルをキャッシュする辞書（bigcategoryごとに読み込む）
middle_maps = {}

def load_middle_map(bigcategory):
    filename = f"/home/nakagawa/category_soft_clu_balanced_balanced_kmeans+_results_12_{bigcategory}_layer2_12a6a2ver.json"
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
with open("/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_3layer_12a6a2_a.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)
