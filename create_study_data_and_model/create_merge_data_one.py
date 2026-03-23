import json
from collections import defaultdict

# 入力ファイル（big + middle 混在データ）
with open("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3_inc_layer3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# name ごとに統合
merged = {}
for item in data:
    name = item["name"]
    if name not in merged:
        merged[name] = {
            "name": name,
            "categories": set(),
            "bigcategories": set(),
            "middlecategories": set(),  # 空でも作っておく
            "layer3categories": set(),
            "description": item["description"]
        }
    merged[name]["categories"].add(item["category"])
    merged[name]["bigcategories"].add(item["bigcategory"])
    if "middlecategory" in item and item["middlecategory"]:
        merged[name]["middlecategories"].add(item["middlecategory"])
    if "layer3category" in item and item["layer3category"]:
        merged[name]["layer3categories"].add(item["layer3category"])    
            

# セット → ソート済みリストに変換して整形
final_data = []
for entry in merged.values():
    output_entry = {
        "name": entry["name"],
        "categories": sorted(entry["categories"]),
        "bigcategories": sorted(entry["bigcategories"]),
        "description": entry["description"]
    }
    if entry["middlecategories"]:  # 空でなければ追加
        output_entry["middlecategories"] = sorted(entry["middlecategories"])
    if entry["layer3categories"]:
        output_entry["layer3categories"] = sorted(entry["layer3categories"])
    final_data.append(output_entry)

# 出力ファイル
with open("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_all_study_data_4layer_8_3_2_3.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"{len(final_data)} 件のサービスに統合完了（middlecategory 有無対応）")