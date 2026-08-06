import json
import os
from collections import defaultdict

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# 処理したい a, b の最大値
max_a = 12  # 例: 0〜4
max_b = 6  # 例: 0〜3

# パスのテンプレート
input_template = legacy_path("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/split/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data_4layer_8_3_2_3_{}_{}.json")
output_template = legacy_path("/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_8_3_2_3_{}_{}.json")

for a in range(max_a + 1):
    for b in range(max_b + 1):
        input_path = input_template.format(a, b)
        output_path = output_template.format(a, b)

        if not os.path.exists(input_path):
            print(f"{input_path} は存在しません。スキップします。")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merged = {}
        for item in data:
            name = item["name"]
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "categories": set(),
                    "bigcategories": set(),
                    "middlecategories": set(),
                    "layer3categories": set(),
                    "description": item["description"]
                }
            merged[name]["categories"].add(item["category"])
            merged[name]["bigcategories"].add(item["bigcategory"])
            if "middlecategory" in item and item["middlecategory"]:
                merged[name]["middlecategories"].add(item["middlecategory"])
            if "layer3category" in item and item["layer3category"]:
                merged[name]["layer3categories"].add(item["layer3category"])    
            

        final_data = []
        for entry in merged.values():
            output_entry = {
                "name": entry["name"],
                "categories": sorted(entry["categories"]),
                "bigcategories": sorted(entry["bigcategories"]),
                "description": entry["description"]
            }
            if entry["middlecategories"]:
                output_entry["middlecategories"] = sorted(entry["middlecategories"])
            if entry["layer3categories"]:
                output_entry["layer3categories"] = sorted(entry["layer3categories"])
            final_data.append(output_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"[{a}_{b}] {len(final_data)} 件のサービスに統合完了 → {output_path}")
