from collections import defaultdict
import json


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

with open(legacy_path('/home/nakagawa/softfill_mashup_data_new_api_category_8category_mid7_7-3_split_new.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)


# 同じ名前のアイテムをまとめる
def merge_data(data):
    merged = defaultdict(lambda: {"categories": set(), "description": "", "id": 0, "bigclusters": set(), "middleclusters": set()})

    for item in data:
        name = item["name"]
        merged[name]["categories"].add(item["category"])
        merged[name]["description"] = item["description"]  # 上書きで最新のものを使用
        merged[name]["id"] = item["id"]  # 上書きで最新のものを使用
        merged[name]["bigclusters"].add(item["bigcluster"])
        merged[name]["middleclusters"].add(item["middlecluster"])

    # 結果を整形
    result = []
    for name, attributes in merged.items():
        result.append({
            "name": name,
            "categories": list(attributes["categories"]),
            "description": attributes["description"],
            "id": attributes["id"],
            "bigclusters": list(attributes["bigclusters"]),
            "middleclusters": list(attributes["middleclusters"])
        })

    return result


# 実行
merged_result = merge_data(data)


with open(legacy_path('/home/nakagawa/softfill_mashup_data_new_api_category_8category_mid7_7-3_new.json'), 'w', encoding='utf-8') as f:
    json.dump(merged_result, f, indent=4, ensure_ascii=False)