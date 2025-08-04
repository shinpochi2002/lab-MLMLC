import json

# 入力データの例
with open('softfill_mashup_data_new_api_to_4category_inclu_nomash.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# カテゴリーごとにAPI名を整理する
category_dict = {}

for api_name, api_data in data.items():
    category_numbers = api_data[0]
    for category in category_numbers:
        if category not in category_dict:
            category_dict[category] = []
        # 重複を避けるために追加する前にチェック
        if api_name not in category_dict[category]:
            category_dict[category].append(api_name)

# 結果をJSONファイルに保存
with open("5categorized_apis_inc_nomash.json", "w", encoding="utf-8") as f:
    json.dump(category_dict, f, ensure_ascii=False, indent=4)

print("カテゴリごとのAPI名が保存されました: 4categorized_apis.json")

