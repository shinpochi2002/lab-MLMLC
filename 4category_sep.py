import json

# カテゴリーごとのAPI名リストを読み込み
with open('8cluster_to_apis.json', 'r', encoding='utf-8') as f:
    categorized_apis = json.load(f)

# 元のデータを読み込み
with open('/home/nakagawa/softfill_mashup_data_new_api_inclu_nomash/softfill_mashup_data_new_api_category_inclu_nomash.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# カテゴリーごとにデータを抽出して保存
for category, api_names in categorized_apis.items():
    category_data = {api_name: original_data[api_name] for api_name in api_names if api_name in original_data}
    
    # カテゴリーごとのファイルに保存
    with open(f"softfill_mashup_data_new_api_8category_inclu_nomash_category{category}ver.json", "w", encoding="utf-8") as f:
        json.dump(category_data, f, ensure_ascii=False, indent=4)

print("カテゴリーごとにデータが保存されました。")
