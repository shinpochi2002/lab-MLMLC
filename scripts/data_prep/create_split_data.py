import json

with open('/home/nakagawa/softfill_mashup_data_new_api_inclu_nomash/nocategory/softfill_mashup_data_new_api_category_inclu_nomash.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


def split_to_flat_list(data):
    result = []
    id = 1
    for name, value in data.items():
        categories = value[0]  # カテゴリーリスト
        description = value[1]   # 説明文
        for category in categories:
            result.append({
                "name": name,
                "category": category,
                "description": description,
                "id": id
            })
        id+=1
    return result

# 分割したデータを生成
split_data = split_to_flat_list(data)

# JSONとして保存する場合
with open("softfill_mashup_data_new_api_category_inclu_nomash_split.json", "w", encoding="utf-8") as f:
    json.dump(split_data, f, ensure_ascii=False, indent=4)

