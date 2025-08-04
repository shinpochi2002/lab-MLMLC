import json

# ファイルパスの設定
mashup_data_file = 'softfill_mashup_new_data.json'
api_category_file = 'api_name_category.json'

# mashup_data_new.json ファイルを読み込む
with open(mashup_data_file, 'r', encoding='utf-8') as f:
    mashup_data = json.load(f)

# api_name_category.json ファイルを読み込む
with open(api_category_file, 'r', encoding='utf-8') as f:
    api_name_to_category = json.load(f)

notinclude=[]
notincount=0
dangerlist=[]

# API名をカテゴリーに置き換える処理
for key, value in mashup_data.items():
    if isinstance(value, list):
        if isinstance(value[0], list):
            value.pop(0)
        for item in value:
            if isinstance(item, list):
                for i, api_name in enumerate(item):
                    if api_name in api_name_to_category:
                        # API名をカテゴリーに置き換える
                        item[i] = api_name_to_category[api_name]
                    else:
                        item[i] = "Attention!!!!"
                        notincount+=1
                        if api_name not in notinclude:
                            notinclude.append(api_name)
                            dangerlist.append(key)


# 置き換え後のデータを新しいファイルに保存
with open('softfill_mashup_data_new_api_to_category.json', 'w', encoding='utf-8') as f:
    json.dump(mashup_data, f, ensure_ascii=False, indent=4)

print("API名のカテゴリー置き換えが完了しました！")
print(notinclude)
print(len(notinclude))
print(dangerlist)
print(len(dangerlist))