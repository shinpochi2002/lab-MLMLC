import json

# ファイルパスを指定
mashup_file_path = "mashup_data_new.json"
apilist_file_path = "hardapilist.json"
new_mashup_file_path = "hardfil_mashup_new_data.json"  # 新しいファイル名

# mashup_data_new.jsonを読み込む
with open(mashup_file_path, "r", encoding="utf-8") as mashup_file:
    mashup_data = json.load(mashup_file)

# apilist.jsonを読み込む
with open(apilist_file_path, "r", encoding="utf-8") as apilist_file:
    api_list = json.load(apilist_file)

# apilist.jsonに含まれるAPI名でmashup_data_new.jsonからデータを削除
for api_name in api_list:
    if api_name in mashup_data:
        del mashup_data[api_name]

# ID番号を詰める（1から順番に振り直す）
new_id = 1
for key in mashup_data:
    mashup_data[key][-1] = new_id  # ID番号は各エントリの最後にあることを想定
    new_id += 1

# 新しいファイル名で保存
with open(new_mashup_file_path, "w", encoding="utf-8") as new_mashup_file:
    json.dump(mashup_data, new_mashup_file, indent=4, ensure_ascii=False)

print(f"apilist.jsonに含まれるAPI名のデータを削除し、ID番号を詰めたデータを '{new_mashup_file_path}' に保存しました。")
