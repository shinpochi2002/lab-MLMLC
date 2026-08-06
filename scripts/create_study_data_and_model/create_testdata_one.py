import json
import random

# 元のJSONファイルを読み込む
with open('/home/nakagawa/studydata/layer3/layer3_12a6a2/softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_12a6a2.json', 'r') as file:
    data = json.load(file)

# 全てのデータを取得
all_data = data

# データのID一覧を取得 (ここではidフィールドが配列の3番目の位置にあることを前提)
all_ids = [api_info["name"] for api_info in all_data]

alldatacount=len(all_ids)
print(alldatacount)

# テストデータの件数を全体の25%に設定
test_size = int(alldatacount*0.20)
bigpa=500
print(test_size)

# ランダムに25%のIDを抽出
random_ids = random.sample(all_ids, test_size)  

# 抽出されたIDに対応するデータを取得
random_data = [api_info for api_info in all_data if api_info["name"] in random_ids]

# 抽出されたID以外のデータを取得
remaining_data = [api_info for api_info in all_data if api_info["name"] not in random_ids]

# 抽出されたデータを新しいJSONファイルに保存
with open('/home/nakagawa/studydata/layer3/layer3_12a6a2/softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_12a6a2_test_data.json', 'w') as file:
    json.dump(random_data, file, indent=4, ensure_ascii=False)

# 抽出されなかったデータを別のJSONファイルに保存
with open('/home/nakagawa/studydata/layer3/layer3_12a6a2/softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_12a6a2_study_data.json', 'w') as file:
    json.dump(remaining_data, file, indent=4, ensure_ascii=False)

print(f"{test_size}件のテストデータと残りのデータがそれぞれJSONファイルに保存されました。")