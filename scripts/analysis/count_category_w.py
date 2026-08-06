import json
import csv

# JSONファイルからデータを読み込む
with open('/home/nakagawa/softfill_mashup_data_new_api/nocategory/softfill_mashup_data_new_api_to_category.json', 'r', encoding='utf-8') as f:
    api_data = json.load(f)

# カテゴリとAPIをラベルとして処理
all_categories = set()
labels = []

# api_dataの処理
for api_name, api_info in api_data.items():
    categories = api_info[1]  # カテゴリ情報を取得
    labels.append(categories)
    all_categories.update(categories)

# カテゴリーの出現回数をカウントする辞書を作成
category_count = {}

count_dic = {}
# カテゴリーのカウントを増加
for label in labels:
    for cate in label:
        if cate in category_count:
            category_count[cate] += 1
        else:
            category_count[cate] = 1

for co in category_count.values():
    if co in count_dic:
        count_dic[co] += 1
    else:
        count_dic[co] = 1

with open('category_count.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # ヘッダー行を書き込む
    writer.writerow(["category", "Count"])
    
    # データ行を書き込む
    for category, count in category_count.items():
        writer.writerow([category, count])

with open('count_category.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # ヘッダー行を書き込む
    writer.writerow(["categorycount", "Count"])
    
    # データ行を書き込む
    for apicount, count in count_dic.items():
        writer.writerow([apicount, count])        


print("カテゴリーのカウント結果をテキストファイルに書き込みました。")
