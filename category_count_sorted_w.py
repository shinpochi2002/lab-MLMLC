import json

# JSONファイルからデータを読み込む
with open('api_data_new.json', 'r', encoding='utf-8') as f:
    api_data = json.load(f)

# カテゴリとAPIをラベルとして処理
all_categories = set()
labels = []

# api_dataの処理
for api_name, api_info in api_data.items():
    categories = api_info[0][0]  # カテゴリ情報を取得
    labels.append(categories)
    all_categories.update(categories)

# カテゴリーの出現回数をカウントする辞書を作成
category_count = {}

# カテゴリーのカウントを増加
for label in labels:
    if label in category_count:
        category_count[label] += 1
    else:
        category_count[label] = 1

# カウント数を降順に並べ替える
sorted_category_count = sorted(category_count.items(), key=lambda x: x[1], reverse=True)

# カウント結果をテキストファイルに書き込む
with open('category_count_sorted.txt', 'w', encoding='utf-8') as f:
    f.write("Category Count (Descending Order):\n")
    for category, count in sorted_category_count:
        f.write(f"{category}: {count}\n")

print("カテゴリーのカウント結果を降順に並べてテキストファイルに書き込みました。")
