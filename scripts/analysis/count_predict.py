# ファイル名を指定
input_file = "/home/nakagawa/8category_small_multi_log.txt"

# ファイルからデータを読み込む
with open(input_file, "r", encoding="utf-8") as file:
    input_data = file.read()

# 入力データの処理
lines = input_data.strip().split("\n")
predicted_categories = []

for i in range(1, len(lines), 3):  # 予測カテゴリの行は各「Processing text」行の次の行
    predicted_list = eval(lines[i])  # カテゴリリストを取得して評価
    print(predicted_list)
    predicted_categories.append(predicted_list)

# 各リストの長さを計算し、平均を求める
average_category_count = sum(len(categories) for categories in predicted_categories) / len(predicted_categories)

# 結果を表示
print(f"予測したカテゴリの数の平均: {average_category_count:.2f}")
