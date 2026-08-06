import json
from collections import Counter
import numpy as np

# ジニ係数を計算する関数
def gini_coefficient(values):
    """
    ジニ係数を計算する関数
    :param values: リストまたは配列 (非負の数値)
    :return: ジニ係数 (0から1の範囲)
    """
    if len(values) == 0:
        return 0

    sorted_values = sorted(values)  # 値を昇順にソート
    n = len(sorted_values)         # 要素数
    cumulative_sum = np.cumsum(sorted_values)  # 累積和
    total_sum = np.sum(sorted_values)          # 合計

    # ジニ係数の計算
    gini = 1 - 2 * np.sum((cumulative_sum - sorted_values) / total_sum) / n
    return gini

# JSONファイルを読み込み
with open('/home/nakagawa/softfill_mashup_data_new_api_category_4category_all_split.json', 'r', encoding='utf-8') as f:
    api_data = json.load(f)

# カテゴリのリストを収集
categories = [api_info["cluster"] for api_info in api_data]

# カテゴリの出現回数をカウント
category_counts = Counter(categories)

# カウント結果をリスト形式に変換
values = list(category_counts.values())

# ジニ係数を計算
gini_value = gini_coefficient(values)

# 結果を出力
print("カテゴリごとの出現回数:")
for category, count in category_counts.items():
    print(f"{category}: {count}")

print(f"\nジニ係数: {gini_value:.4f}")
