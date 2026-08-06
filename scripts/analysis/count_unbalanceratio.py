import json
from collections import Counter
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

def gini_index(counter_obj):
    total = sum(counter_obj.values())
    print(total)
    if total == 0:
        return None
    
      # 最大のカテゴリの割合

    proportions = [count / total for count in counter_obj.values()]
    gini = 1 - sum(p ** 2 for p in proportions)
    max_percentage = max(proportions)
    
    # デバッグ用出力
    print("カテゴリの割合:", proportions)
    
    return gini,max_percentage

def calculate_z_score(counter_obj):
    """
    カテゴリ数をカウントした Counter オブジェクトを受け取り、
    全カテゴリの平均割合とZスコアを計算する。
    """
    total = sum(counter_obj.values())  # 全データの件数
    if total == 0:
        return None, None  # データがない場合
    
    percentages = np.array([count / total for count in counter_obj.values()])  # 各カテゴリの割合
    mean_percentage = np.mean(percentages)  # 全カテゴリの平均割合
    std_percentage = np.std(percentages)  # 標準偏差
    
    # Zスコアの計算（標準偏差が0ならNaNになるので回避）
    z_score = (max(percentages) - mean_percentage) / std_percentage if std_percentage > 0 else 0
    
    return mean_percentage, z_score,std_percentage

def manual_variance(counter_obj):
    """
    カテゴリのカウント情報から分散を手動計算
    """
    total = sum(counter_obj.values())
    if total == 0:
        return None  # データがない場合
    
    percentages = [count / total for count in counter_obj.values()]  # 各カテゴリの割合
    mean_percentage = sum(percentages) / len(percentages)  # 平均
    variance = sum((p - mean_percentage) ** 2 for p in percentages) / len(percentages)  # 分散
    
    return variance

def mean_absolute_deviation(counter_obj):
    """
    平均絶対偏差 (MAD) を計算し、カテゴリの不均衡度を測る。
    """
    total = sum(counter_obj.values())  # 全データの件数
    num_categories = len(counter_obj)  # カテゴリ数
    if total == 0 or num_categories == 0:
        return None  # データがない場合
    
    # 各カテゴリの実際の割合
    P = np.array([count / total for count in counter_obj.values()])
    
    # 理想的な均等分布（すべてのカテゴリが同じ割合）
    Q = np.full(len(P), 1 / num_categories)

    # 平均絶対偏差（MAD）を計算
    mad = np.mean(np.abs(P - Q))
    
    return mad

def calculate_z_score_difference(counter_obj):
    """
    カテゴリ数をカウントした Counter オブジェクトを受け取り、
    最大カテゴリと最小カテゴリの Zスコアを計算し、その差を求める。
    """
    total = sum(counter_obj.values())  # 全データ数
    if total == 0:
        return None, None, None  # データがない場合
    
    # 各カテゴリの割合を計算
    percentages = np.array([count / total for count in counter_obj.values()])
    
    # 平均と標準偏差を計算
    mean_percentage = np.mean(percentages)
    std_percentage = np.std(percentages)
    
    # 最大カテゴリと最小カテゴリの割合
    max_percentage = np.max(percentages)
    min_percentage = np.min(percentages)

    # Zスコアの計算（標準偏差が0なら0にする）
    z_max = (max_percentage - mean_percentage) / std_percentage if std_percentage > 0 else 0
    z_min = (mean_percentage - min_percentage) / std_percentage if std_percentage > 0 else 0

    # Zスコアの差
    z_diff = z_max - z_min

    return z_max, z_min, z_diff



# JSON ファイルを読み込む
with open(legacy_path("/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_study_data.json"), "r", encoding="utf-8") as f:
    data = json.load(f)  # data は「リスト」だと想定

# カテゴリごとの出現数をカウントする
category_counts = Counter()

for item in data:
    # 1つの要素に複数のカテゴリが含まれている可能性を想定
    for cat in item["categories"]:
        category_counts[cat] += 1

# ジニ係数を計算
gini,max_percentage = gini_index(category_counts)
mean_percentage, z_score , std_percentage = calculate_z_score(category_counts)
variance_manual = manual_variance(category_counts)
mad_value = mean_absolute_deviation(category_counts)
# 指標を計算
z_max, z_min, z_diff = calculate_z_score_difference(category_counts)

print("カテゴリの出現数:", category_counts)
# print("最大カテゴリの割合:", max_percentage)
# print("全カテゴリの平均割合:", mean_percentage)
# print("Zスコア (不均衡度):", z_score)
# print("標準偏差",std_percentage)
# print("ジニ係数:", gini)
# print("手動計算した分散:", variance_manual)
# print(f"平均絶対偏差 (MAD): {mad_value:.4f}")
print("最大カテゴリのZスコア:", z_max)
print("最小カテゴリのZスコア:", z_min)
print("Zスコアの差 (不均衡度):", z_diff)

