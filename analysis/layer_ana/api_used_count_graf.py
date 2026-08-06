import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


# ==================================
# API利用回数取得
# ==================================
def compute_api_usage_counts(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    api_counter = Counter()

    for mashup, value in data.items():
        for api in value[2]:
            api_counter[api] += 1

    return api_counter


# ==================================
# 分布計算
# ==================================
def compute_usage_distribution(api_counter):

    usage_distribution = defaultdict(int)

    for api, count in api_counter.items():
        usage_distribution[count] += 1

    total_api = len(api_counter)

    usage_counts = sorted(usage_distribution.keys())
    api_ratios = [(usage_distribution[u] / total_api) * 100 for u in usage_counts]

    return usage_counts, api_ratios


# ==================================
# 棒グラフ（xを詰める）
# ==================================
def plot_bar_distribution(usage_counts, api_ratios):

    x_labels = []
    y = []

    for u, r in zip(usage_counts, api_ratios):
        if r > 0:
            x_labels.append(u)
            y.append(r)

    x_pos = range(len(x_labels))

    plt.figure(figsize=(16,8))

    plt.bar(x_pos, y)

    plt.xlabel("API Usage Count")
    plt.ylabel("API Ratio (%)")

    plt.xticks(x_pos, x_labels, rotation=90)

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# ==================================
# scatter log-log（詰めない）
# ==================================
def plot_loglog_scatter(usage_counts, api_ratios):

    x = []
    y = []

    for u, r in zip(usage_counts, api_ratios):
        if r > 0:
            x.append(u)
            y.append(r)

    plt.figure(figsize=(16,8))

    plt.scatter(x, y, s=40)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("API Usage Count (log)",fontsize=24)
    plt.ylabel("API Ratio (%) (log)",fontsize=24)

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# ==================================
# 実行
# ==================================

api_counter = compute_api_usage_counts("mashup_data_new.json")

usage_counts, api_ratios = compute_usage_distribution(api_counter)

# 棒グラフ
plot_bar_distribution(usage_counts, api_ratios)

# log-log scatter
plot_loglog_scatter(usage_counts, api_ratios)