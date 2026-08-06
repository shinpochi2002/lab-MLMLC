import json
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# JSONファイルを読み込む
with open(legacy_path('/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_study_data.json'), 'r', encoding='utf-8') as file:
    data = json.load(file)

# # 件数をカウント
# count = len(data)
# print(count)
keywords = [
    "documents",
    "nutrition",
    "location",
    "parsing",
    "health",
    "ride-share",
    "music",
    "internet-things",
    "weather",
    "streaming",
    "applications",
    "images",
    "qa",
    "payments",
    "wiki",
    "sales",
    "ecommerce",
    "bots",
    "database",
    "accounting",
    "blockchain",
    "data-as-a-service",
    "blogging",
    "feedback",
    "merchants",
    "data-mining",
    "application-development",
    "standards",
    "translation",
    "fitness",
    "calendars",
    "random",
    "reference",
    "mapping",
    "esports",
    "government",
    "storage",
    "file-sharing",
    "photos",
    "other",
    "learning-management-systems",
    "library",
    "video",
    "classification",
    "enterprise",
    "email",
    "seo",
    "support",
    "booking",
    "security",
    "electronic-signature",
    "geography",
    "auto",
    "wi-fi",
    "bitcoin",
    "platform-as-a-service",
    "home-automation",
    "authentication",
    "machine-learning",
    "personal-information-management",
    "data",
    "analytics",
    "purchasing",
    "advertising",
    "office",
    "invoicing",
    "rewards",
    "predictions",
    "media",
    "news-services",
    "identity",
    "collaboration",
    "messaging",
    "shipping",
    "performance",
    "food",
    "url-shortener",
    "fax",
    "gambling",
    "backend",
    "domains",
    "medicine",
    "biometrics",
    "books",
    "screenshots",
    "environment",
    "printing",
    "covid-19",
    "chat",
    "language",
    "travel",
    "real-estate",
    "movies",
    "sentiment",
    "search",
    "games",
    "social",
    "cryptocurrency",
    "project-management",
    "air-travel",
    "recommendations",
    "coupons",
    "transportation",
    "mail",
    "voice",
    "web-site-management",
    "recognition",
    "widgets",
    "sports",
    "business",
    "science",
    "intelligence",
    "financial",
    "products",
    "pdf",
    "cloud",
    "hosting",
    "tools",
    "barcodes",
    "design",
    "addresses",
    "events",
    "forms",
    "semantics",
    "cycling",
    "politics",
    "content",
    "monitoring",
    "animals",
    "entertainment",
    "api",
    "education",
    "telephony",
    "framework",
    "hardware",
    "bookmarks",
    "medical",
    "tagging",
    "feeds",
    "natural-language-processing",
    "spam",
    "text",
    "healthcare",
    "spreadsheets",
    "big-data",
    "developers",
    "marketing",
    "stocks",
    "statistics",
    "astronomy",
    "jobs",
    "reporting"
]

# 構成サービス（3番目の要素）を抽出してカウント
services = []
for keyword in keywords:
    count = 0
    for value in data:
        for category in value["categories"]:  # 3番目の要素を取得
            if category == keyword:
                count += 1
            else:
                pass
    print(keyword,":",count)


# # 構成サービスの数をカウントし、降順ソート
# service_count = Counter(services)
# sorted_services = service_count.most_common()  # [(サービス名, カウント), ...]

# # グラフ用のデータを抽出
# labels = [service for service, _ in sorted_services]
# counts = [count for _, count in sorted_services]

# # ヒストグラム作成
# plt.figure(figsize=(10, 6))
# plt.hist(counts, bins=2000, color="skyblue", edgecolor="black", alpha=0.7,log=True)  # ビン数を調整可能
# plt.title("Histogram of Service Counts(Log Scale)", fontsize=14)
# plt.xlabel("Number of Composed Services", fontsize=12)
# plt.ylabel("Number of Configuration Services", fontsize=12)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# # グラフをPNG形式で保存
# output_path = "service_count_histogram.png"
# plt.tight_layout()
# plt.savefig(output_path, format='png', dpi=300)
# plt.show()
# # 結果を出力
# print(f"全体のデータ件数: {count}")

#df_top_services = pd.DataFrame(sorted_services, columns=["Service Name", "Count"])

#print(df_top_services)
