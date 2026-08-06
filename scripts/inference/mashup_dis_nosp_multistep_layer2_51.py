import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl


M0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_0.json'
M1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_1.json'
M2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_2.json'
M3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_3.json'
M4API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_4.json'
M5API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_5.json'
M6API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_6.json'
M7API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_7.json'
M8API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_8.json'
M9API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_9.json'
M10API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_10.json'
M11API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_11.json'
M12API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_12.json'
M13API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_13.json'
M14API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_14.json'
M15API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_15.json'
M16API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_16.json'
M17API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_17.json'
M18API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_18.json'
M19API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_19.json'
M20API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_20.json'
M21API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_21.json'
M22API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_22.json'
M23API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_23.json'
M24API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_24.json'
M25API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_25.json'
M26API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_26.json'
M27API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_27.json'
M28API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_28.json'
M29API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_29.json'
M30API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_30.json'
M31API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_31.json'
M32API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_32.json'
M33API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_33.json'
M34API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_34.json'
M35API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_35.json'
M36API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_36.json'
M37API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_37.json'
M38API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_38.json'
M39API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_39.json'
M40API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_40.json'
M41API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_41.json'
M42API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_42.json'
M43API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_43.json'
M44API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_44.json'
M45API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_45.json'
M46API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_46.json'
M47API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_47.json'
M48API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_48.json'
M49API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_49.json'
M50API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_51_50.json'



with open(M0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_all_categories_list = json.load(f)

with open(M1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_all_categories_list = json.load(f)

with open(M2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_all_categories_list = json.load(f)

with open(M3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_all_categories_list = json.load(f)

with open(M4API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_all_categories_list = json.load(f)

with open(M5API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_all_categories_list = json.load(f)

with open(M6API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_all_categories_list = json.load(f)

with open(M7API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_all_categories_list = json.load(f)

with open(M8API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_all_categories_list = json.load(f)

with open(M9API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_all_categories_list = json.load(f)

with open(M10API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_all_categories_list = json.load(f)

with open(M11API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_all_categories_list = json.load(f)

with open(M12API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid12_all_categories_list = json.load(f)

with open(M13API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid13_all_categories_list = json.load(f)

with open(M14API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid14_all_categories_list = json.load(f)

with open(M15API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid15_all_categories_list = json.load(f)

with open(M16API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid16_all_categories_list = json.load(f)

with open(M17API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid17_all_categories_list = json.load(f)

with open(M18API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid18_all_categories_list = json.load(f)

with open(M19API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid19_all_categories_list = json.load(f)

with open(M20API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid20_all_categories_list = json.load(f)

with open(M21API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid21_all_categories_list = json.load(f)

with open(M22API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid22_all_categories_list = json.load(f)

with open(M23API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid23_all_categories_list = json.load(f)

with open(M24API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid24_all_categories_list = json.load(f)

with open(M25API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid25_all_categories_list = json.load(f)

with open(M26API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid26_all_categories_list = json.load(f)

with open(M27API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid27_all_categories_list = json.load(f)

with open(M28API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid28_all_categories_list = json.load(f)

with open(M29API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid29_all_categories_list = json.load(f)

with open(M30API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid30_all_categories_list = json.load(f)

with open(M31API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid31_all_categories_list = json.load(f)

with open(M32API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid32_all_categories_list = json.load(f)

with open(M33API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid33_all_categories_list = json.load(f)

with open(M34API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid34_all_categories_list = json.load(f)

with open(M35API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid35_all_categories_list = json.load(f)

with open(M36API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid36_all_categories_list = json.load(f)

with open(M37API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid37_all_categories_list = json.load(f)

with open(M38API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid38_all_categories_list = json.load(f)

with open(M39API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid39_all_categories_list = json.load(f)

with open(M40API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid40_all_categories_list = json.load(f)

with open(M41API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid41_all_categories_list = json.load(f)

with open(M42API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid42_all_categories_list = json.load(f)

with open(M43API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid43_all_categories_list = json.load(f)

with open(M44API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid44_all_categories_list = json.load(f)

with open(M45API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid45_all_categories_list = json.load(f)

with open(M46API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid46_all_categories_list = json.load(f)

with open(M47API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid47_all_categories_list = json.load(f)

with open(M48API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid48_all_categories_list = json.load(f)

with open(M49API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid49_all_categories_list = json.load(f)

with open(M50API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid50_all_categories_list = json.load(f)




big_all_categories_list = [
    28, 4, 47, 20, 5, 33, 36, 44, 2, 16,
    6, 18, 45, 30, 25, 22, 48, 32, 38, 43,
    3, 10, 49, 24, 15, 8, 19, 26, 23, 0,
    31, 34, 46, 9, 17, 14, 39, 40, 29, 35,
    13, 50, 41, 21, 42, 11, 1, 12, 7, 37,
    27
]



allsmall_all_categories_list = [
    "home-automation",
    "events",
    "location",
    "voice",
    "file-sharing",
    "blogging",
    "screenshots",
    "platform-as-a-service",
    "bookmarks",
    "qa",
    "blockchain",
    "merchants",
    "spam",
    "addresses",
    "fitness",
    "electronic-signature",
    "politics",
    "news-services",
    "database",
    "advertising",
    "weather",
    "movies",
    "cycling",
    "gambling",
    "support",
    "email",
    "streaming",
    "shipping",
    "analytics",
    "sentiment",
    "accounting",
    "spreadsheets",
    "developers",
    "medical",
    "messaging",
    "ecommerce",
    "mail",
    "data-as-a-service",
    "domains",
    "feeds",
    "bitcoin",
    "environment",
    "cloud",
    "video",
    "applications",
    "financial",
    "coupons",
    "esports",
    "random",
    "sports",
    "booking",
    "pdf",
    "printing",
    "security",
    "chat",
    "reporting",
    "cryptocurrency",
    "images",
    "forms",
    "web-site-management",
    "internet-things",
    "data",
    "calendars",
    "fax",
    "statistics",
    "geography",
    "media",
    "books",
    "barcodes",
    "wiki",
    "real-estate",
    "sales",
    "semantics",
    "hardware",
    "food",
    "other",
    "telephony",
    "auto",
    "storage",
    "purchasing",
    "healthcare",
    "url-shortener",
    "big-data",
    "air-travel",
    "seo",
    "photos",
    "tools",
    "science",
    "widgets",
    "api",
    "application-development",
    "social",
    "transportation",
    "ride-share",
    "identity",
    "hosting",
    "payments",
    "parsing",
    "recognition",
    "monitoring",
    "learning-management-systems",
    "music",
    "government",
    "medicine",
    "nutrition",
    "office",
    "content",
    "library",
    "stocks",
    "education",
    "search",
    "mapping",
    "covid-19",
    "business",
    "astronomy",
    "language",
    "enterprise",
    "standards",
    "wi-fi",
    "games",
    "intelligence",
    "framework",
    "entertainment",
    "feedback",
    "animals",
    "marketing",
    "machine-learning",
    "health",
    "bots",
    "design",
    "jobs",
    "natural-language-processing",
    "recommendations",
    "classification",
    "products",
    "reference",
    "documents",
    "authentication",
    "data-mining",
    "tagging",
    "biometrics",
    "collaboration",
    "translation",
    "project-management",
    "invoicing",
    "backend",
    "predictions",
    "personal-information-management",
    "travel",
    "text",
    "performance",
    "rewards"
]

# Create mappings for the categories
big_category_to_index = {category: idx for idx, category in enumerate(big_all_categories_list)}
big_index_to_category = {idx: category for category, idx in big_category_to_index.items()}

m0_category_to_index = {category: idx for idx, category in enumerate(mid0_all_categories_list)}
m0_index_to_category = {idx: category for category, idx in m0_category_to_index.items()}

m1_category_to_index = {category: idx for idx, category in enumerate(mid1_all_categories_list)}
m1_index_to_category = {idx: category for category, idx in m1_category_to_index.items()}

m2_category_to_index = {category: idx for idx, category in enumerate(mid2_all_categories_list)}
m2_index_to_category = {idx: category for category, idx in m2_category_to_index.items()}

m3_category_to_index = {category: idx for idx, category in enumerate(mid3_all_categories_list)}
m3_index_to_category = {idx: category for category, idx in m3_category_to_index.items()}

m4_category_to_index = {category: idx for idx, category in enumerate(mid4_all_categories_list)}
m4_index_to_category = {idx: category for category, idx in m4_category_to_index.items()}

m5_category_to_index = {category: idx for idx, category in enumerate(mid5_all_categories_list)}
m5_index_to_category = {idx: category for category, idx in m5_category_to_index.items()}

m6_category_to_index = {category: idx for idx, category in enumerate(mid6_all_categories_list)}
m6_index_to_category = {idx: category for category, idx in m6_category_to_index.items()}

m7_category_to_index = {category: idx for idx, category in enumerate(mid7_all_categories_list)}
m7_index_to_category = {idx: category for category, idx in m7_category_to_index.items()}

m8_category_to_index = {category: idx for idx, category in enumerate(mid8_all_categories_list)}
m8_index_to_category = {idx: category for category, idx in m8_category_to_index.items()}

m9_category_to_index = {category: idx for idx, category in enumerate(mid9_all_categories_list)}
m9_index_to_category = {idx: category for category, idx in m9_category_to_index.items()}

m10_category_to_index = {category: idx for idx, category in enumerate(mid10_all_categories_list)}
m10_index_to_category = {idx: category for category, idx in m10_category_to_index.items()}

m11_category_to_index = {category: idx for idx, category in enumerate(mid11_all_categories_list)}
m11_index_to_category = {idx: category for category, idx in m11_category_to_index.items()}

m12_category_to_index = {category: idx for idx, category in enumerate(mid12_all_categories_list)}
m12_index_to_category = {idx: category for category, idx in m12_category_to_index.items()}

m13_category_to_index = {category: idx for idx, category in enumerate(mid13_all_categories_list)}
m13_index_to_category = {idx: category for category, idx in m13_category_to_index.items()}

m14_category_to_index = {category: idx for idx, category in enumerate(mid14_all_categories_list)}
m14_index_to_category = {idx: category for category, idx in m14_category_to_index.items()}

m15_category_to_index = {category: idx for idx, category in enumerate(mid15_all_categories_list)}
m15_index_to_category = {idx: category for category, idx in m15_category_to_index.items()}

m16_category_to_index = {category: idx for idx, category in enumerate(mid16_all_categories_list)}
m16_index_to_category = {idx: category for category, idx in m16_category_to_index.items()}

m17_category_to_index = {category: idx for idx, category in enumerate(mid17_all_categories_list)}
m17_index_to_category = {idx: category for category, idx in m17_category_to_index.items()}

m18_category_to_index = {category: idx for idx, category in enumerate(mid18_all_categories_list)}
m18_index_to_category = {idx: category for category, idx in m18_category_to_index.items()}

m19_category_to_index = {category: idx for idx, category in enumerate(mid19_all_categories_list)}
m19_index_to_category = {idx: category for category, idx in m19_category_to_index.items()}

m20_category_to_index = {category: idx for idx, category in enumerate(mid20_all_categories_list)}
m20_index_to_category = {idx: category for category, idx in m20_category_to_index.items()}

m21_category_to_index = {category: idx for idx, category in enumerate(mid21_all_categories_list)}
m21_index_to_category = {idx: category for category, idx in m21_category_to_index.items()}

m22_category_to_index = {category: idx for idx, category in enumerate(mid22_all_categories_list)}
m22_index_to_category = {idx: category for category, idx in m22_category_to_index.items()}

m23_category_to_index = {category: idx for idx, category in enumerate(mid23_all_categories_list)}
m23_index_to_category = {idx: category for category, idx in m23_category_to_index.items()}

m24_category_to_index = {category: idx for idx, category in enumerate(mid24_all_categories_list)}
m24_index_to_category = {idx: category for category, idx in m24_category_to_index.items()}

m25_category_to_index = {category: idx for idx, category in enumerate(mid25_all_categories_list)}
m25_index_to_category = {idx: category for category, idx in m25_category_to_index.items()}

m26_category_to_index = {category: idx for idx, category in enumerate(mid26_all_categories_list)}
m26_index_to_category = {idx: category for category, idx in m26_category_to_index.items()}

m27_category_to_index = {category: idx for idx, category in enumerate(mid27_all_categories_list)}
m27_index_to_category = {idx: category for category, idx in m27_category_to_index.items()}

m28_category_to_index = {category: idx for idx, category in enumerate(mid28_all_categories_list)}
m28_index_to_category = {idx: category for category, idx in m28_category_to_index.items()}

m29_category_to_index = {category: idx for idx, category in enumerate(mid29_all_categories_list)}
m29_index_to_category = {idx: category for category, idx in m29_category_to_index.items()}

m30_category_to_index = {category: idx for idx, category in enumerate(mid30_all_categories_list)}
m30_index_to_category = {idx: category for category, idx in m30_category_to_index.items()}

m31_category_to_index = {category: idx for idx, category in enumerate(mid31_all_categories_list)}
m31_index_to_category = {idx: category for category, idx in m31_category_to_index.items()}

m32_category_to_index = {category: idx for idx, category in enumerate(mid32_all_categories_list)}
m32_index_to_category = {idx: category for category, idx in m32_category_to_index.items()}

m33_category_to_index = {category: idx for idx, category in enumerate(mid33_all_categories_list)}
m33_index_to_category = {idx: category for category, idx in m33_category_to_index.items()}

m34_category_to_index = {category: idx for idx, category in enumerate(mid34_all_categories_list)}
m34_index_to_category = {idx: category for category, idx in m34_category_to_index.items()}

m35_category_to_index = {category: idx for idx, category in enumerate(mid35_all_categories_list)}
m35_index_to_category = {idx: category for category, idx in m35_category_to_index.items()}

m36_category_to_index = {category: idx for idx, category in enumerate(mid36_all_categories_list)}
m36_index_to_category = {idx: category for category, idx in m36_category_to_index.items()}

m37_category_to_index = {category: idx for idx, category in enumerate(mid37_all_categories_list)}
m37_index_to_category = {idx: category for category, idx in m37_category_to_index.items()}

m38_category_to_index = {category: idx for idx, category in enumerate(mid38_all_categories_list)}
m38_index_to_category = {idx: category for category, idx in m38_category_to_index.items()}

m39_category_to_index = {category: idx for idx, category in enumerate(mid39_all_categories_list)}
m39_index_to_category = {idx: category for category, idx in m39_category_to_index.items()}

m40_category_to_index = {category: idx for idx, category in enumerate(mid40_all_categories_list)}
m40_index_to_category = {idx: category for category, idx in m40_category_to_index.items()}

m41_category_to_index = {category: idx for idx, category in enumerate(mid41_all_categories_list)}
m41_index_to_category = {idx: category for category, idx in m41_category_to_index.items()}

m42_category_to_index = {category: idx for idx, category in enumerate(mid42_all_categories_list)}
m42_index_to_category = {idx: category for category, idx in m42_category_to_index.items()}

m43_category_to_index = {category: idx for idx, category in enumerate(mid43_all_categories_list)}
m43_index_to_category = {idx: category for category, idx in m43_category_to_index.items()}

m44_category_to_index = {category: idx for idx, category in enumerate(mid44_all_categories_list)}
m44_index_to_category = {idx: category for category, idx in m44_category_to_index.items()}

m45_category_to_index = {category: idx for idx, category in enumerate(mid45_all_categories_list)}
m45_index_to_category = {idx: category for category, idx in m45_category_to_index.items()}

m46_category_to_index = {category: idx for idx, category in enumerate(mid46_all_categories_list)}
m46_index_to_category = {idx: category for category, idx in m46_category_to_index.items()}

m47_category_to_index = {category: idx for idx, category in enumerate(mid47_all_categories_list)}
m47_index_to_category = {idx: category for category, idx in m47_category_to_index.items()}

m48_category_to_index = {category: idx for idx, category in enumerate(mid48_all_categories_list)}
m48_index_to_category = {idx: category for category, idx in m48_category_to_index.items()}

m49_category_to_index = {category: idx for idx, category in enumerate(mid49_all_categories_list)}
m49_index_to_category = {idx: category for category, idx in m49_category_to_index.items()}

m50_category_to_index = {category: idx for idx, category in enumerate(mid50_all_categories_list)}
m50_index_to_category = {idx: category for category, idx in m50_category_to_index.items()}

allsmall_category_to_index = {category: idx for idx, category in enumerate(allsmall_all_categories_list)}
allsmall_index_to_category = {idx: category for category, idx in allsmall_category_to_index.items()}


BMODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_model_weights.pth"
M0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_0_model_weights.pth"
M1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_1_model_weights.pth"
M2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_2_model_weights.pth"
M3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_3_model_weights.pth"
M4MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_4_model_weights.pth"
M5MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_5_model_weights.pth"
M6MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_6_model_weights.pth"
M7MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_7_model_weights.pth"
M8MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_8_model_weights.pth"
M9MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_9_model_weights.pth"
M10MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_10_model_weights.pth"
M11MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_11_model_weights.pth"
M12MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_12_model_weights.pth"
M13MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_13_model_weights.pth"
M14MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_14_model_weights.pth"
M15MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_15_model_weights.pth"
M16MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_16_model_weights.pth"
M17MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_17_model_weights.pth"
M18MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_18_model_weights.pth"
M19MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_19_model_weights.pth"
M20MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_20_model_weights.pth"
M21MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_21_model_weights.pth"
M22MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_22_model_weights.pth"
M23MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_23_model_weights.pth"
M24MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_24_model_weights.pth"
M25MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_25_model_weights.pth"
M26MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_26_model_weights.pth"
M27MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_27_model_weights.pth"
M28MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_28_model_weights.pth"
M29MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_29_model_weights.pth"
M30MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_30_model_weights.pth"
M31MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_31_model_weights.pth"
M32MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_32_model_weights.pth"
M33MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_33_model_weights.pth"
M34MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_34_model_weights.pth"
M35MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_35_model_weights.pth"
M36MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_36_model_weights.pth"
M37MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_37_model_weights.pth"
M38MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_38_model_weights.pth"
M39MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_39_model_weights.pth"
M40MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_40_model_weights.pth"
M41MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_41_model_weights.pth"
M42MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_42_model_weights.pth"
M43MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_43_model_weights.pth"
M44MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_44_model_weights.pth"
M45MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_45_model_weights.pth"
M46MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_46_model_weights.pth"
M47MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_47_model_weights.pth"
M48MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_48_model_weights.pth"
M49MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_49_model_weights.pth"
M50MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_51_50_model_weights.pth"


print(m0_index_to_category)

# Total number of labels
big_number_of_labels = len(big_all_categories_list)

m0_number_of_labels = len(mid0_all_categories_list)
m1_number_of_labels = len(mid1_all_categories_list)
m2_number_of_labels = len(mid2_all_categories_list)
m3_number_of_labels = len(mid3_all_categories_list)
m4_number_of_labels = len(mid4_all_categories_list)
m5_number_of_labels = len(mid5_all_categories_list)
m6_number_of_labels = len(mid6_all_categories_list)
m7_number_of_labels = len(mid7_all_categories_list)
m8_number_of_labels = len(mid8_all_categories_list)
m9_number_of_labels = len(mid9_all_categories_list)
m10_number_of_labels = len(mid10_all_categories_list)
m11_number_of_labels = len(mid11_all_categories_list)
m12_number_of_labels = len(mid12_all_categories_list)
m13_number_of_labels = len(mid13_all_categories_list)
m14_number_of_labels = len(mid14_all_categories_list)
m15_number_of_labels = len(mid15_all_categories_list)
m16_number_of_labels = len(mid16_all_categories_list)
m17_number_of_labels = len(mid17_all_categories_list)
m18_number_of_labels = len(mid18_all_categories_list)
m19_number_of_labels = len(mid19_all_categories_list)
m20_number_of_labels = len(mid20_all_categories_list)
m21_number_of_labels = len(mid21_all_categories_list)
m22_number_of_labels = len(mid22_all_categories_list)
m23_number_of_labels = len(mid23_all_categories_list)
m24_number_of_labels = len(mid24_all_categories_list)
m25_number_of_labels = len(mid25_all_categories_list)
m26_number_of_labels = len(mid26_all_categories_list)
m27_number_of_labels = len(mid27_all_categories_list)
m28_number_of_labels = len(mid28_all_categories_list)
m29_number_of_labels = len(mid29_all_categories_list)
m30_number_of_labels = len(mid30_all_categories_list)
m31_number_of_labels = len(mid31_all_categories_list)
m32_number_of_labels = len(mid32_all_categories_list)
m33_number_of_labels = len(mid33_all_categories_list)
m34_number_of_labels = len(mid34_all_categories_list)
m35_number_of_labels = len(mid35_all_categories_list)
m36_number_of_labels = len(mid36_all_categories_list)
m37_number_of_labels = len(mid37_all_categories_list)
m38_number_of_labels = len(mid38_all_categories_list)
m39_number_of_labels = len(mid39_all_categories_list)
m40_number_of_labels = len(mid40_all_categories_list)
m41_number_of_labels = len(mid41_all_categories_list)
m42_number_of_labels = len(mid42_all_categories_list)
m43_number_of_labels = len(mid43_all_categories_list)
m44_number_of_labels = len(mid44_all_categories_list)
m45_number_of_labels = len(mid45_all_categories_list)
m46_number_of_labels = len(mid46_all_categories_list)
m47_number_of_labels = len(mid47_all_categories_list)
m48_number_of_labels = len(mid48_all_categories_list)
m49_number_of_labels = len(mid49_all_categories_list)
m50_number_of_labels = len(mid50_all_categories_list)
 
allsmall_number_of_labels = len(allsmall_all_categories_list)

EVALUATION_DATA_PATH = '/home/nakagawa/collaboration_test_data.json'

# Load evaluation data from 'mashup_data_new_test_data.json'
with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f:
    mashup_data = json.load(f)

# Extract evaluation texts and labels from mashup_data_new_test_data.json
evaluation_texts = []
evaluation_labels = []

# Extract categories and descriptions for evaluation
for api_info in mashup_data:
    bigcategories = api_info["bigclusters"]  # Extract categories as labels
    smallcategories = api_info["categories"]
    description = api_info["description"]  # Extract description as the text
    evaluation_texts.append(description)
    evaluation_labels.append(smallcategories)


# Define the BERT model for multilabel classification
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Pooled output for classification
        logits = self.classifier(pooled_output)
        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# Function to load the saved model
def load_model(model_class, model_name, num_labels, path=None):
    model = model_class(model_name, num_labels)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
    return model.to('cuda')

# Prediction function for a single text
def predict(text, model, tokenizer, index_to_category, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to('cuda')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    
    predicted_indices = np.where(probs > threshold)[0]
    recommended_categories = [(index_to_category[idx], probs[idx]) for idx in predicted_indices]
    
    return recommended_categories, probs


def run_model_and_get_results(text, model, tokenizer, index_to_category, threshold=0.5):
    """
    モデルによる予測を行い、結果を返す関数。

    Args:
        texts (list of str): 入力テキストのリスト
        model (torch.nn.Module): 学習済みモデル
        tokenizer (BertTokenizer): トークナイザー
        index_to_category (dict): カテゴリインデックスからカテゴリ名へのマッピング

    Returns:
        list of list of float: 各テキストに対応する予測確率のリスト
    """
    _, probs = predict(text, model, tokenizer, index_to_category)  # predict関数を利用
    category_predictions = {
        index_to_category[i] : float(prob) for i, prob in enumerate(probs)
    }
    # 閾値以上のカテゴリを抽出
    recommended_categories = [
        (category, prob) for category, prob in category_predictions.items() if prob >= threshold
    ]
    print(recommended_categories)

    return recommended_categories

def convert_labels_to_binary_vectors(labels, category_to_index, num_categories):
    """
    カテゴリ名リストをバイナリベクトルに変換する関数。

    Args:
        labels (list of list of str): 各データポイントのカテゴリ名リスト
        category_to_index (dict): カテゴリ名 -> インデックスのマッピング
        num_categories (int): カテゴリの総数（ベクトルの次元数）

    Returns:
        list of list of int: 各データポイントに対応するバイナリベクトルのリスト
    """
    vector = [0] * num_categories  # 全て0で初期化
    for label in labels:
        if label in category_to_index:  # マッピングが存在する場合のみ処理
            vector[category_to_index[label]] = 1
    return vector

def calculate_metrics(true_set, predicted_set):
    """
    適合率、再現率、Fスコアを計算する。
    """
    true_set = set(true_set)
    predicted_set = set(predicted_set)
    true_positive = len(true_set.intersection(predicted_set))
    precision = true_positive / len(predicted_set) if predicted_set else 0
    recall = true_positive / len(true_set) if true_set else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    print(f"Precision: {precision}, Recall: {recall}, F-Score: {f_score}")
    return precision, recall, f_score

def average_metrics(metrics_list):
    
    """
    渡されたリストから適合率、再現率、Fスコアの平均を計算する。
    """
    avg_precision = sum(m[0] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_recall = sum(m[1] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_f_score = sum(m[2] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    print(f"Precision: {avg_precision}, Recall: {avg_recall}, F-Score: {avg_f_score}")
    return avg_precision, avg_recall, avg_f_score



model_name = "/home/nakagawa/models/bert-base-uncased/"
tokenizer = BertTokenizer.from_pretrained(model_name)

bigmodel = load_model(BertForMultilabelClassification, model_name, big_number_of_labels, path=BMODEL_WEIGHTS_PATH)


# Main evaluation function (modified version of main2)
def main2():
    # 実行
    precisions=[]
    recalls=[]
    f1scores=[]
    metricslist = []
    for idx, text in enumerate(evaluation_texts):
        allrecommended = []
        print(f"Processing text {idx + 1}/{len(evaluation_texts)}")
        print(text[:100])
        
        bigresults = run_model_and_get_results(
            text=text,
            model=bigmodel,
            tokenizer=tokenizer,
            index_to_category=big_index_to_category,
        )

        print(bigresults)
        for nextrun in bigresults:
                if nextrun[0] == 0:
                    m0model = load_model(BertForMultilabelClassification, model_name, m0_number_of_labels, path=M0MODEL_WEIGHTS_PATH)
                    mid0results = run_model_and_get_results(
                        text=text,
                        model=m0model,
                        tokenizer=tokenizer,
                        index_to_category=m0_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid0results]
                    allrecommended += recommended_indices
                    del m0model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 1:
                    m1model = load_model(BertForMultilabelClassification, model_name, m1_number_of_labels, path=M1MODEL_WEIGHTS_PATH)
                    mid1results = run_model_and_get_results(
                        text=text,
                        model=m1model,
                        tokenizer=tokenizer,
                        index_to_category=m1_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid1results]
                    allrecommended += recommended_indices
                    del m1model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 2:
                    m2model = load_model(BertForMultilabelClassification, model_name, m2_number_of_labels, path=M2MODEL_WEIGHTS_PATH)
                    mid2results = run_model_and_get_results(
                        text=text,
                        model=m2model,
                        tokenizer=tokenizer,
                        index_to_category=m2_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid2results]
                    allrecommended += recommended_indices
                    del m2model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 3:
                    m3model = load_model(BertForMultilabelClassification, model_name, m3_number_of_labels, path=M3MODEL_WEIGHTS_PATH)
                    mid3results = run_model_and_get_results(
                        text=text,
                        model=m3model,
                        tokenizer=tokenizer,
                        index_to_category=m3_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid3results]
                    allrecommended += recommended_indices
                    del m3model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 4:
                    m4model = load_model(BertForMultilabelClassification, model_name, m4_number_of_labels, path=M4MODEL_WEIGHTS_PATH)
                    mid4results = run_model_and_get_results(
                        text=text,
                        model=m4model,
                        tokenizer=tokenizer,
                        index_to_category=m4_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid4results]
                    allrecommended += recommended_indices
                    del m4model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 5:
                    m5model = load_model(BertForMultilabelClassification, model_name, m5_number_of_labels, path=M5MODEL_WEIGHTS_PATH)
                    mid5results = run_model_and_get_results(
                        text=text,
                        model=m5model,
                        tokenizer=tokenizer,
                        index_to_category=m5_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid5results]
                    allrecommended += recommended_indices
                    del m5model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 6:
                    m6model = load_model(BertForMultilabelClassification, model_name, m6_number_of_labels, path=M6MODEL_WEIGHTS_PATH)
                    mid6results = run_model_and_get_results(
                        text=text,
                        model=m6model,
                        tokenizer=tokenizer,
                        index_to_category=m6_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid6results]
                    allrecommended += recommended_indices
                    del m6model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 7:
                    m7model = load_model(BertForMultilabelClassification, model_name, m7_number_of_labels, path=M7MODEL_WEIGHTS_PATH)
                    mid7results = run_model_and_get_results(
                        text=text,
                        model=m7model,
                        tokenizer=tokenizer,
                        index_to_category=m7_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid7results]
                    allrecommended += recommended_indices
                    del m7model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 8:
                    m8model = load_model(BertForMultilabelClassification, model_name, m8_number_of_labels, path=M8MODEL_WEIGHTS_PATH)
                    mid8results = run_model_and_get_results(
                        text=text,
                        model=m8model,
                        tokenizer=tokenizer,
                        index_to_category=m8_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid8results]
                    allrecommended += recommended_indices
                    del m8model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 9:
                    m9model = load_model(BertForMultilabelClassification, model_name, m9_number_of_labels, path=M9MODEL_WEIGHTS_PATH)
                    mid9results = run_model_and_get_results(
                        text=text,
                        model=m9model,
                        tokenizer=tokenizer,
                        index_to_category=m9_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid9results]
                    allrecommended += recommended_indices
                    del m9model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 10:
                    m10model = load_model(BertForMultilabelClassification, model_name, m10_number_of_labels, path=M10MODEL_WEIGHTS_PATH)
                    mid10results = run_model_and_get_results(
                        text=text,
                        model=m10model,
                        tokenizer=tokenizer,
                        index_to_category=m10_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid10results]
                    allrecommended += recommended_indices
                    del m10model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 11:
                    m11model = load_model(BertForMultilabelClassification, model_name, m11_number_of_labels, path=M11MODEL_WEIGHTS_PATH)
                    mid11results = run_model_and_get_results(
                        text=text,
                        model=m11model,
                        tokenizer=tokenizer,
                        index_to_category=m11_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid11results]
                    allrecommended += recommended_indices
                    del m11model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 12:
                    m12model = load_model(BertForMultilabelClassification, model_name, m12_number_of_labels, path=M12MODEL_WEIGHTS_PATH)
                    mid12results = run_model_and_get_results(
                        text=text,
                        model=m12model,
                        tokenizer=tokenizer,
                        index_to_category=m12_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid12results]
                    allrecommended += recommended_indices
                    del m12model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 13:
                    m13model = load_model(BertForMultilabelClassification, model_name, m13_number_of_labels, path=M13MODEL_WEIGHTS_PATH)
                    mid13results = run_model_and_get_results(
                        text=text,
                        model=m13model,
                        tokenizer=tokenizer,
                        index_to_category=m13_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid13results]
                    allrecommended += recommended_indices
                    del m13model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 14:
                    m14model = load_model(BertForMultilabelClassification, model_name, m14_number_of_labels, path=M14MODEL_WEIGHTS_PATH)
                    mid14results = run_model_and_get_results(
                        text=text,
                        model=m14model,
                        tokenizer=tokenizer,
                        index_to_category=m14_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid14results]
                    allrecommended += recommended_indices
                    del m14model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 15:
                    m15model = load_model(BertForMultilabelClassification, model_name, m15_number_of_labels, path=M15MODEL_WEIGHTS_PATH)
                    mid15results = run_model_and_get_results(
                        text=text,
                        model=m15model,
                        tokenizer=tokenizer,
                        index_to_category=m15_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid15results]
                    allrecommended += recommended_indices
                    del m15model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 16:
                    m16model = load_model(BertForMultilabelClassification, model_name, m16_number_of_labels, path=M16MODEL_WEIGHTS_PATH)
                    mid16results = run_model_and_get_results(
                        text=text,
                        model=m16model,
                        tokenizer=tokenizer,
                        index_to_category=m16_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid16results]
                    allrecommended += recommended_indices
                    del m16model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 17:
                    m17model = load_model(BertForMultilabelClassification, model_name, m17_number_of_labels, path=M17MODEL_WEIGHTS_PATH)
                    mid17results = run_model_and_get_results(
                        text=text,
                        model=m17model,
                        tokenizer=tokenizer,
                        index_to_category=m17_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid17results]
                    allrecommended += recommended_indices
                    del m17model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 18:
                    m18model = load_model(BertForMultilabelClassification, model_name, m18_number_of_labels, path=M18MODEL_WEIGHTS_PATH)
                    mid18results = run_model_and_get_results(
                        text=text,
                        model=m18model,
                        tokenizer=tokenizer,
                        index_to_category=m18_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid18results]
                    allrecommended += recommended_indices
                    del m18model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 19:
                    m19model = load_model(BertForMultilabelClassification, model_name, m19_number_of_labels, path=M19MODEL_WEIGHTS_PATH)
                    mid19results = run_model_and_get_results(
                        text=text,
                        model=m19model,
                        tokenizer=tokenizer,
                        index_to_category=m19_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid19results]
                    allrecommended += recommended_indices
                    del m19model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 20:
                    m20model = load_model(BertForMultilabelClassification, model_name, m20_number_of_labels, path=M20MODEL_WEIGHTS_PATH)
                    mid20results = run_model_and_get_results(
                        text=text,
                        model=m20model,
                        tokenizer=tokenizer,
                        index_to_category=m20_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid20results]
                    allrecommended += recommended_indices
                    del m20model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 21:
                    m21model = load_model(BertForMultilabelClassification, model_name, m21_number_of_labels, path=M21MODEL_WEIGHTS_PATH)
                    mid21results = run_model_and_get_results(
                        text=text,
                        model=m21model,
                        tokenizer=tokenizer,
                        index_to_category=m21_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid21results]
                    allrecommended += recommended_indices
                    del m21model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 22:
                    m22model = load_model(BertForMultilabelClassification, model_name, m22_number_of_labels, path=M22MODEL_WEIGHTS_PATH)
                    mid22results = run_model_and_get_results(
                        text=text,
                        model=m22model,
                        tokenizer=tokenizer,
                        index_to_category=m22_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid22results]
                    allrecommended += recommended_indices
                    del m22model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 23:
                    m23model = load_model(BertForMultilabelClassification, model_name, m23_number_of_labels, path=M23MODEL_WEIGHTS_PATH)
                    mid23results = run_model_and_get_results(
                        text=text,
                        model=m23model,
                        tokenizer=tokenizer,
                        index_to_category=m23_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid23results]
                    allrecommended += recommended_indices
                    del m23model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 24:
                    m24model = load_model(BertForMultilabelClassification, model_name, m24_number_of_labels, path=M24MODEL_WEIGHTS_PATH)
                    mid24results = run_model_and_get_results(
                        text=text,
                        model=m24model,
                        tokenizer=tokenizer,
                        index_to_category=m24_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid24results]
                    allrecommended += recommended_indices
                    del m24model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 25:
                    m25model = load_model(BertForMultilabelClassification, model_name, m25_number_of_labels, path=M25MODEL_WEIGHTS_PATH)
                    mid25results = run_model_and_get_results(
                        text=text,
                        model=m25model,
                        tokenizer=tokenizer,
                        index_to_category=m25_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid25results]
                    allrecommended += recommended_indices
                    del m25model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 26:
                    m26model = load_model(BertForMultilabelClassification, model_name, m26_number_of_labels, path=M26MODEL_WEIGHTS_PATH)
                    mid26results = run_model_and_get_results(
                        text=text,
                        model=m26model,
                        tokenizer=tokenizer,
                        index_to_category=m26_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid26results]
                    allrecommended += recommended_indices
                    del m26model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 27:
                    m27model = load_model(BertForMultilabelClassification, model_name, m27_number_of_labels, path=M27MODEL_WEIGHTS_PATH)
                    mid27results = run_model_and_get_results(
                        text=text,
                        model=m27model,
                        tokenizer=tokenizer,
                        index_to_category=m27_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid27results]
                    allrecommended += recommended_indices
                    del m27model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 28:
                    m28model = load_model(BertForMultilabelClassification, model_name, m28_number_of_labels, path=M28MODEL_WEIGHTS_PATH)
                    mid28results = run_model_and_get_results(
                        text=text,
                        model=m28model,
                        tokenizer=tokenizer,
                        index_to_category=m28_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid28results]
                    allrecommended += recommended_indices
                    del m28model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 29:
                    m29model = load_model(BertForMultilabelClassification, model_name, m29_number_of_labels, path=M29MODEL_WEIGHTS_PATH)
                    mid29results = run_model_and_get_results(
                        text=text,
                        model=m29model,
                        tokenizer=tokenizer,
                        index_to_category=m29_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid29results]
                    allrecommended += recommended_indices
                    del m29model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 30:
                    m30model = load_model(BertForMultilabelClassification, model_name, m30_number_of_labels, path=M30MODEL_WEIGHTS_PATH)
                    mid30results = run_model_and_get_results(
                        text=text,
                        model=m30model,
                        tokenizer=tokenizer,
                        index_to_category=m30_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid30results]
                    allrecommended += recommended_indices
                    del m30model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 31:
                    m31model = load_model(BertForMultilabelClassification, model_name, m31_number_of_labels, path=M31MODEL_WEIGHTS_PATH)
                    mid31results = run_model_and_get_results(
                        text=text,
                        model=m31model,
                        tokenizer=tokenizer,
                        index_to_category=m31_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid31results]
                    allrecommended += recommended_indices
                    del m31model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 32:
                    m32model = load_model(BertForMultilabelClassification, model_name, m32_number_of_labels, path=M32MODEL_WEIGHTS_PATH)
                    mid32results = run_model_and_get_results(
                        text=text,
                        model=m32model,
                        tokenizer=tokenizer,
                        index_to_category=m32_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid32results]
                    allrecommended += recommended_indices
                    del m32model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 33:
                    m33model = load_model(BertForMultilabelClassification, model_name, m33_number_of_labels, path=M33MODEL_WEIGHTS_PATH)
                    mid33results = run_model_and_get_results(
                        text=text,
                        model=m33model,
                        tokenizer=tokenizer,
                        index_to_category=m33_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid33results]
                    allrecommended += recommended_indices
                    del m33model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 34:
                    m34model = load_model(BertForMultilabelClassification, model_name, m34_number_of_labels, path=M34MODEL_WEIGHTS_PATH)
                    mid34results = run_model_and_get_results(
                        text=text,
                        model=m34model,
                        tokenizer=tokenizer,
                        index_to_category=m34_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid34results]
                    allrecommended += recommended_indices
                    del m34model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 35:
                    m35model = load_model(BertForMultilabelClassification, model_name, m35_number_of_labels, path=M35MODEL_WEIGHTS_PATH)
                    mid35results = run_model_and_get_results(
                        text=text,
                        model=m35model,
                        tokenizer=tokenizer,
                        index_to_category=m35_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid35results]
                    allrecommended += recommended_indices
                    del m35model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 36:
                    m36model = load_model(BertForMultilabelClassification, model_name, m36_number_of_labels, path=M36MODEL_WEIGHTS_PATH)
                    mid36results = run_model_and_get_results(
                        text=text,
                        model=m36model,
                        tokenizer=tokenizer,
                        index_to_category=m36_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid36results]
                    allrecommended += recommended_indices
                    del m36model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 37:
                    m37model = load_model(BertForMultilabelClassification, model_name, m37_number_of_labels, path=M37MODEL_WEIGHTS_PATH)
                    mid37results = run_model_and_get_results(
                        text=text,
                        model=m37model,
                        tokenizer=tokenizer,
                        index_to_category=m37_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid37results]
                    allrecommended += recommended_indices
                    del m37model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 38:
                    m38model = load_model(BertForMultilabelClassification, model_name, m38_number_of_labels, path=M38MODEL_WEIGHTS_PATH)
                    mid38results = run_model_and_get_results(
                        text=text,
                        model=m38model,
                        tokenizer=tokenizer,
                        index_to_category=m38_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid38results]
                    allrecommended += recommended_indices
                    del m38model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 39:
                    m39model = load_model(BertForMultilabelClassification, model_name, m39_number_of_labels, path=M39MODEL_WEIGHTS_PATH)
                    mid39results = run_model_and_get_results(
                        text=text,
                        model=m39model,
                        tokenizer=tokenizer,
                        index_to_category=m39_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid39results]
                    allrecommended += recommended_indices
                    del m39model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 40:
                    m40model = load_model(BertForMultilabelClassification, model_name, m40_number_of_labels, path=M40MODEL_WEIGHTS_PATH)
                    mid40results = run_model_and_get_results(
                        text=text,
                        model=m40model,
                        tokenizer=tokenizer,
                        index_to_category=m40_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid40results]
                    allrecommended += recommended_indices
                    del m40model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 41:
                    m41model = load_model(BertForMultilabelClassification, model_name, m41_number_of_labels, path=M41MODEL_WEIGHTS_PATH)
                    mid41results = run_model_and_get_results(
                        text=text,
                        model=m41model,
                        tokenizer=tokenizer,
                        index_to_category=m41_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid41results]
                    allrecommended += recommended_indices
                    del m41model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 42:
                    m42model = load_model(BertForMultilabelClassification, model_name, m42_number_of_labels, path=M42MODEL_WEIGHTS_PATH)
                    mid42results = run_model_and_get_results(
                        text=text,
                        model=m42model,
                        tokenizer=tokenizer,
                        index_to_category=m42_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid42results]
                    allrecommended += recommended_indices
                    del m42model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 43:
                    m43model = load_model(BertForMultilabelClassification, model_name, m43_number_of_labels, path=M43MODEL_WEIGHTS_PATH)
                    mid43results = run_model_and_get_results(
                        text=text,
                        model=m43model,
                        tokenizer=tokenizer,
                        index_to_category=m43_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid43results]
                    allrecommended += recommended_indices
                    del m43model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 44:
                    m44model = load_model(BertForMultilabelClassification, model_name, m44_number_of_labels, path=M44MODEL_WEIGHTS_PATH)
                    mid44results = run_model_and_get_results(
                        text=text,
                        model=m44model,
                        tokenizer=tokenizer,
                        index_to_category=m44_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid44results]
                    allrecommended += recommended_indices
                    del m44model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 45:
                    m45model = load_model(BertForMultilabelClassification, model_name, m45_number_of_labels, path=M45MODEL_WEIGHTS_PATH)
                    mid45results = run_model_and_get_results(
                        text=text,
                        model=m45model,
                        tokenizer=tokenizer,
                        index_to_category=m45_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid45results]
                    allrecommended += recommended_indices
                    del m45model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 46:
                    m46model = load_model(BertForMultilabelClassification, model_name, m46_number_of_labels, path=M46MODEL_WEIGHTS_PATH)
                    mid46results = run_model_and_get_results(
                        text=text,
                        model=m46model,
                        tokenizer=tokenizer,
                        index_to_category=m46_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid46results]
                    allrecommended += recommended_indices
                    del m46model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 47:
                    m47model = load_model(BertForMultilabelClassification, model_name, m47_number_of_labels, path=M47MODEL_WEIGHTS_PATH)
                    mid47results = run_model_and_get_results(
                        text=text,
                        model=m47model,
                        tokenizer=tokenizer,
                        index_to_category=m47_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid47results]
                    allrecommended += recommended_indices
                    del m47model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 48:
                    m48model = load_model(BertForMultilabelClassification, model_name, m48_number_of_labels, path=M48MODEL_WEIGHTS_PATH)
                    mid48results = run_model_and_get_results(
                        text=text,
                        model=m48model,
                        tokenizer=tokenizer,
                        index_to_category=m48_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid48results]
                    allrecommended += recommended_indices
                    del m48model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 49:
                    m49model = load_model(BertForMultilabelClassification, model_name, m49_number_of_labels, path=M49MODEL_WEIGHTS_PATH)
                    mid49results = run_model_and_get_results(
                        text=text,
                        model=m49model,
                        tokenizer=tokenizer,
                        index_to_category=m49_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid49results]
                    allrecommended += recommended_indices
                    del m49model
                    torch.cuda.empty_cache()


                elif nextrun[0] == 50:
                    m50model = load_model(BertForMultilabelClassification, model_name, m50_number_of_labels, path=M50MODEL_WEIGHTS_PATH)
                    mid50results = run_model_and_get_results(
                        text=text,
                        model=m50model,
                        tokenizer=tokenizer,
                        index_to_category=m50_index_to_category,
                    )
                    recommended_indices = [category[0] for category in mid50results]
                    allrecommended += recommended_indices
                    del m50model
                    torch.cuda.empty_cache()
                    
        print(allrecommended)
        print(evaluation_labels[idx])
        precision,recall,f1 = calculate_metrics(evaluation_labels[idx],allrecommended)
        metricslist.append((precision,recall,f1))
        #print(allrecommended_binary)
        #print(evaluate_label_binary)
    
    average_metrics(metricslist)


if __name__ == "__main__":
    main2()