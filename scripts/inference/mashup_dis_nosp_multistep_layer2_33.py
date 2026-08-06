import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

M0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_0.json')

M1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_1.json')

M2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_2.json')

M3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_3.json')

M4API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_4.json')

M5API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_5.json')

M6API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_6.json')

M7API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_7.json')

M8API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_8.json')

M9API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_9.json')

M10API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_10.json')

M11API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_11.json')

M12API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_12.json')

M13API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_13.json')

M14API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_14.json')

M15API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_15.json')

M16API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_16.json')

M17API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_17.json')

M18API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_18.json')

M19API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_19.json')

M20API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_20.json')

M21API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_21.json')

M22API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_22.json')

M23API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_23.json')

M24API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_24.json')

M25API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_25.json')

M26API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_26.json')

M27API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_27.json')

M28API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_28.json')

M29API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_29.json')

M30API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_30.json')

M31API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_31.json')

M32API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_33_32.json')



EVALUATION_DATA_PATH = legacy_path('/home/nakagawa/collaboration_test_data.json')


BMODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_model_weights.pth")

M0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_0_model_weights.pth")

M1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_1_model_weights.pth")

M2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_2_model_weights.pth")

M3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_3_model_weights.pth")

M4MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_4_model_weights.pth")

M5MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_5_model_weights.pth")

M6MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_6_model_weights.pth")

M7MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_7_model_weights.pth")

M8MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_8_model_weights.pth")

M9MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_9_model_weights.pth")

M10MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_10_model_weights.pth")

M11MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_11_model_weights.pth")

M12MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_12_model_weights.pth")

M13MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_13_model_weights.pth")

M14MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_14_model_weights.pth")

M15MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_15_model_weights.pth")

M16MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_16_model_weights.pth")

M17MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_17_model_weights.pth")

M18MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_18_model_weights.pth")

M19MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_19_model_weights.pth")

M20MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_20_model_weights.pth")

M21MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_21_model_weights.pth")

M22MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_22_model_weights.pth")

M23MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_23_model_weights.pth")

M24MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_24_model_weights.pth")

M25MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_25_model_weights.pth")

M26MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_26_model_weights.pth")

M27MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_27_model_weights.pth")

M28MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_28_model_weights.pth")

M29MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_29_model_weights.pth")

M30MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_30_model_weights.pth")

M31MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_31_model_weights.pth")

M32MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_33_32_model_weights.pth")




# Load class list from 'apilist.json' (API names list)
with open(M0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M4API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M5API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M6API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M7API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M8API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M9API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M10API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M11API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M12API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid12_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M13API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid13_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M14API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid14_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M15API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid15_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M16API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid16_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M17API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid17_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M18API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid18_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M19API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid19_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M20API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid20_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M21API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid21_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M22API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid22_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M23API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid23_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
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



big_all_categories_list = [
    28, 25, 12, 26, 24, 23, 18, 31, 14, 0, 13, 20, 11,
    30, 6, 4, 8, 9, 3, 7, 19, 5, 15, 27, 21, 1, 2, 29,
    17, 10, 22, 32, 16
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

allsmall_category_to_index = {category: idx for idx, category in enumerate(allsmall_all_categories_list)}
allsmall_index_to_category = {idx: category for category, idx in allsmall_category_to_index.items()}


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
 
allsmall_number_of_labels = len(allsmall_all_categories_list)


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



model_name = legacy_path("/home/nakagawa/models/bert-base-uncased/")
tokenizer = BertTokenizer.from_pretrained(model_name)

bigmodel = load_model(BertForMultilabelClassification, model_name, big_number_of_labels, path=BMODEL_WEIGHTS_PATH)

m0model = load_model(BertForMultilabelClassification, model_name, m0_number_of_labels, path=M0MODEL_WEIGHTS_PATH)

m1model = load_model(BertForMultilabelClassification, model_name, m1_number_of_labels, path=M1MODEL_WEIGHTS_PATH)

m2model = load_model(BertForMultilabelClassification, model_name, m2_number_of_labels, path=M2MODEL_WEIGHTS_PATH)

m3model = load_model(BertForMultilabelClassification, model_name, m3_number_of_labels, path=M3MODEL_WEIGHTS_PATH)

m4model = load_model(BertForMultilabelClassification, model_name, m4_number_of_labels, path=M4MODEL_WEIGHTS_PATH)

m5model = load_model(BertForMultilabelClassification, model_name, m5_number_of_labels, path=M5MODEL_WEIGHTS_PATH)

m6model = load_model(BertForMultilabelClassification, model_name, m6_number_of_labels, path=M6MODEL_WEIGHTS_PATH)

m7model = load_model(BertForMultilabelClassification, model_name, m7_number_of_labels, path=M7MODEL_WEIGHTS_PATH)

m8model = load_model(BertForMultilabelClassification, model_name, m8_number_of_labels, path=M8MODEL_WEIGHTS_PATH)

m9model = load_model(BertForMultilabelClassification, model_name, m9_number_of_labels, path=M9MODEL_WEIGHTS_PATH)

m10model = load_model(BertForMultilabelClassification, model_name, m10_number_of_labels, path=M10MODEL_WEIGHTS_PATH)

m11model = load_model(BertForMultilabelClassification, model_name, m11_number_of_labels, path=M11MODEL_WEIGHTS_PATH)

m12model = load_model(BertForMultilabelClassification, model_name, m12_number_of_labels, path=M12MODEL_WEIGHTS_PATH)

m13model = load_model(BertForMultilabelClassification, model_name, m13_number_of_labels, path=M13MODEL_WEIGHTS_PATH)

m14model = load_model(BertForMultilabelClassification, model_name, m14_number_of_labels, path=M14MODEL_WEIGHTS_PATH)

m15model = load_model(BertForMultilabelClassification, model_name, m15_number_of_labels, path=M15MODEL_WEIGHTS_PATH)

m16model = load_model(BertForMultilabelClassification, model_name, m16_number_of_labels, path=M16MODEL_WEIGHTS_PATH)

m17model = load_model(BertForMultilabelClassification, model_name, m17_number_of_labels, path=M17MODEL_WEIGHTS_PATH)

m18model = load_model(BertForMultilabelClassification, model_name, m18_number_of_labels, path=M18MODEL_WEIGHTS_PATH)

m19model = load_model(BertForMultilabelClassification, model_name, m19_number_of_labels, path=M19MODEL_WEIGHTS_PATH)

m20model = load_model(BertForMultilabelClassification, model_name, m20_number_of_labels, path=M20MODEL_WEIGHTS_PATH)

m21model = load_model(BertForMultilabelClassification, model_name, m21_number_of_labels, path=M21MODEL_WEIGHTS_PATH)

m22model = load_model(BertForMultilabelClassification, model_name, m22_number_of_labels, path=M22MODEL_WEIGHTS_PATH)

m23model = load_model(BertForMultilabelClassification, model_name, m23_number_of_labels, path=M23MODEL_WEIGHTS_PATH)

m24model = load_model(BertForMultilabelClassification, model_name, m24_number_of_labels, path=M24MODEL_WEIGHTS_PATH)

m25model = load_model(BertForMultilabelClassification, model_name, m25_number_of_labels, path=M25MODEL_WEIGHTS_PATH)

m26model = load_model(BertForMultilabelClassification, model_name, m26_number_of_labels, path=M26MODEL_WEIGHTS_PATH)

m27model = load_model(BertForMultilabelClassification, model_name, m27_number_of_labels, path=M27MODEL_WEIGHTS_PATH)

m28model = load_model(BertForMultilabelClassification, model_name, m28_number_of_labels, path=M28MODEL_WEIGHTS_PATH)

m29model = load_model(BertForMultilabelClassification, model_name, m29_number_of_labels, path=M29MODEL_WEIGHTS_PATH)

m30model = load_model(BertForMultilabelClassification, model_name, m30_number_of_labels, path=M30MODEL_WEIGHTS_PATH)

m31model = load_model(BertForMultilabelClassification, model_name, m31_number_of_labels, path=M31MODEL_WEIGHTS_PATH)

m32model = load_model(BertForMultilabelClassification, model_name, m32_number_of_labels, path=M32MODEL_WEIGHTS_PATH)

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
                mid0results = run_model_and_get_results(
                text=text,
                model=m0model,
                tokenizer=tokenizer,
                index_to_category=m0_index_to_category,
                )
                recommended_indices = [category[0] for category in mid0results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 1:
                mid1results = run_model_and_get_results(
                text=text,
                model=m1model,
                tokenizer=tokenizer,
                index_to_category=m1_index_to_category,
                )
                recommended_indices = [category[0] for category in mid1results]
                allrecommended+=recommended_indices  
                
            elif nextrun[0] == 2:
                mid2results = run_model_and_get_results(
                text=text,
                model=m2model,
                tokenizer=tokenizer,
                index_to_category=m2_index_to_category,
                )
                recommended_indices = [category[0] for category in mid2results]
                allrecommended+=recommended_indices
            
            elif nextrun[0] == 3:
                mid3results = run_model_and_get_results(
                text=text,
                model=m3model,
                tokenizer=tokenizer,
                index_to_category=m3_index_to_category,
                )
                recommended_indices = [category[0] for category in mid3results]
                allrecommended+=recommended_indices
            
            elif nextrun[0] == 4:
                mid4results = run_model_and_get_results(
                text=text,
                model=m4model,
                tokenizer=tokenizer,
                index_to_category=m4_index_to_category,
                )
                recommended_indices = [category[0] for category in mid4results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 5:
                mid5results = run_model_and_get_results(
                text=text,
                model=m5model,
                tokenizer=tokenizer,
                index_to_category=m5_index_to_category,
                )
                recommended_indices = [category[0] for category in mid5results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 6:
                mid6results = run_model_and_get_results(
                text=text,
                model=m6model,
                tokenizer=tokenizer,
                index_to_category=m6_index_to_category,
                )
                recommended_indices = [category[0] for category in mid6results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 7:
                mid7results = run_model_and_get_results(
                text=text,
                model=m7model,
                tokenizer=tokenizer,
                index_to_category=m7_index_to_category,
                )
                recommended_indices = [category[0] for category in mid7results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 8:
                mid8results = run_model_and_get_results(
                text=text,
                model=m8model,
                tokenizer=tokenizer,
                index_to_category=m8_index_to_category,
                )
                recommended_indices = [category[0] for category in mid8results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 9:
                mid9results = run_model_and_get_results(
                text=text,
                model=m9model,
                tokenizer=tokenizer,
                index_to_category=m9_index_to_category,
                )
                recommended_indices = [category[0] for category in mid9results]
                allrecommended+=recommended_indices  

            elif nextrun[0] == 10:
                mid10results = run_model_and_get_results(
                text=text,
                model=m10model,
                tokenizer=tokenizer,
                index_to_category=m10_index_to_category,
                )
                recommended_indices = [category[0] for category in mid10results]
                allrecommended+=recommended_indices   

            elif nextrun[0] == 11:
                mid11results = run_model_and_get_results(
                text=text,
                model=m11model,
                tokenizer=tokenizer,
                index_to_category=m11_index_to_category,
                )
                recommended_indices = [category[0] for category in mid11results]
                allrecommended+=recommended_indices   

            elif nextrun[0] == 12:
                mid12results = run_model_and_get_results(
                text=text,
                model=m12model,
                tokenizer=tokenizer,
                index_to_category=m12_index_to_category,
                )
                recommended_indices = [category[0] for category in mid12results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 13:
                mid13results = run_model_and_get_results(
                text=text,
                model=m13model,
                tokenizer=tokenizer,
                index_to_category=m13_index_to_category,
                )
                recommended_indices = [category[0] for category in mid13results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 14:
                mid14results = run_model_and_get_results(
                text=text,
                model=m14model,
                tokenizer=tokenizer,
                index_to_category=m14_index_to_category,
                )
                recommended_indices = [category[0] for category in mid14results]
                allrecommended+=recommended_indices  

            elif nextrun[0] == 15:
                mid15results = run_model_and_get_results(
                text=text,
                model=m15model,
                tokenizer=tokenizer,
                index_to_category=m15_index_to_category,
                )
                recommended_indices = [category[0] for category in mid15results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 16:
                mid16results = run_model_and_get_results(
                text=text,
                model=m16model,
                tokenizer=tokenizer,
                index_to_category=m16_index_to_category,
                )
                recommended_indices = [category[0] for category in mid16results]
                allrecommended+=recommended_indices

            elif nextrun[0] == 17:
                mid17results = run_model_and_get_results(
                text=text,
                model=m17model,
                tokenizer=tokenizer,
                index_to_category=m17_index_to_category,
                )
                recommended_indices = [category[0] for category in mid17results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 18:
                mid18results = run_model_and_get_results(
                text=text,
                model=m18model,
                tokenizer=tokenizer,
                index_to_category=m18_index_to_category,
                )
                recommended_indices = [category[0] for category in mid18results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 19:
                mid19results = run_model_and_get_results(
                text=text,
                model=m19model,
                tokenizer=tokenizer,
                index_to_category=m19_index_to_category,
                )
                recommended_indices = [category[0] for category in mid19results]
                allrecommended+=recommended_indices  

            elif nextrun[0] == 20:
                mid20results = run_model_and_get_results(
                text=text,
                model=m20model,
                tokenizer=tokenizer,
                index_to_category=m20_index_to_category,
                )
                recommended_indices = [category[0] for category in mid20results]
                allrecommended+=recommended_indices   

            elif nextrun[0] == 21:
                mid21results = run_model_and_get_results(
                text=text,
                model=m21model,
                tokenizer=tokenizer,
                index_to_category=m21_index_to_category,
                )
                recommended_indices = [category[0] for category in mid21results]
                allrecommended+=recommended_indices   

            elif nextrun[0] == 22:
                mid22results = run_model_and_get_results(
                text=text,
                model=m22model,
                tokenizer=tokenizer,
                index_to_category=m22_index_to_category,
                )
                recommended_indices = [category[0] for category in mid22results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 23:
                mid23results = run_model_and_get_results(
                text=text,
                model=m23model,
                tokenizer=tokenizer,
                index_to_category=m23_index_to_category,
                )
                recommended_indices = [category[0] for category in mid23results]
                allrecommended+=recommended_indices 

            elif nextrun[0] == 24:
                mid24results = run_model_and_get_results(
                text=text,
                model=m24model,
                tokenizer=tokenizer,
                index_to_category=m24_index_to_category,
                )
                recommended_indices = [category[0] for category in mid24results]
                allrecommended += recommended_indices  

            elif nextrun[0] == 25:
                mid25results = run_model_and_get_results(
                text=text,
                model=m25model,
                tokenizer=tokenizer,
                index_to_category=m25_index_to_category,
                )
                recommended_indices = [category[0] for category in mid25results]
                allrecommended += recommended_indices

            elif nextrun[0] == 26:
                mid26results = run_model_and_get_results(
                text=text,
                model=m26model,
                tokenizer=tokenizer,
                index_to_category=m26_index_to_category,
                )
                recommended_indices = [category[0] for category in mid26results]
                allrecommended += recommended_indices

            elif nextrun[0] == 27:
                mid27results = run_model_and_get_results(
                text=text,
                model=m27model,
                tokenizer=tokenizer,
                index_to_category=m27_index_to_category,
                )
                recommended_indices = [category[0] for category in mid27results]
                allrecommended += recommended_indices  

            elif nextrun[0] == 28:
                mid28results = run_model_and_get_results(
                text=text,
                model=m28model,
                tokenizer=tokenizer,
                index_to_category=m28_index_to_category,
                )
                recommended_indices = [category[0] for category in mid28results]
                allrecommended += recommended_indices  

            elif nextrun[0] == 29:
                mid29results = run_model_and_get_results(
                text=text,
                model=m29model,
                tokenizer=tokenizer,
                index_to_category=m29_index_to_category,
                )
                recommended_indices = [category[0] for category in mid29results]
                allrecommended += recommended_indices  

            elif nextrun[0] == 30:
                mid30results = run_model_and_get_results(
                text=text,
                model=m30model,
                tokenizer=tokenizer,
                index_to_category=m30_index_to_category,
                )
                recommended_indices = [category[0] for category in mid30results]
                allrecommended += recommended_indices   

            elif nextrun[0] == 31:
                mid31results = run_model_and_get_results(
                text=text,
                model=m31model,
                tokenizer=tokenizer,
                index_to_category=m31_index_to_category,
                )
                recommended_indices = [category[0] for category in mid31results]
                allrecommended += recommended_indices   

            elif nextrun[0] == 32:
                mid32results = run_model_and_get_results(
                text=text,
                model=m32model,
                tokenizer=tokenizer,
                index_to_category=m32_index_to_category,
                )
                recommended_indices = [category[0] for category in mid32results]
                allrecommended += recommended_indices  
                                           
            
        print(allrecommended)
        print(evaluation_labels[idx])
        precision,recall,f1 = calculate_metrics(evaluation_labels[idx],allrecommended)
        metricslist.append((precision,recall,f1))
        #print(allrecommended_binary)
        #print(evaluate_label_binary)
    
    average_metrics(metricslist)


if __name__ == "__main__":
    main2()