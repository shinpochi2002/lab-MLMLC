import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl


M0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid0_new.json'
M0_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid0_0_new.json'
M0_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid0_1_new.json'
M0_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid0_2_new.json'
M0_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid0_3_new.json'

M1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid1_new.json'
M1_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid1_0_new.json'
M1_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid1_1_new.json'
M1_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid1_2_new.json'
M1_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid1_3_new.json'

M2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid2_new.json'
M2_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid2_0_new.json'
M2_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid2_1_new.json'
M2_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid2_2_new.json'
M2_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid2_3_new.json'

M3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid3_new.json'
M3_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid3_0_new.json'
M3_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid3_1_new.json'
M3_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid3_2_new.json'
M3_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid3_3_new.json'

M4API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid4_new.json'
M4_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid4_0_new.json'
M4_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid4_1_new.json'
M4_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid4_2_new.json'
M4_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid4_3_new.json'

M5API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid5_new.json'
M5_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid5_0_new.json'
M5_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid5_1_new.json'
M5_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid5_2_new.json'
M5_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid5_3_new.json'

M6API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid6_new.json'
M6_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid6_0_new.json'
M6_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid6_1_new.json'
M6_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid6_2_new.json'
M6_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid6_3_new.json'

M7API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid7_new.json'
M7_0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid7_0_new.json'
M7_1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid7_1_new.json'
M7_2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid7_2_new.json'
M7_3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_mid7_3_new.json'

EVALUATION_DATA_PATH = '/home/nakagawa/collaboration_test_data.json'

BMODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_all_big_new_model_weights_test.pth"
M0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid0_new_model_weights_test.pth"
M0_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid0-0_new_model_weights_test.pth"
M0_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid0-1_new_model_weights_test.pth"
M0_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid0-2_new_model_weights_test.pth"
M0_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid0-3_new_model_weights_test.pth"

M1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid1_new_model_weights_test.pth"
M1_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid1-0_new_model_weights_test.pth"
M1_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid1-1_new_model_weights_test.pth"
M1_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid1-2_new_model_weights_test.pth"
M1_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid1-3_new_model_weights_test.pth"

M2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid2_new_model_weights_test.pth"
M2_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid2-0_new_model_weights_test.pth"
M2_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid2-1_new_model_weights_test.pth"
M2_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid2-2_new_model_weights_test.pth"
M2_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid2-3_new_model_weights_test.pth"

M3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid3_new_model_weights_test.pth"
M3_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid3-0_new_model_weights_test.pth"
M3_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid3-1_new_model_weights_test.pth"
M3_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid3-2_new_model_weights_test.pth"
M3_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid3-3_new_model_weights_test.pth"

M4MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid4_new_model_weights_test.pth"
M4_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid4-0_new_model_weights_test.pth"
M4_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid4-1_new_model_weights_test.pth"
M4_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid4-2_new_model_weights_test.pth"
M4_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid4-3_new_model_weights_test.pth"

M5MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid5_new_model_weights_test.pth"
M5_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid5-0_new_model_weights_test.pth"
M5_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid5-1_new_model_weights_test.pth"
M5_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid5-2_new_model_weights_test.pth"
M5_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid5-3_new_model_weights_test.pth"

M6MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid6_new_model_weights_test.pth"
M6_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid6-0_new_model_weights_test.pth"
M6_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid6-1_new_model_weights_test.pth"
M6_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid6-2_new_model_weights_test.pth"
M6_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid6-3_new_model_weights_test.pth"

M7MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid7_new_model_weights_test.pth"
M7_0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid7-0_new_model_weights_test.pth"
M7_1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid7-1_new_model_weights_test.pth"
M7_2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid7-2_new_model_weights_test.pth"
M7_3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_8category_mid7-3_new_model_weights_test.pth"


# Load class list from 'apilist.json' (API names list)
with open(M0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M0_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M0_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M0_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M0_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names


# Load class list from 'apilist.json' (API names list)
with open(M1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M1_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M1_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M1_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M1_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M2_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M2_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M2_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M2_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M3_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M3_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M3_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M3_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M4API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M4_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M4_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M4_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M4_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid4_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M5API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M5_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M5_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M5_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M5_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid5_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M6API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M6_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M6_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M6_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M6_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid6_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M7API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M7_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M7_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M7_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M7_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid7_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names


big_all_categories_list = [0,1,2,3,4,5,6,7]
allmid_all_categories_list = ["0-0","0-1","0-2","0-3",
                              "1-0","1-1","1-2","1-3",
                              "2-0","2-1","2-2","2-3",
                              "3-0","3-1","3-2","3-3",
                              "4-0","4-1","4-2","4-3",
                              "5-0","5-1","5-2","5-3",
                              "6-0","6-1","6-2","6-3",
                              "7-0","7-1","7-2","7-3"]

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

m0_0_category_to_index = {category: idx for idx, category in enumerate(mid0_0_all_categories_list)}
m0_0_index_to_category = {idx: category for category, idx in m0_0_category_to_index.items()}

m0_1_category_to_index = {category: idx for idx, category in enumerate(mid0_1_all_categories_list)}
m0_1_index_to_category = {idx: category for category, idx in m0_1_category_to_index.items()}

m0_2_category_to_index = {category: idx for idx, category in enumerate(mid0_2_all_categories_list)}
m0_2_index_to_category = {idx: category for category, idx in m0_2_category_to_index.items()}

m0_3_category_to_index = {category: idx for idx, category in enumerate(mid0_3_all_categories_list)}
m0_3_index_to_category = {idx: category for category, idx in m0_3_category_to_index.items()}



m1_category_to_index = {category: idx for idx, category in enumerate(mid1_all_categories_list)}
m1_index_to_category = {idx: category for category, idx in m1_category_to_index.items()}

m1_0_category_to_index = {category: idx for idx, category in enumerate(mid1_0_all_categories_list)}
m1_0_index_to_category = {idx: category for category, idx in m1_0_category_to_index.items()}

m1_1_category_to_index = {category: idx for idx, category in enumerate(mid1_1_all_categories_list)}
m1_1_index_to_category = {idx: category for category, idx in m1_1_category_to_index.items()}

m1_2_category_to_index = {category: idx for idx, category in enumerate(mid1_2_all_categories_list)}
m1_2_index_to_category = {idx: category for category, idx in m1_2_category_to_index.items()}

m1_3_category_to_index = {category: idx for idx, category in enumerate(mid1_3_all_categories_list)}
m1_3_index_to_category = {idx: category for category, idx in m1_3_category_to_index.items()}



m2_category_to_index = {category: idx for idx, category in enumerate(mid2_all_categories_list)}
m2_index_to_category = {idx: category for category, idx in m2_category_to_index.items()}

m2_0_category_to_index = {category: idx for idx, category in enumerate(mid2_0_all_categories_list)}
m2_0_index_to_category = {idx: category for category, idx in m2_0_category_to_index.items()}

m2_1_category_to_index = {category: idx for idx, category in enumerate(mid2_1_all_categories_list)}
m2_1_index_to_category = {idx: category for category, idx in m2_1_category_to_index.items()}

m2_2_category_to_index = {category: idx for idx, category in enumerate(mid2_2_all_categories_list)}
m2_2_index_to_category = {idx: category for category, idx in m2_2_category_to_index.items()}

m2_3_category_to_index = {category: idx for idx, category in enumerate(mid2_3_all_categories_list)}
m2_3_index_to_category = {idx: category for category, idx in m2_3_category_to_index.items()}



m3_category_to_index = {category: idx for idx, category in enumerate(mid3_all_categories_list)}
m3_index_to_category = {idx: category for category, idx in m3_category_to_index.items()}

m3_0_category_to_index = {category: idx for idx, category in enumerate(mid3_0_all_categories_list)}
m3_0_index_to_category = {idx: category for category, idx in m3_0_category_to_index.items()}

m3_1_category_to_index = {category: idx for idx, category in enumerate(mid3_1_all_categories_list)}
m3_1_index_to_category = {idx: category for category, idx in m3_1_category_to_index.items()}

m3_2_category_to_index = {category: idx for idx, category in enumerate(mid3_2_all_categories_list)}
m3_2_index_to_category = {idx: category for category, idx in m3_2_category_to_index.items()}

m3_3_category_to_index = {category: idx for idx, category in enumerate(mid3_3_all_categories_list)}
m3_3_index_to_category = {idx: category for category, idx in m3_3_category_to_index.items()}



m4_category_to_index = {category: idx for idx, category in enumerate(mid4_all_categories_list)}
m4_index_to_category = {idx: category for category, idx in m4_category_to_index.items()}

m4_0_category_to_index = {category: idx for idx, category in enumerate(mid4_0_all_categories_list)}
m4_0_index_to_category = {idx: category for category, idx in m4_0_category_to_index.items()}

m4_1_category_to_index = {category: idx for idx, category in enumerate(mid4_1_all_categories_list)}
m4_1_index_to_category = {idx: category for category, idx in m4_1_category_to_index.items()}

m4_2_category_to_index = {category: idx for idx, category in enumerate(mid4_2_all_categories_list)}
m4_2_index_to_category = {idx: category for category, idx in m4_2_category_to_index.items()}

m4_3_category_to_index = {category: idx for idx, category in enumerate(mid4_3_all_categories_list)}
m4_3_index_to_category = {idx: category for category, idx in m4_3_category_to_index.items()}



m5_category_to_index = {category: idx for idx, category in enumerate(mid5_all_categories_list)}
m5_index_to_category = {idx: category for category, idx in m5_category_to_index.items()}

m5_0_category_to_index = {category: idx for idx, category in enumerate(mid5_0_all_categories_list)}
m5_0_index_to_category = {idx: category for category, idx in m5_0_category_to_index.items()}

m5_1_category_to_index = {category: idx for idx, category in enumerate(mid5_1_all_categories_list)}
m5_1_index_to_category = {idx: category for category, idx in m5_1_category_to_index.items()}

m5_2_category_to_index = {category: idx for idx, category in enumerate(mid5_2_all_categories_list)}
m5_2_index_to_category = {idx: category for category, idx in m5_2_category_to_index.items()}

m5_3_category_to_index = {category: idx for idx, category in enumerate(mid5_3_all_categories_list)}
m5_3_index_to_category = {idx: category for category, idx in m5_3_category_to_index.items()}



m6_category_to_index = {category: idx for idx, category in enumerate(mid6_all_categories_list)}
m6_index_to_category = {idx: category for category, idx in m6_category_to_index.items()}

m6_0_category_to_index = {category: idx for idx, category in enumerate(mid6_0_all_categories_list)}
m6_0_index_to_category = {idx: category for category, idx in m6_0_category_to_index.items()}

m6_1_category_to_index = {category: idx for idx, category in enumerate(mid6_1_all_categories_list)}
m6_1_index_to_category = {idx: category for category, idx in m6_1_category_to_index.items()}

m6_2_category_to_index = {category: idx for idx, category in enumerate(mid6_2_all_categories_list)}
m6_2_index_to_category = {idx: category for category, idx in m6_2_category_to_index.items()}

m6_3_category_to_index = {category: idx for idx, category in enumerate(mid6_3_all_categories_list)}
m6_3_index_to_category = {idx: category for category, idx in m6_3_category_to_index.items()}



m7_category_to_index = {category: idx for idx, category in enumerate(mid7_all_categories_list)}
m7_index_to_category = {idx: category for category, idx in m7_category_to_index.items()}

m7_0_category_to_index = {category: idx for idx, category in enumerate(mid7_0_all_categories_list)}
m7_0_index_to_category = {idx: category for category, idx in m7_0_category_to_index.items()}

m7_1_category_to_index = {category: idx for idx, category in enumerate(mid7_1_all_categories_list)}
m7_1_index_to_category = {idx: category for category, idx in m7_1_category_to_index.items()}

m7_2_category_to_index = {category: idx for idx, category in enumerate(mid7_2_all_categories_list)}
m7_2_index_to_category = {idx: category for category, idx in m7_2_category_to_index.items()}

m7_3_category_to_index = {category: idx for idx, category in enumerate(mid7_3_all_categories_list)}
m7_3_index_to_category = {idx: category for category, idx in m7_3_category_to_index.items()}

allmid_category_to_index = {category: idx for idx, category in enumerate(allmid_all_categories_list)}
allmid_index_to_category = {idx: category for category, idx in allmid_category_to_index.items()}

allsmall_category_to_index = {category: idx for idx, category in enumerate(allsmall_all_categories_list)}
allsmall_index_to_category = {idx: category for category, idx in allsmall_category_to_index.items()}


print(allmid_category_to_index)
print(m0_index_to_category)
print(m0_0_index_to_category)

# Total number of labels
big_number_of_labels = len(big_all_categories_list)

m0_number_of_labels = len(mid0_all_categories_list) 
m0_0_number_of_labels = len(mid0_0_all_categories_list)
m0_1_number_of_labels = len(mid0_1_all_categories_list)
m0_2_number_of_labels = len(mid0_2_all_categories_list)
m0_3_number_of_labels = len(mid0_3_all_categories_list)

m1_number_of_labels = len(mid1_all_categories_list)
m1_0_number_of_labels = len(mid1_0_all_categories_list)
m1_1_number_of_labels = len(mid1_1_all_categories_list)
m1_2_number_of_labels = len(mid1_2_all_categories_list)
m1_3_number_of_labels = len(mid1_3_all_categories_list) 

m2_number_of_labels = len(mid2_all_categories_list)
m2_0_number_of_labels = len(mid2_0_all_categories_list)
m2_1_number_of_labels = len(mid2_1_all_categories_list)
m2_2_number_of_labels = len(mid2_2_all_categories_list)
m2_3_number_of_labels = len(mid2_3_all_categories_list) 

m3_number_of_labels = len(mid3_all_categories_list)
m3_0_number_of_labels = len(mid3_0_all_categories_list)
m3_1_number_of_labels = len(mid3_1_all_categories_list)
m3_2_number_of_labels = len(mid3_2_all_categories_list)
m3_3_number_of_labels = len(mid3_3_all_categories_list)

m4_number_of_labels = len(mid4_all_categories_list)
m4_0_number_of_labels = len(mid4_0_all_categories_list)
m4_1_number_of_labels = len(mid4_1_all_categories_list)
m4_2_number_of_labels = len(mid4_2_all_categories_list)
m4_3_number_of_labels = len(mid4_3_all_categories_list)

m5_number_of_labels = len(mid5_all_categories_list)
m5_0_number_of_labels = len(mid5_0_all_categories_list)
m5_1_number_of_labels = len(mid5_1_all_categories_list)
m5_2_number_of_labels = len(mid5_2_all_categories_list)
m5_3_number_of_labels = len(mid5_3_all_categories_list)

m6_number_of_labels = len(mid6_all_categories_list)
m6_0_number_of_labels = len(mid6_0_all_categories_list)
m6_1_number_of_labels = len(mid6_1_all_categories_list)
m6_2_number_of_labels = len(mid6_2_all_categories_list)
m6_3_number_of_labels = len(mid6_3_all_categories_list)

m7_number_of_labels = len(mid7_all_categories_list)
m7_0_number_of_labels = len(mid7_0_all_categories_list)
m7_1_number_of_labels = len(mid7_1_all_categories_list)
m7_2_number_of_labels = len(mid7_2_all_categories_list)
m7_3_number_of_labels = len(mid7_3_all_categories_list)

allmid_number_of_labels = len(allmid_all_categories_list) 
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
    middlecategories = api_info["middleclusters"]
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
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
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

#bigmodel = load_model(BertForMultilabelClassification, model_name, big_number_of_labels, path=BMODEL_WEIGHTS_PATH)
#m0model = load_model(BertForMultilabelClassification, model_name, m0_number_of_labels, path=M0MODEL_WEIGHTS_PATH)
# m0_0model = load_model(BertForMultilabelClassification, model_name, m0_0_number_of_labels, path=M0_0MODEL_WEIGHTS_PATH)
# m0_1model = load_model(BertForMultilabelClassification, model_name, m0_1_number_of_labels, path=M0_1MODEL_WEIGHTS_PATH)
# m0_2model = load_model(BertForMultilabelClassification, model_name, m0_2_number_of_labels, path=M0_2MODEL_WEIGHTS_PATH)
# m0_3model = load_model(BertForMultilabelClassification, model_name, m0_3_number_of_labels, path=M0_3MODEL_WEIGHTS_PATH)

#m1model = load_model(BertForMultilabelClassification, model_name, m1_number_of_labels, path=M1MODEL_WEIGHTS_PATH)
# m1_0model = load_model(BertForMultilabelClassification, model_name, m1_0_number_of_labels, path=M1_0MODEL_WEIGHTS_PATH)
# m1_1model = load_model(BertForMultilabelClassification, model_name, m1_1_number_of_labels, path=M1_1MODEL_WEIGHTS_PATH)
# m1_2model = load_model(BertForMultilabelClassification, model_name, m1_2_number_of_labels, path=M1_2MODEL_WEIGHTS_PATH)
# m1_3model = load_model(BertForMultilabelClassification, model_name, m1_3_number_of_labels, path=M1_3MODEL_WEIGHTS_PATH)

#m2model = load_model(BertForMultilabelClassification, model_name, m2_number_of_labels, path=M2MODEL_WEIGHTS_PATH)
# m2_0model = load_model(BertForMultilabelClassification, model_name, m2_0_number_of_labels, path=M2_0MODEL_WEIGHTS_PATH)
# m2_1model = load_model(BertForMultilabelClassification, model_name, m2_1_number_of_labels, path=M2_1MODEL_WEIGHTS_PATH)
# m2_2model = load_model(BertForMultilabelClassification, model_name, m2_2_number_of_labels, path=M2_2MODEL_WEIGHTS_PATH)
# m2_3model = load_model(BertForMultilabelClassification, model_name, m2_3_number_of_labels, path=M2_3MODEL_WEIGHTS_PATH)

#m3model = load_model(BertForMultilabelClassification, model_name, m3_number_of_labels, path=M3MODEL_WEIGHTS_PATH)
# m3_0model = load_model(BertForMultilabelClassification, model_name, m3_0_number_of_labels, path=M3_0MODEL_WEIGHTS_PATH)
# m3_1model = load_model(BertForMultilabelClassification, model_name, m3_1_number_of_labels, path=M3_1MODEL_WEIGHTS_PATH)
# m3_2model = load_model(BertForMultilabelClassification, model_name, m3_2_number_of_labels, path=M3_2MODEL_WEIGHTS_PATH)
# m3_3model = load_model(BertForMultilabelClassification, model_name, m3_3_number_of_labels, path=M3_3MODEL_WEIGHTS_PATH)

#m4model = load_model(BertForMultilabelClassification, model_name, m4_number_of_labels, path=M4MODEL_WEIGHTS_PATH)
# m4_0model = load_model(BertForMultilabelClassification, model_name, m4_0_number_of_labels, path=M4_0MODEL_WEIGHTS_PATH)
# m4_1model = load_model(BertForMultilabelClassification, model_name, m4_1_number_of_labels, path=M4_1MODEL_WEIGHTS_PATH)
# m4_2model = load_model(BertForMultilabelClassification, model_name, m4_2_number_of_labels, path=M4_2MODEL_WEIGHTS_PATH)
# m4_3model = load_model(BertForMultilabelClassification, model_name, m4_3_number_of_labels, path=M4_3MODEL_WEIGHTS_PATH)

#m5model = load_model(BertForMultilabelClassification, model_name, m5_number_of_labels, path=M5MODEL_WEIGHTS_PATH)
# m5_0model = load_model(BertForMultilabelClassification, model_name, m5_0_number_of_labels, path=M5_0MODEL_WEIGHTS_PATH)
# m5_1model = load_model(BertForMultilabelClassification, model_name, m5_1_number_of_labels, path=M5_1MODEL_WEIGHTS_PATH)
# m5_2model = load_model(BertForMultilabelClassification, model_name, m5_2_number_of_labels, path=M5_2MODEL_WEIGHTS_PATH)
# m5_3model = load_model(BertForMultilabelClassification, model_name, m5_3_number_of_labels, path=M5_3MODEL_WEIGHTS_PATH)

#m6model = load_model(BertForMultilabelClassification, model_name, m6_number_of_labels, path=M6MODEL_WEIGHTS_PATH)
# m6_0model = load_model(BertForMultilabelClassification, model_name, m6_0_number_of_labels, path=M6_0MODEL_WEIGHTS_PATH)
# m6_1model = load_model(BertForMultilabelClassification, model_name, m6_1_number_of_labels, path=M6_1MODEL_WEIGHTS_PATH)
# m6_2model = load_model(BertForMultilabelClassification, model_name, m6_2_number_of_labels, path=M6_2MODEL_WEIGHTS_PATH)
# m6_3model = load_model(BertForMultilabelClassification, model_name, m6_3_number_of_labels, path=M6_3MODEL_WEIGHTS_PATH)

#m7model = load_model(BertForMultilabelClassification, model_name, m7_number_of_labels, path=M7MODEL_WEIGHTS_PATH)
# m7_0model = load_model(BertForMultilabelClassification, model_name, m7_0_number_of_labels, path=M7_0MODEL_WEIGHTS_PATH)
# m7_1model = load_model(BertForMultilabelClassification, model_name, m7_1_number_of_labels, path=M7_1MODEL_WEIGHTS_PATH)
# m7_2model = load_model(BertForMultilabelClassification, model_name, m7_2_number_of_labels, path=M7_2MODEL_WEIGHTS_PATH)
# m7_3model = load_model(BertForMultilabelClassification, model_name, m7_3_number_of_labels, path=M7_3MODEL_WEIGHTS_PATH)

# Main evaluation function (modified version of main2)
def main2():
    # 実行
    metricslist = []
    for idx, text in enumerate(evaluation_texts):
        allrecommended = []
        print(f"Processing text {idx + 1}/{len(evaluation_texts)}")
        print(text[:100])
        m0_0model = load_model(BertForMultilabelClassification, model_name, m0_0_number_of_labels, path=M0_0MODEL_WEIGHTS_PATH)
        m0_1model = load_model(BertForMultilabelClassification, model_name, m0_1_number_of_labels, path=M0_1MODEL_WEIGHTS_PATH)
        m0_2model = load_model(BertForMultilabelClassification, model_name, m0_2_number_of_labels, path=M0_2MODEL_WEIGHTS_PATH)
        m0_3model = load_model(BertForMultilabelClassification, model_name, m0_3_number_of_labels, path=M0_3MODEL_WEIGHTS_PATH)
    
        mid0_0results = run_model_and_get_results(
        text=text,
        model=m0_0model,
        tokenizer=tokenizer,
        index_to_category=m0_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid0_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid0_1results = run_model_and_get_results(
        text=text,
        model=m0_1model,
        tokenizer=tokenizer,
        index_to_category=m0_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid0_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid0_2results = run_model_and_get_results(
        text=text,
        model=m0_2model,
        tokenizer=tokenizer,
        index_to_category=m0_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid0_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid0_3results = run_model_and_get_results(
        text=text,
        model=m0_3model,
        tokenizer=tokenizer,
        index_to_category=m0_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid0_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m0_0model
        del m0_1model
        del m0_2model
        del m0_3model
        torch.cuda.empty_cache()
            
        m1_0model = load_model(BertForMultilabelClassification, model_name, m1_0_number_of_labels, path=M1_0MODEL_WEIGHTS_PATH)
        m1_1model = load_model(BertForMultilabelClassification, model_name, m1_1_number_of_labels, path=M1_1MODEL_WEIGHTS_PATH)
        m1_2model = load_model(BertForMultilabelClassification, model_name, m1_2_number_of_labels, path=M1_2MODEL_WEIGHTS_PATH)
        m1_3model = load_model(BertForMultilabelClassification, model_name, m1_3_number_of_labels, path=M1_3MODEL_WEIGHTS_PATH)
    
        mid1_0results = run_model_and_get_results(
        text=text,
        model=m1_0model,
        tokenizer=tokenizer,
        index_to_category=m1_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid1_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices
            
        mid1_1results = run_model_and_get_results(
        text=text,
        model=m1_1model,
        tokenizer=tokenizer,
        index_to_category=m1_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid1_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices
           
        mid1_2results = run_model_and_get_results(
        text=text,
        model=m1_2model,
        tokenizer=tokenizer,
        index_to_category=m1_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid1_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid1_3results = run_model_and_get_results(
        text=text,
        model=m1_3model,
        tokenizer=tokenizer,
        index_to_category=m1_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid1_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m1_0model
        del m1_1model
        del m1_2model
        del m1_3model
        torch.cuda.empty_cache()
            
        m2_0model = load_model(BertForMultilabelClassification, model_name, m2_0_number_of_labels, path=M2_0MODEL_WEIGHTS_PATH)
        m2_1model = load_model(BertForMultilabelClassification, model_name, m2_1_number_of_labels, path=M2_1MODEL_WEIGHTS_PATH)
        m2_2model = load_model(BertForMultilabelClassification, model_name, m2_2_number_of_labels, path=M2_2MODEL_WEIGHTS_PATH)
        m2_3model = load_model(BertForMultilabelClassification, model_name, m2_3_number_of_labels, path=M2_3MODEL_WEIGHTS_PATH)
        
        mid2_0results = run_model_and_get_results(
        text=text,
        model=m2_0model,
        tokenizer=tokenizer,
        index_to_category=m2_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid2_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid2_1results = run_model_and_get_results(
        text=text,
        model=m2_1model,
        tokenizer=tokenizer,
        index_to_category=m2_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid2_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid2_2results = run_model_and_get_results(
        text=text,
        model=m2_2model,
        tokenizer=tokenizer,
        index_to_category=m2_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid2_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid2_3results = run_model_and_get_results(
        text=text,
        model=m2_3model,
        tokenizer=tokenizer,
        index_to_category=m2_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid2_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m2_0model
        del m2_1model
        del m2_2model
        del m2_3model
        torch.cuda.empty_cache()
            
        m3_0model = load_model(BertForMultilabelClassification, model_name, m3_0_number_of_labels, path=M3_0MODEL_WEIGHTS_PATH)
        m3_1model = load_model(BertForMultilabelClassification, model_name, m3_1_number_of_labels, path=M3_1MODEL_WEIGHTS_PATH)
        m3_2model = load_model(BertForMultilabelClassification, model_name, m3_2_number_of_labels, path=M3_2MODEL_WEIGHTS_PATH)
        m3_3model = load_model(BertForMultilabelClassification, model_name, m3_3_number_of_labels, path=M3_3MODEL_WEIGHTS_PATH)
        
        mid3_0results = run_model_and_get_results(
        text=text,
        model=m3_0model,
        tokenizer=tokenizer,
        index_to_category=m3_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid3_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid3_1results = run_model_and_get_results(
        text=text,
        model=m3_1model,
        tokenizer=tokenizer,
        index_to_category=m3_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid3_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid3_2results = run_model_and_get_results(
        text=text,
        model=m3_2model,
        tokenizer=tokenizer,
        index_to_category=m3_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid3_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid3_3results = run_model_and_get_results(
        text=text,
        model=m3_3model,
        tokenizer=tokenizer,
        index_to_category=m3_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid3_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m3_0model
        del m3_1model
        del m3_2model
        del m3_3model
        torch.cuda.empty_cache()
                    
            
        m4_0model = load_model(BertForMultilabelClassification, model_name, m4_0_number_of_labels, path=M4_0MODEL_WEIGHTS_PATH)
        m4_1model = load_model(BertForMultilabelClassification, model_name, m4_1_number_of_labels, path=M4_1MODEL_WEIGHTS_PATH)
        m4_2model = load_model(BertForMultilabelClassification, model_name, m4_2_number_of_labels, path=M4_2MODEL_WEIGHTS_PATH)
        m4_3model = load_model(BertForMultilabelClassification, model_name, m4_3_number_of_labels, path=M4_3MODEL_WEIGHTS_PATH)
        
                
        mid4_0results = run_model_and_get_results(
        text=text,
        model=m4_0model,
        tokenizer=tokenizer,
        index_to_category=m4_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid4_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

                 
        mid4_1results = run_model_and_get_results(
        text=text,
        model=m4_1model,
        tokenizer=tokenizer,
        index_to_category=m4_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid4_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

                    
        mid4_2results = run_model_and_get_results(
        text=text,
        model=m4_2model,
        tokenizer=tokenizer,
        index_to_category=m4_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid4_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

                    
        mid4_3results = run_model_and_get_results(
        text=text,
        model=m4_3model,
        tokenizer=tokenizer,
        index_to_category=m4_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid4_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m4_0model
        del m4_1model
        del m4_2model
        del m4_3model
        torch.cuda.empty_cache()

        m5_0model = load_model(BertForMultilabelClassification, model_name, m5_0_number_of_labels, path=M5_0MODEL_WEIGHTS_PATH)
        m5_1model = load_model(BertForMultilabelClassification, model_name, m5_1_number_of_labels, path=M5_1MODEL_WEIGHTS_PATH)
        m5_2model = load_model(BertForMultilabelClassification, model_name, m5_2_number_of_labels, path=M5_2MODEL_WEIGHTS_PATH)
        m5_3model = load_model(BertForMultilabelClassification, model_name, m5_3_number_of_labels, path=M5_3MODEL_WEIGHTS_PATH)
        
                
        mid5_0results = run_model_and_get_results(
        text=text,
        model=m5_0model,
        tokenizer=tokenizer,
        index_to_category=m5_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid5_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid5_1results = run_model_and_get_results(
        text=text,
        model=m5_1model,
        tokenizer=tokenizer,
        index_to_category=m5_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid5_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices
       
        mid5_2results = run_model_and_get_results(
        text=text,
        model=m5_2model,
        tokenizer=tokenizer,
        index_to_category=m5_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid5_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        mid5_3results = run_model_and_get_results(
        text=text,
        model=m5_3model,
        tokenizer=tokenizer,
        index_to_category=m5_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid5_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m5_0model
        del m5_1model
        del m5_2model
        del m5_3model
        torch.cuda.empty_cache()

        m6_0model = load_model(BertForMultilabelClassification, model_name, m6_0_number_of_labels, path=M6_0MODEL_WEIGHTS_PATH)
        m6_1model = load_model(BertForMultilabelClassification, model_name, m6_1_number_of_labels, path=M6_1MODEL_WEIGHTS_PATH)
        m6_2model = load_model(BertForMultilabelClassification, model_name, m6_2_number_of_labels, path=M6_2MODEL_WEIGHTS_PATH)
        m6_3model = load_model(BertForMultilabelClassification, model_name, m6_3_number_of_labels, path=M6_3MODEL_WEIGHTS_PATH)
        
        mid6_0results = run_model_and_get_results(
        text=text,
        model=m6_0model,
        tokenizer=tokenizer,
        index_to_category=m6_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid6_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

    
        mid6_1results = run_model_and_get_results(
        text=text,
        model=m6_1model,
        tokenizer=tokenizer,
        index_to_category=m6_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid6_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

    
        mid6_2results = run_model_and_get_results(
        text=text,
        model=m6_2model,
        tokenizer=tokenizer,
        index_to_category=m6_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid6_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

    
        mid6_3results = run_model_and_get_results(
        text=text,
        model=m6_3model,
        tokenizer=tokenizer,
        index_to_category=m6_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid6_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m6_0model
        del m6_1model
        del m6_2model
        del m6_3model
        torch.cuda.empty_cache()
    
        m7_0model = load_model(BertForMultilabelClassification, model_name, m7_0_number_of_labels, path=M7_0MODEL_WEIGHTS_PATH)
        m7_1model = load_model(BertForMultilabelClassification, model_name, m7_1_number_of_labels, path=M7_1MODEL_WEIGHTS_PATH)
        m7_2model = load_model(BertForMultilabelClassification, model_name, m7_2_number_of_labels, path=M7_2MODEL_WEIGHTS_PATH)
        m7_3model = load_model(BertForMultilabelClassification, model_name, m7_3_number_of_labels, path=M7_3MODEL_WEIGHTS_PATH)
        
                
        mid7_0results = run_model_and_get_results(
        text=text,
        model=m7_0model,
        tokenizer=tokenizer,
        index_to_category=m7_0_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid7_0results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

    
        mid7_1results = run_model_and_get_results(
        text=text,
        model=m7_1model,
        tokenizer=tokenizer,
        index_to_category=m7_1_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid7_1results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

    
        mid7_2results = run_model_and_get_results(
        text=text,
        model=m7_2model,
        tokenizer=tokenizer,
        index_to_category=m7_2_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid7_2results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

    
        mid7_3results = run_model_and_get_results(
        text=text,
        model=m7_3model,
        tokenizer=tokenizer,
        index_to_category=m7_3_index_to_category,
        )
        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
        recommended_indices = [category[0] for category in mid7_3results]  # インデックスを抽出
        #print(recommended_indices)
        # インデックスからカテゴリ名に変換
        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
        #print("Recommended Categories:", recommended_categories)
        allrecommended+=recommended_indices

        del m7_0model
        del m7_1model
        del m7_2model
        del m7_3model
        torch.cuda.empty_cache()
            
        print(allrecommended)
        print(evaluation_labels[idx])
        precision,recall,f1 = calculate_metrics(evaluation_labels[idx],allrecommended)
        metricslist.append((precision,recall,f1))
        # allrecommended_binary = convert_labels_to_binary_vectors(allrecommended,allsmall_category_to_index,allsmall_number_of_labels)
        # evaluate_label_binary = convert_labels_to_binary_vectors(evaluation_labels[idx],allsmall_category_to_index,allsmall_number_of_labels)
        # resultbinary.append(allrecommended_binary)
        # truebinary.append(evaluate_label_binary)
        # print(resultbinary)
        # print(truebinary)
    

    average_metrics(metricslist)
    # array_resultbinary=np.array(resultbinary)
    # array_truebinary=np.array(truebinary)

    # print(classification_report(array_truebinary,array_resultbinary,target_names=list(allsmall_index_to_category.values())))
    # report_dict = classification_report(
    # array_truebinary,
    # array_resultbinary,
    # target_names=list(allsmall_index_to_category.values()),
    # output_dict=True
    # )

    # # DataFrame に変換して CSV 出力
    # df = pd.DataFrame(report_dict).transpose()
    # df.to_csv("multistep_8_4_small_ver2.csv", encoding="utf-8")


if __name__ == "__main__":
    main2()