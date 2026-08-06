import json
import logging
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl



from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

M0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid0.json')
M0_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid0-0.json')
M0_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid0-1.json')
M0_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid0-2.json')
M0_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid0-3.json')

M1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid1.json')
M1_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid1-0.json')
M1_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid1-1.json')
M1_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid1-2.json')
M1_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid1-3.json')

M2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid2.json')
M2_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid2-0.json')
M2_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid2-1.json')
M2_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid2-2.json')
M2_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid2-3.json')

M3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid3.json')
M3_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid3-0.json')
M3_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid3-1.json')
M3_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid3-2.json')
M3_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid3-3.json')

M4API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid4.json')
M4_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid4-0.json')
M4_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid4-1.json')
M4_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid4-2.json')
M4_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid4-3.json')

M5API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid5.json')
M5_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid5-0.json')
M5_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid5-1.json')
M5_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid5-2.json')
M5_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid5-3.json')

M6API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid6.json')
M6_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid6-0.json')
M6_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid6-1.json')
M6_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid6-2.json')
M6_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid6-3.json')

M7API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid7.json')
M7_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid7-0.json')
M7_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid7-1.json')
M7_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid7-2.json')
M7_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid7-3.json')

M8API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid8.json')
M8_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid8-0.json')
M8_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid8-1.json')
M8_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid8-2.json')
M8_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid8-3.json')

M9API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid9.json')
M9_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid9-0.json')
M9_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid9-1.json')
M9_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid9-2.json')
M9_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid9-3.json')

M10API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid10.json')
M10_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid10-0.json')
M10_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid10-1.json')
M10_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid10-2.json')
M10_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid10-3.json')

M11API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid11.json')
M11_0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid11-0.json')
M11_1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid11-1.json')
M11_2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid11-2.json')
M11_3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_12big_4mid_categorylist_big12_mid11-3.json')

EVALUATION_DATA_PATH = legacy_path('/home/nakagawa/test_data_mashup_apis.json')

BMODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_model_weights.pth")

M0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid0_model_weights.pth")
M0_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid0-0_model_weights.pth")
M0_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid0-1_model_weights.pth")
M0_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid0-2_model_weights.pth")
M0_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid0-3_model_weights.pth")

M1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid1_model_weights.pth")
M1_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid1-0_model_weights.pth")
M1_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid1-1_model_weights.pth")
M1_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid1-2_model_weights.pth")
M1_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid1-3_model_weights.pth")

M2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid2_model_weights.pth")
M2_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid2-0_model_weights.pth")
M2_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid2-1_model_weights.pth")
M2_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid2-2_model_weights.pth")
M2_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid2-3_model_weights.pth")

M3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid3_model_weights.pth")
M3_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid3-0_model_weights.pth")
M3_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid3-1_model_weights.pth")
M3_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid3-2_model_weights.pth")
M3_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid3-3_model_weights.pth")

M4MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid4_model_weights.pth")
M4_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid4-0_model_weights.pth")
M4_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid4-1_model_weights.pth")
M4_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid4-2_model_weights.pth")
M4_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid4-3_model_weights.pth")

M5MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid5_model_weights.pth")
M5_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid5-0_model_weights.pth")
M5_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid5-1_model_weights.pth")
M5_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid5-2_model_weights.pth")
M5_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid5-3_model_weights.pth")

M6MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid6_model_weights.pth")
M6_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid6-0_model_weights.pth")
M6_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid6-1_model_weights.pth")
M6_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid6-2_model_weights.pth")
M6_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid6-3_model_weights.pth")

M7MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid7_model_weights.pth")
M7_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid7-0_model_weights.pth")
M7_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid7-1_model_weights.pth")
M7_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid7-2_model_weights.pth")
M7_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid7-3_model_weights.pth")

M8MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid8_model_weights.pth")
M8_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid8-0_model_weights.pth")
M8_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid8-1_model_weights.pth")
M8_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid8-2_model_weights.pth")
M8_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid8-3_model_weights.pth")

M9MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid9_model_weights.pth")
M9_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid9-0_model_weights.pth")
M9_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid9-1_model_weights.pth")
M9_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid9-2_model_weights.pth")
M9_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid9-3_model_weights.pth")

M10MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid10_model_weights.pth")
M10_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid10-0_model_weights.pth")
M10_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid10-1_model_weights.pth")
M10_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid10-2_model_weights.pth")
M10_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid10-3_model_weights.pth")

M11MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid11_model_weights.pth")
M11_0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid11-0_model_weights.pth")
M11_1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid11-1_model_weights.pth")
M11_2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid11-2_model_weights.pth")
M11_3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/studied_model/mashup_softfill_api_big12_mid4_big12_mid11-3_model_weights.pth")


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



# Load class list from 'apilist.json' (API names list)
with open(M8API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M8_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M8_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M8_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M8_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid8_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M9API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M9_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M9_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M9_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M9_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid9_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M10API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M10_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M10_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M10_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M10_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid10_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names



# Load class list from 'apilist.json' (API names list)
with open(M11API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M11_0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M11_1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M11_2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M11_3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid11_3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names


big_all_categories_list = [9,3,7,11,10,1,0,4,2,6,5,8]

allmid_all_categories_list = ["0-0","0-1","0-2","0-3",
                              "1-0","1-1","1-2","1-3",
                              "2-0","2-1","2-2","2-3",
                              "3-0","3-1","3-2","3-3",
                              "4-0","4-1","4-2","4-3",
                              "5-0","5-1","5-2","5-3",
                              "6-0","6-1","6-2","6-3",
                              "7-0","7-1","7-2","7-3",
                              "8-0","8-1","8-2","8-3",
                              "9-0","9-1","9-2","9-3",
                              "10-0","10-1","10-2","10-3",
                              "11-0","11-1","11-2","11-3"]

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



m8_category_to_index = {category: idx for idx, category in enumerate(mid8_all_categories_list)}
m8_index_to_category = {idx: category for category, idx in m8_category_to_index.items()}

m8_0_category_to_index = {category: idx for idx, category in enumerate(mid8_0_all_categories_list)}
m8_0_index_to_category = {idx: category for category, idx in m8_0_category_to_index.items()}

m8_1_category_to_index = {category: idx for idx, category in enumerate(mid8_1_all_categories_list)}
m8_1_index_to_category = {idx: category for category, idx in m8_1_category_to_index.items()}

m8_2_category_to_index = {category: idx for idx, category in enumerate(mid8_2_all_categories_list)}
m8_2_index_to_category = {idx: category for category, idx in m8_2_category_to_index.items()}

m8_3_category_to_index = {category: idx for idx, category in enumerate(mid8_3_all_categories_list)}
m8_3_index_to_category = {idx: category for category, idx in m8_3_category_to_index.items()}



m9_category_to_index = {category: idx for idx, category in enumerate(mid9_all_categories_list)}
m9_index_to_category = {idx: category for category, idx in m9_category_to_index.items()}

m9_0_category_to_index = {category: idx for idx, category in enumerate(mid9_0_all_categories_list)}
m9_0_index_to_category = {idx: category for category, idx in m9_0_category_to_index.items()}

m9_1_category_to_index = {category: idx for idx, category in enumerate(mid9_1_all_categories_list)}
m9_1_index_to_category = {idx: category for category, idx in m9_1_category_to_index.items()}

m9_2_category_to_index = {category: idx for idx, category in enumerate(mid9_2_all_categories_list)}
m9_2_index_to_category = {idx: category for category, idx in m9_2_category_to_index.items()}

m9_3_category_to_index = {category: idx for idx, category in enumerate(mid9_3_all_categories_list)}
m9_3_index_to_category = {idx: category for category, idx in m9_3_category_to_index.items()}



m10_category_to_index = {category: idx for idx, category in enumerate(mid10_all_categories_list)}
m10_index_to_category = {idx: category for category, idx in m10_category_to_index.items()}

m10_0_category_to_index = {category: idx for idx, category in enumerate(mid10_0_all_categories_list)}
m10_0_index_to_category = {idx: category for category, idx in m10_0_category_to_index.items()}

m10_1_category_to_index = {category: idx for idx, category in enumerate(mid10_1_all_categories_list)}
m10_1_index_to_category = {idx: category for category, idx in m10_1_category_to_index.items()}

m10_2_category_to_index = {category: idx for idx, category in enumerate(mid10_2_all_categories_list)}
m10_2_index_to_category = {idx: category for category, idx in m10_2_category_to_index.items()}

m10_3_category_to_index = {category: idx for idx, category in enumerate(mid10_3_all_categories_list)}
m10_3_index_to_category = {idx: category for category, idx in m10_3_category_to_index.items()}



m11_category_to_index = {category: idx for idx, category in enumerate(mid11_all_categories_list)}
m11_index_to_category = {idx: category for category, idx in m11_category_to_index.items()}

m11_0_category_to_index = {category: idx for idx, category in enumerate(mid11_0_all_categories_list)}
m11_0_index_to_category = {idx: category for category, idx in m11_0_category_to_index.items()}

m11_1_category_to_index = {category: idx for idx, category in enumerate(mid11_1_all_categories_list)}
m11_1_index_to_category = {idx: category for category, idx in m11_1_category_to_index.items()}

m11_2_category_to_index = {category: idx for idx, category in enumerate(mid11_2_all_categories_list)}
m11_2_index_to_category = {idx: category for category, idx in m11_2_category_to_index.items()}

m11_3_category_to_index = {category: idx for idx, category in enumerate(mid11_3_all_categories_list)}
m11_3_index_to_category = {idx: category for category, idx in m11_3_category_to_index.items()}



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

m8_number_of_labels = len(mid8_all_categories_list) 
m8_0_number_of_labels = len(mid8_0_all_categories_list)
m8_1_number_of_labels = len(mid8_1_all_categories_list)
m8_2_number_of_labels = len(mid8_2_all_categories_list)
m8_3_number_of_labels = len(mid8_3_all_categories_list)

m9_number_of_labels = len(mid9_all_categories_list) 
m9_0_number_of_labels = len(mid9_0_all_categories_list)
m9_1_number_of_labels = len(mid9_1_all_categories_list)
m9_2_number_of_labels = len(mid9_2_all_categories_list)
m9_3_number_of_labels = len(mid9_3_all_categories_list)

m10_number_of_labels = len(mid10_all_categories_list) 
m10_0_number_of_labels = len(mid10_0_all_categories_list)
m10_1_number_of_labels = len(mid10_1_all_categories_list)
m10_2_number_of_labels = len(mid10_2_all_categories_list)
m10_3_number_of_labels = len(mid10_3_all_categories_list)

m11_number_of_labels = len(mid11_all_categories_list) 
m11_0_number_of_labels = len(mid11_0_all_categories_list)
m11_1_number_of_labels = len(mid11_1_all_categories_list)
m11_2_number_of_labels = len(mid11_2_all_categories_list)
m11_3_number_of_labels = len(mid11_3_all_categories_list)

allmid_number_of_labels = len(allmid_all_categories_list) 
allsmall_number_of_labels = len(allsmall_all_categories_list)


# Load evaluation data from 'mashup_data_new_test_data.json'
with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f:
    mashup_data = json.load(f)

# Extract evaluation texts and labels from mashup_data_new_test_data.json
evaluation_texts = []
evaluation_labels = []

# Extract categories and descriptions for evaluation
for api_name,api_info in mashup_data.items():
    description = api_info[1]  # Extract description as the text
    used_api = api_info[2]
    evaluation_texts.append(description)
    evaluation_labels.append(used_api)


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
    false_positive = len(predicted_set - true_set)
    false_negative = len(true_set - predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0
    recall = true_positive / len(true_set) if true_set else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    print(f"Precision: {precision}, Recall: {recall}, F-Score: {f_score}")
    print(f"False Positive: {false_positive}, False Negative: {false_negative}")
    return precision, recall, f_score, false_positive, false_negative

def average_metrics(metrics_list):
    
    """
    渡されたリストから適合率、再現率、Fスコアの平均を計算する。
    """
    avg_precision = sum(m[0] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_recall = sum(m[1] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_f_score = sum(m[2] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_false_positive = sum(m[3] for m in metrics_list) / len(metrics_list)
    avg_false_negative = sum(m[4] for m in metrics_list) / len(metrics_list)
    print("テストごと平均")
    print(f"Precision: {avg_precision}, Recall: {avg_recall}, F-Score: {avg_f_score}")
    print(f"Average False Positive: {avg_false_positive}, False Negative: {avg_false_negative}")
    return avg_precision, avg_recall, avg_f_score, avg_false_positive, avg_false_negative

def per_label_metrics(true_labels_list, pred_labels_list, all_labels):
    """
    各ラベルに対して Precision, Recall, F1 を計算する。
    """
    # ラベルごとの TP / FP / FN を初期化
    label_tp = {label: 0 for label in all_labels}
    label_fp = {label: 0 for label in all_labels}
    label_fn = {label: 0 for label in all_labels}

    # 全サンプルを走査して TP / FP / FN をカウント
    for true_set, pred_set in zip(true_labels_list, pred_labels_list):
        for label in all_labels:
            if label in true_set and label in pred_set:
                label_tp[label] += 1
            elif label not in true_set and label in pred_set:
                label_fp[label] += 1
            elif label in true_set and label not in pred_set:
                label_fn[label] += 1
            # 両方に含まれない場合は何もしない

    # ラベルごとの Precision / Recall / F1 を計算
    metrics = {}
    precision_list = []
    recall_list = []
    f1_list = []
    for label in all_labels:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
        
        # 平均用リストに追加
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # ラベルごとの平均値
        average_metrics = {
        "precision": sum(precision_list) / len(precision_list) if precision_list else 0,
        "recall": sum(recall_list) / len(recall_list) if recall_list else 0,
        "f1": sum(f1_list) / len(f1_list) if f1_list else 0,
        }

    return metrics, average_metrics

model_name = legacy_path("/home/nakagawa/models/bert-base-uncased/")
tokenizer = BertTokenizer.from_pretrained(model_name)

bigmodel = load_model(BertForMultilabelClassification, model_name, big_number_of_labels, path=BMODEL_WEIGHTS_PATH)
m0model = load_model(BertForMultilabelClassification, model_name, m0_number_of_labels, path=M0MODEL_WEIGHTS_PATH)
# m0_0model = load_model(BertForMultilabelClassification, model_name, m0_0_number_of_labels, path=M0_0MODEL_WEIGHTS_PATH)
# m0_1model = load_model(BertForMultilabelClassification, model_name, m0_1_number_of_labels, path=M0_1MODEL_WEIGHTS_PATH)
# m0_2model = load_model(BertForMultilabelClassification, model_name, m0_2_number_of_labels, path=M0_2MODEL_WEIGHTS_PATH)
# m0_3model = load_model(BertForMultilabelClassification, model_name, m0_3_number_of_labels, path=M0_3MODEL_WEIGHTS_PATH)

m1model = load_model(BertForMultilabelClassification, model_name, m1_number_of_labels, path=M1MODEL_WEIGHTS_PATH)
# m1_0model = load_model(BertForMultilabelClassification, model_name, m1_0_number_of_labels, path=M1_0MODEL_WEIGHTS_PATH)
# m1_1model = load_model(BertForMultilabelClassification, model_name, m1_1_number_of_labels, path=M1_1MODEL_WEIGHTS_PATH)
# m1_2model = load_model(BertForMultilabelClassification, model_name, m1_2_number_of_labels, path=M1_2MODEL_WEIGHTS_PATH)
# m1_3model = load_model(BertForMultilabelClassification, model_name, m1_3_number_of_labels, path=M1_3MODEL_WEIGHTS_PATH)

m2model = load_model(BertForMultilabelClassification, model_name, m2_number_of_labels, path=M2MODEL_WEIGHTS_PATH)
# m2_0model = load_model(BertForMultilabelClassification, model_name, m2_0_number_of_labels, path=M2_0MODEL_WEIGHTS_PATH)
# m2_1model = load_model(BertForMultilabelClassification, model_name, m2_1_number_of_labels, path=M2_1MODEL_WEIGHTS_PATH)
# m2_2model = load_model(BertForMultilabelClassification, model_name, m2_2_number_of_labels, path=M2_2MODEL_WEIGHTS_PATH)
# m2_3model = load_model(BertForMultilabelClassification, model_name, m2_3_number_of_labels, path=M2_3MODEL_WEIGHTS_PATH)

m3model = load_model(BertForMultilabelClassification, model_name, m3_number_of_labels, path=M3MODEL_WEIGHTS_PATH)
# m3_0model = load_model(BertForMultilabelClassification, model_name, m3_0_number_of_labels, path=M3_0MODEL_WEIGHTS_PATH)
# m3_1model = load_model(BertForMultilabelClassification, model_name, m3_1_number_of_labels, path=M3_1MODEL_WEIGHTS_PATH)
# m3_2model = load_model(BertForMultilabelClassification, model_name, m3_2_number_of_labels, path=M3_2MODEL_WEIGHTS_PATH)
# m3_3model = load_model(BertForMultilabelClassification, model_name, m3_3_number_of_labels, path=M3_3MODEL_WEIGHTS_PATH)

m4model = load_model(BertForMultilabelClassification, model_name, m4_number_of_labels, path=M4MODEL_WEIGHTS_PATH)
# m4_0model = load_model(BertForMultilabelClassification, model_name, m4_0_number_of_labels, path=M4_0MODEL_WEIGHTS_PATH)
# m4_1model = load_model(BertForMultilabelClassification, model_name, m4_1_number_of_labels, path=M4_1MODEL_WEIGHTS_PATH)
# m4_2model = load_model(BertForMultilabelClassification, model_name, m4_2_number_of_labels, path=M4_2MODEL_WEIGHTS_PATH)
# m4_3model = load_model(BertForMultilabelClassification, model_name, m4_3_number_of_labels, path=M4_3MODEL_WEIGHTS_PATH)

m5model = load_model(BertForMultilabelClassification, model_name, m5_number_of_labels, path=M5MODEL_WEIGHTS_PATH)
# m5_0model = load_model(BertForMultilabelClassification, model_name, m5_0_number_of_labels, path=M5_0MODEL_WEIGHTS_PATH)
# m5_1model = load_model(BertForMultilabelClassification, model_name, m5_1_number_of_labels, path=M5_1MODEL_WEIGHTS_PATH)
# m5_2model = load_model(BertForMultilabelClassification, model_name, m5_2_number_of_labels, path=M5_2MODEL_WEIGHTS_PATH)
# m5_3model = load_model(BertForMultilabelClassification, model_name, m5_3_number_of_labels, path=M5_3MODEL_WEIGHTS_PATH)

m6model = load_model(BertForMultilabelClassification, model_name, m6_number_of_labels, path=M6MODEL_WEIGHTS_PATH)
# m6_0model = load_model(BertForMultilabelClassification, model_name, m6_0_number_of_labels, path=M6_0MODEL_WEIGHTS_PATH)
# m6_1model = load_model(BertForMultilabelClassification, model_name, m6_1_number_of_labels, path=M6_1MODEL_WEIGHTS_PATH)
# m6_2model = load_model(BertForMultilabelClassification, model_name, m6_2_number_of_labels, path=M6_2MODEL_WEIGHTS_PATH)
# m6_3model = load_model(BertForMultilabelClassification, model_name, m6_3_number_of_labels, path=M6_3MODEL_WEIGHTS_PATH)

m7model = load_model(BertForMultilabelClassification, model_name, m7_number_of_labels, path=M7MODEL_WEIGHTS_PATH)
# m7_0model = load_model(BertForMultilabelClassification, model_name, m7_0_number_of_labels, path=M7_0MODEL_WEIGHTS_PATH)
# m7_1model = load_model(BertForMultilabelClassification, model_name, m7_1_number_of_labels, path=M7_1MODEL_WEIGHTS_PATH)
# m7_2model = load_model(BertForMultilabelClassification, model_name, m7_2_number_of_labels, path=M7_2MODEL_WEIGHTS_PATH)
# m7_3model = load_model(BertForMultilabelClassification, model_name, m7_3_number_of_labels, path=M7_3MODEL_WEIGHTS_PATH)

m8model = load_model(BertForMultilabelClassification, model_name, m8_number_of_labels, path=M8MODEL_WEIGHTS_PATH)

m9model = load_model(BertForMultilabelClassification, model_name, m9_number_of_labels, path=M9MODEL_WEIGHTS_PATH)

m10model = load_model(BertForMultilabelClassification, model_name, m10_number_of_labels, path=M10MODEL_WEIGHTS_PATH)

m11model = load_model(BertForMultilabelClassification, model_name, m11_number_of_labels, path=M11MODEL_WEIGHTS_PATH)

# Main evaluation function (modified version of main2)
def main2():
    # 実行
    metricslist = []
    true_labels_list = []
    pred_labels_list = []
    for idx, text in enumerate(evaluation_texts):
        allrecommended = []
        allapirec = []
        print(f"Processing text {idx + 1}/{len(evaluation_texts)}")
        print(text[:100])
        
        bigresults = run_model_and_get_results(
            text=text,
            model=bigmodel,
            tokenizer=tokenizer,
            index_to_category=big_index_to_category,
        )

        #print(bigresults)
        for nextrun in bigresults:
            if nextrun[0] == 0:
                m0_0model = load_model(BertForMultilabelClassification, model_name, m0_0_number_of_labels, path=M0_0MODEL_WEIGHTS_PATH)
                m0_1model = load_model(BertForMultilabelClassification, model_name, m0_1_number_of_labels, path=M0_1MODEL_WEIGHTS_PATH)
                m0_2model = load_model(BertForMultilabelClassification, model_name, m0_2_number_of_labels, path=M0_2MODEL_WEIGHTS_PATH)
                m0_3model = load_model(BertForMultilabelClassification, model_name, m0_3_number_of_labels, path=M0_3MODEL_WEIGHTS_PATH)
                mid0results = run_model_and_get_results(
                text=text,
                model=m0model,
                tokenizer=tokenizer,
                index_to_category=m0_index_to_category,
                )
                for nextrun in mid0results:
                    if nextrun[0] == "0-0":
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


                    elif nextrun[0] == "0-1":
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

                    elif nextrun[0] == "0-2":
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

                    elif nextrun[0] == "0-3":
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
            
            elif nextrun[0] == 1:
                m1_0model = load_model(BertForMultilabelClassification, model_name, m1_0_number_of_labels, path=M1_0MODEL_WEIGHTS_PATH)
                m1_1model = load_model(BertForMultilabelClassification, model_name, m1_1_number_of_labels, path=M1_1MODEL_WEIGHTS_PATH)
                m1_2model = load_model(BertForMultilabelClassification, model_name, m1_2_number_of_labels, path=M1_2MODEL_WEIGHTS_PATH)
                m1_3model = load_model(BertForMultilabelClassification, model_name, m1_3_number_of_labels, path=M1_3MODEL_WEIGHTS_PATH)
                mid1results = run_model_and_get_results(
                text=text,
                model=m1model,
                tokenizer=tokenizer,
                index_to_category=m1_index_to_category,
                )
                for nextrun in mid1results:
                    if nextrun[0] == "1-0":
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

                    elif nextrun[0] == "1-1":
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

                    elif nextrun[0] == "1-2":
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

                    elif nextrun[0] == "1-3":
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
            
            elif nextrun[0] == 2:
                m2_0model = load_model(BertForMultilabelClassification, model_name, m2_0_number_of_labels, path=M2_0MODEL_WEIGHTS_PATH)
                m2_1model = load_model(BertForMultilabelClassification, model_name, m2_1_number_of_labels, path=M2_1MODEL_WEIGHTS_PATH)
                m2_2model = load_model(BertForMultilabelClassification, model_name, m2_2_number_of_labels, path=M2_2MODEL_WEIGHTS_PATH)
                m2_3model = load_model(BertForMultilabelClassification, model_name, m2_3_number_of_labels, path=M2_3MODEL_WEIGHTS_PATH)
                mid2results = run_model_and_get_results(
                text=text,
                model=m2model,
                tokenizer=tokenizer,
                index_to_category=m2_index_to_category,
                )
                for nextrun in mid2results:
                    if nextrun[0] == "2-0":
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

                    elif nextrun[0] == "2-1":
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

                    elif nextrun[0] == "2-2":
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

                    elif nextrun[0] == "2-3":
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
            
            elif nextrun[0] == 3:
                m3_0model = load_model(BertForMultilabelClassification, model_name, m3_0_number_of_labels, path=M3_0MODEL_WEIGHTS_PATH)
                m3_1model = load_model(BertForMultilabelClassification, model_name, m3_1_number_of_labels, path=M3_1MODEL_WEIGHTS_PATH)
                m3_2model = load_model(BertForMultilabelClassification, model_name, m3_2_number_of_labels, path=M3_2MODEL_WEIGHTS_PATH)
                m3_3model = load_model(BertForMultilabelClassification, model_name, m3_3_number_of_labels, path=M3_3MODEL_WEIGHTS_PATH)
                mid3results = run_model_and_get_results(
                text=text,
                model=m3model,
                tokenizer=tokenizer,
                index_to_category=m3_index_to_category,
                )
                for nextrun in mid3results:
                    if nextrun[0] == "3-0":
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

                    elif nextrun[0] == "3-1":
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

                    elif nextrun[0] == "3-2":
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

                    elif nextrun[0] == "3-3":
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
                    
            elif nextrun[0] == 4:
                m4_0model = load_model(BertForMultilabelClassification, model_name, m4_0_number_of_labels, path=M4_0MODEL_WEIGHTS_PATH)
                m4_1model = load_model(BertForMultilabelClassification, model_name, m4_1_number_of_labels, path=M4_1MODEL_WEIGHTS_PATH)
                m4_2model = load_model(BertForMultilabelClassification, model_name, m4_2_number_of_labels, path=M4_2MODEL_WEIGHTS_PATH)
                m4_3model = load_model(BertForMultilabelClassification, model_name, m4_3_number_of_labels, path=M4_3MODEL_WEIGHTS_PATH)
                mid4results = run_model_and_get_results(
                text=text,
                model=m4model,
                tokenizer=tokenizer,
                index_to_category=m4_index_to_category,
                )
                for nextrun in mid4results:
                    if nextrun[0] == "4-0":
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

                    elif nextrun[0] == "4-1":
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

                    elif nextrun[0] == "4-2":
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

                    elif nextrun[0] == "4-3":
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

            
            elif nextrun[0] == 5:
                m5_0model = load_model(BertForMultilabelClassification, model_name, m5_0_number_of_labels, path=M5_0MODEL_WEIGHTS_PATH)
                m5_1model = load_model(BertForMultilabelClassification, model_name, m5_1_number_of_labels, path=M5_1MODEL_WEIGHTS_PATH)
                m5_2model = load_model(BertForMultilabelClassification, model_name, m5_2_number_of_labels, path=M5_2MODEL_WEIGHTS_PATH)
                m5_3model = load_model(BertForMultilabelClassification, model_name, m5_3_number_of_labels, path=M5_3MODEL_WEIGHTS_PATH)
                mid5results = run_model_and_get_results(
                text=text,
                model=m5model,
                tokenizer=tokenizer,
                index_to_category=m5_index_to_category,
                )
                for nextrun in mid5results:
                    if nextrun[0] == "5-0":
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

                    elif nextrun[0] == "5-1":
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

                    elif nextrun[0] == "5-2":
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

                    elif nextrun[0] == "5-3":
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

            elif nextrun[0] == 6:
                m6_0model = load_model(BertForMultilabelClassification, model_name, m6_0_number_of_labels, path=M6_0MODEL_WEIGHTS_PATH)
                m6_1model = load_model(BertForMultilabelClassification, model_name, m6_1_number_of_labels, path=M6_1MODEL_WEIGHTS_PATH)
                m6_2model = load_model(BertForMultilabelClassification, model_name, m6_2_number_of_labels, path=M6_2MODEL_WEIGHTS_PATH)
                m6_3model = load_model(BertForMultilabelClassification, model_name, m6_3_number_of_labels, path=M6_3MODEL_WEIGHTS_PATH)
                mid6results = run_model_and_get_results(
                text=text,
                model=m6model,
                tokenizer=tokenizer,
                index_to_category=m6_index_to_category,
                )
                for nextrun in mid6results:
                    if nextrun[0] == "6-0":
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

                    elif nextrun[0] == "6-1":
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

                    elif nextrun[0] == "6-2":
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

                    elif nextrun[0] == "6-3":
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
            
            elif nextrun[0] == 7:
                m7_0model = load_model(BertForMultilabelClassification, model_name, m7_0_number_of_labels, path=M7_0MODEL_WEIGHTS_PATH)
                m7_1model = load_model(BertForMultilabelClassification, model_name, m7_1_number_of_labels, path=M7_1MODEL_WEIGHTS_PATH)
                m7_2model = load_model(BertForMultilabelClassification, model_name, m7_2_number_of_labels, path=M7_2MODEL_WEIGHTS_PATH)
                m7_3model = load_model(BertForMultilabelClassification, model_name, m7_3_number_of_labels, path=M7_3MODEL_WEIGHTS_PATH)
                mid7results = run_model_and_get_results(
                text=text,
                model=m7model,
                tokenizer=tokenizer,
                index_to_category=m7_index_to_category,
                )
                for nextrun in mid7results:
                    if nextrun[0] == "7-0":
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

                    elif nextrun[0] == "7-1":
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

                    elif nextrun[0] == "7-2":
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

                    elif nextrun[0] == "7-3":
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

            elif nextrun[0] == 8:
                m8_0model = load_model(BertForMultilabelClassification, model_name, m8_0_number_of_labels, path=M8_0MODEL_WEIGHTS_PATH)
                m8_1model = load_model(BertForMultilabelClassification, model_name, m8_1_number_of_labels, path=M8_1MODEL_WEIGHTS_PATH)
                m8_2model = load_model(BertForMultilabelClassification, model_name, m8_2_number_of_labels, path=M8_2MODEL_WEIGHTS_PATH)
                m8_3model = load_model(BertForMultilabelClassification, model_name, m8_3_number_of_labels, path=M8_3MODEL_WEIGHTS_PATH)
                mid8results = run_model_and_get_results(
                text=text,
                model=m8model,
                tokenizer=tokenizer,
                index_to_category=m8_index_to_category,
                )
                for nextrun in mid8results:
                    if nextrun[0] == "8-0":
                        mid8_0results = run_model_and_get_results(
                        text=text,
                        model=m8_0model,
                        tokenizer=tokenizer,
                        index_to_category=m8_0_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid8_0results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "8-1":
                        mid8_1results = run_model_and_get_results(
                        text=text,
                        model=m8_1model,
                        tokenizer=tokenizer,
                        index_to_category=m8_1_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid8_1results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "8-2":
                        mid8_2results = run_model_and_get_results(
                        text=text,
                        model=m8_2model,
                        tokenizer=tokenizer,
                        index_to_category=m8_2_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid8_2results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "8-3":
                        mid8_3results = run_model_and_get_results(
                        text=text,
                        model=m8_3model,
                        tokenizer=tokenizer,
                        index_to_category=m8_3_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid8_3results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                del m8_0model
                del m8_1model
                del m8_2model
                del m8_3model
                torch.cuda.empty_cache()


            elif nextrun[0] == 9:
                m9_0model = load_model(BertForMultilabelClassification, model_name, m9_0_number_of_labels, path=M9_0MODEL_WEIGHTS_PATH)
                m9_1model = load_model(BertForMultilabelClassification, model_name, m9_1_number_of_labels, path=M9_1MODEL_WEIGHTS_PATH)
                m9_2model = load_model(BertForMultilabelClassification, model_name, m9_2_number_of_labels, path=M9_2MODEL_WEIGHTS_PATH)
                m9_3model = load_model(BertForMultilabelClassification, model_name, m9_3_number_of_labels, path=M9_3MODEL_WEIGHTS_PATH)
                mid9results = run_model_and_get_results(
                text=text,
                model=m9model,
                tokenizer=tokenizer,
                index_to_category=m9_index_to_category,
                )
                for nextrun in mid9results:
                    if nextrun[0] == "9-0":
                        mid9_0results = run_model_and_get_results(
                        text=text,
                        model=m9_0model,
                        tokenizer=tokenizer,
                        index_to_category=m9_0_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid9_0results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "9-1":
                        mid9_1results = run_model_and_get_results(
                        text=text,
                        model=m9_1model,
                        tokenizer=tokenizer,
                        index_to_category=m9_1_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid9_1results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "9-2":
                        mid9_2results = run_model_and_get_results(
                        text=text,
                        model=m9_2model,
                        tokenizer=tokenizer,
                        index_to_category=m9_2_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid9_2results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "9-3":
                        mid9_3results = run_model_and_get_results(
                        text=text,
                        model=m9_3model,
                        tokenizer=tokenizer,
                        index_to_category=m9_3_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid9_3results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                del m9_0model
                del m9_1model
                del m9_2model
                del m9_3model
                torch.cuda.empty_cache() 


            elif nextrun[0] == 10:
                m10_0model = load_model(BertForMultilabelClassification, model_name, m10_0_number_of_labels, path=M10_0MODEL_WEIGHTS_PATH)
                m10_1model = load_model(BertForMultilabelClassification, model_name, m10_1_number_of_labels, path=M10_1MODEL_WEIGHTS_PATH)
                m10_2model = load_model(BertForMultilabelClassification, model_name, m10_2_number_of_labels, path=M10_2MODEL_WEIGHTS_PATH)
                m10_3model = load_model(BertForMultilabelClassification, model_name, m10_3_number_of_labels, path=M10_3MODEL_WEIGHTS_PATH)
                mid10results = run_model_and_get_results(
                text=text,
                model=m10model,
                tokenizer=tokenizer,
                index_to_category=m10_index_to_category,
                )
                for nextrun in mid10results:
                    if nextrun[0] == "10-0":
                        mid10_0results = run_model_and_get_results(
                        text=text,
                        model=m10_0model,
                        tokenizer=tokenizer,
                        index_to_category=m10_0_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid10_0results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "10-1":
                        mid10_1results = run_model_and_get_results(
                        text=text,
                        model=m10_1model,
                        tokenizer=tokenizer,
                        index_to_category=m10_1_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid10_1results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "10-2":
                        mid10_2results = run_model_and_get_results(
                        text=text,
                        model=m10_2model,
                        tokenizer=tokenizer,
                        index_to_category=m10_2_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid10_2results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "10-3":
                        mid10_3results = run_model_and_get_results(
                        text=text,
                        model=m10_3model,
                        tokenizer=tokenizer,
                        index_to_category=m10_3_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid10_3results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                del m10_0model
                del m10_1model
                del m10_2model
                del m10_3model
                torch.cuda.empty_cache()


            elif nextrun[0] == 11:
                m11_0model = load_model(BertForMultilabelClassification, model_name, m11_0_number_of_labels, path=M11_0MODEL_WEIGHTS_PATH)
                m11_1model = load_model(BertForMultilabelClassification, model_name, m11_1_number_of_labels, path=M11_1MODEL_WEIGHTS_PATH)
                m11_2model = load_model(BertForMultilabelClassification, model_name, m11_2_number_of_labels, path=M11_2MODEL_WEIGHTS_PATH)
                m11_3model = load_model(BertForMultilabelClassification, model_name, m11_3_number_of_labels, path=M11_3MODEL_WEIGHTS_PATH)
                mid11results = run_model_and_get_results(
                text=text,
                model=m11model,
                tokenizer=tokenizer,
                index_to_category=m11_index_to_category,
                )
                for nextrun in mid11results:
                    if nextrun[0] == "11-0":
                        mid11_0results = run_model_and_get_results(
                        text=text,
                        model=m11_0model,
                        tokenizer=tokenizer,
                        index_to_category=m11_0_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid11_0results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "11-1":
                        mid11_1results = run_model_and_get_results(
                        text=text,
                        model=m11_1model,
                        tokenizer=tokenizer,
                        index_to_category=m11_1_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid11_1results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "11-2":
                        mid11_2results = run_model_and_get_results(
                        text=text,
                        model=m11_2model,
                        tokenizer=tokenizer,
                        index_to_category=m11_2_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid11_2results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                    elif nextrun[0] == "11-3":
                        mid11_3results = run_model_and_get_results(
                        text=text,
                        model=m11_3model,
                        tokenizer=tokenizer,
                        index_to_category=m11_3_index_to_category,
                        )
                        # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                        recommended_indices = [category[0] for category in mid11_3results]  # インデックスを抽出
                        #print(recommended_indices)
                        # インデックスからカテゴリ名に変換
                        #recommended_categories = [m0_1_index_to_category[index] for index in recommended_indices]
                        #print("Recommended Categories:", recommended_categories)
                        allrecommended+=recommended_indices

                del m11_0model
                del m11_1model
                del m11_2model
                del m11_3model
                torch.cuda.empty_cache()               
            
        print(allrecommended)
        
        for cat in allrecommended:
            # パス規則（あなたの指定どおり）
            print(cat+"のAPI推薦")
            model_path   = legacy_path(f"/home/nakagawa/study_category_inc_nomashup_negative_1_3_model_new/bert_multilabel_{cat}_model_weights.pth")
            apilist_path = legacy_path(f"/home/nakagawa/api_category_list/category_inc_nomashup_negative_1_3_new/{cat}_api_labels.json")
            # apilist 読み込み
            try:
                with open(apilist_path, "r", encoding="utf-8") as f:
                    apilist = json.load(f)
            except FileNotFoundError:
                logging.error("ファイルが見つかりません: %s", apilist_path)
                print("飛ばす")
                continue

            category_to_index = {category: idx for idx, category in enumerate(apilist)}
            index_to_category = {idx: category for category, idx in category_to_index.items()}
            num_labels = len(index_to_category)
            
            if num_labels == 1:
                recommended_indices = [category for category in apilist]
                allapirec+=recommended_indices
            
            else:    
                # モデルロード（あなた指定どおり）
                model = load_model(
                    BertForMultilabelClassification,
                    model_name,
                    num_labels,
                    path=model_path
                )
    
                out = run_model_and_get_results(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    index_to_category=index_to_category,  # ← ここで使う
                )
                
                recommended_indices = [category[0] for category in out]
                allapirec+=recommended_indices
            
        print(allapirec)
        print(evaluation_labels[idx])
        
        
        precision,recall,f1,fp,fn = calculate_metrics(evaluation_labels[idx],allapirec)
        metricslist.append((precision,recall,f1,fp,fn))
        true_labels_list.append(evaluation_labels[idx])
        pred_labels_list.append(allapirec)


    average_metrics(metricslist)
    all_labels = set().union(*true_labels_list, *pred_labels_list)
    label_metrics,average_label_metrics= per_label_metrics(true_labels_list, pred_labels_list, all_labels)
    for label, m in label_metrics.items():
        print(label, m)
    print("ラベルごと平均")
    print(average_label_metrics)


if __name__ == "__main__":
    main2()