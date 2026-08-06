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

M0API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_0.json')

M1API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_1.json')

M2API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_2.json')

M3API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_3.json')

M4API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_4.json')

M5API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_5.json')

M6API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_6.json')

M7API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_7.json')

M8API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_8.json')

M9API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_9.json')

M10API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_10.json')

M11API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_11.json')

M12API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_12.json')

M13API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_13.json')

M14API_CATEGORY_LIST_PATH = legacy_path('/home/nakagawa/api_category_list/softfill_inc_categorylist_layer2_15_14.json')


EVALUATION_DATA_PATH = legacy_path('/home/nakagawa/collaboration_test_data.json')


BMODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_model_weights.pth")

M0MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_0_model_weights.pth")

M1MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_1_model_weights.pth")

M2MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_2_model_weights.pth")

M3MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_3_model_weights.pth")

M4MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_4_model_weights.pth")

M5MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_5_model_weights.pth")

M6MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_6_model_weights.pth")

M7MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_7_model_weights.pth")

M8MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_8_model_weights.pth")

M9MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_9_model_weights.pth")

M10MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_10_model_weights.pth")

M11MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_11_model_weights.pth")

M12MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_12_model_weights.pth")

M13MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_13_model_weights.pth")

M14MODEL_WEIGHTS_PATH = legacy_path("/home/nakagawa/mashup_softfill_api_all_layer2_15_14_model_weights.pth")



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


big_all_categories_list = [
    7,
    1,
    13,
    0,
    11,
    3,
    14,
    6,
    10,
    12,
    8,
    4,
    9,
    2,
    5
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
            
        print(allrecommended)
        print(evaluation_labels[idx])
        precision,recall,f1 = calculate_metrics(evaluation_labels[idx],allrecommended)
        metricslist.append((precision,recall,f1))
        #print(allrecommended_binary)
        #print(evaluate_label_binary)
    
    average_metrics(metricslist)


if __name__ == "__main__":
    main2()