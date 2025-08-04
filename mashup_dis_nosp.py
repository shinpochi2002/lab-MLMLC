import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_categorylist_allsmall.json'
EVALUATION_DATA_PATH = '/home/nakagawa/collaboration_test_data_layer1_15.json'
MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_all_layer2_15_model_weights.pth"


# # Load class list from 'apilist.json' (API names list)
# with open(API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
#     all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names
all_categories_list=[
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



# Create mappings for the categories
category_to_index = {category: idx for idx, category in enumerate(all_categories_list)}
index_to_category = {idx: category for category, idx in category_to_index.items()}

# Total number of labels
number_of_labels = len(all_categories_list)  

# Load evaluation data from 'mashup_data_new_test_data.json'
with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f:
    mashup_data = json.load(f)

# Extract evaluation texts and labels from mashup_data_new_test_data.json
evaluation_texts = []
evaluation_labels = []

# Extract categories and descriptions for evaluation
for api_info in mashup_data:
    categories = api_info["bigcategory"]  # Extract categories as labels
    description = api_info["description"]  # Extract description as the text
    evaluation_texts.append(description)
    evaluation_labels.append(categories)

# Convert evaluation labels to binary vectors (1538-dimensional)
evaluation_label_indices = []
for label in evaluation_labels:
    # 単一カテゴリの場合の処理
    indices = [category_to_index[category] for category in label]
    # 単一カテゴリの場合の処理
    label_vector = [0] * number_of_labels
    for idx in indices:
        label_vector[idx] = 1
    evaluation_label_indices.append(label_vector)


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
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
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


def main2():
    # Load the tokenizer
    model_name = "/home/nakagawa/models/bert-base-uncased/"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load the saved model
    model = load_model(BertForMultilabelClassification, model_name, number_of_labels,path=MODEL_WEIGHTS_PATH)

    # Prepare the list to store detailed prediction results
    # detailed_predictions = []
    # all_true_labels = []
    # all_pred_labels = []  # 予測ラベルのバイナリベクトルを保持
    # all_pred_probs = []

    metricslist = []

    # Iterate over each text and save prediction results
    for idx, text in enumerate(evaluation_texts):
        print(f"Processing text {idx + 1}/{len(evaluation_texts)}")
        print(text[:100])
        allrecommended = []
        results = run_model_and_get_results(
                        text=text,
                        model=model,
                        tokenizer=tokenizer,
                        index_to_category=index_to_category,
                        )
        recommended_indices = [category[0] for category in results]  # インデックスを抽出
        allrecommended+=recommended_indices
        print(evaluation_labels[idx])


        precision,recall,f1 = calculate_metrics(evaluation_labels[idx],allrecommended)
        metricslist.append((precision,recall,f1))
    
    
    # Missing categories check
    missing_categories = []
    for label_list in evaluation_labels:
        for category in label_list:
            if category not in category_to_index:
                missing_categories.append(category)

    print("Missing categories:", set(missing_categories))

    average_metrics(metricslist)

if __name__ == "__main__":
    main2()