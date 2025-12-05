import json
import logging
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

EVALUATION_DATA_PATH = '/home/nakagawa/test_data_mashup_apis.json'

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

model_name = "/home/nakagawa/models/bert-base-uncased/"
tokenizer = BertTokenizer.from_pretrained(model_name)
layer1_category_num = "15"

def main2():
    # 実行
    metricslist = []
    true_labels_list = []
    pred_labels_list = []
    for idx, text in enumerate(evaluation_texts):
        all = []
        allapirec = []
        print(f"Processing text {idx + 1}/{len(evaluation_texts)}")
        print(text[:100])
        layer1_model_path   = f"/home/nakagawa/layer2_{layer1_category_num}/mashup_softfill_api_all_layer2_{layer1_category_num}_model_weights.pth"
        layer1_apilist_path = f"/home/nakagawa/api_category_list/layer2_{layer1_category_num}/softfill_inc_categorylist_layer2_{layer1_category_num}.json"
        try:
            with open(layer1_apilist_path, "r", encoding="utf-8") as f:
                layer1_apilist = json.load(f)
        except FileNotFoundError:
            logging.error("ファイルが見つかりません: %s", layer1_apilist_path)
            print("飛ばす")
            continue

        layer1_category_to_index = {category: idx for idx, category in enumerate(layer1_apilist)}
        layer1_index_to_category = {idx: category for category, idx in layer1_category_to_index.items()}
        layer1_num_labels = len(layer1_index_to_category)
        # モデルロード（あなた指定どおり）
        layer1_model = load_model(
            BertForMultilabelClassification,
            model_name,
            layer1_num_labels,
            path=layer1_model_path
        )
        
        layer1_results = run_model_and_get_results(
            text=text,
            model=layer1_model,
            tokenizer=tokenizer,
            index_to_category=layer1_index_to_category,
        )
        
        for l1_cat,l1_val in layer1_results:
            layer2_model_path   = f"/home/nakagawa/layer2_{layer1_category_num}/mashup_softfill_api_all_layer2_{layer1_category_num}_{l1_cat}_model_weights.pth"
            layer2_apilist_path = f"/home/nakagawa/api_category_list/layer2_{layer1_category_num}/softfill_inc_categorylist_layer2_{layer1_category_num}_{l1_cat}.json"
            try:
                with open(layer2_apilist_path, "r", encoding="utf-8") as f:
                    layer2_apilist = json.load(f)
            except FileNotFoundError:
                logging.error("ファイルが見つかりません: %s", layer2_apilist_path)
                print("飛ばす")
                continue

            layer2_category_to_index = {category: idx for idx, category in enumerate(layer2_apilist)}
            layer2_index_to_category = {idx: category for category, idx in layer2_category_to_index.items()}
            layer2_num_labels = len(layer2_index_to_category)
            # モデルロード（あなた指定どおり）
            layer2_model = load_model(
                BertForMultilabelClassification,
                model_name,
                layer2_num_labels,
                path=layer2_model_path
            )
            
            layer2_results = run_model_and_get_results(
                text=text,
                model=layer2_model,
                tokenizer=tokenizer,
                index_to_category=layer2_index_to_category,
            )

            for cat,l2_val in layer2_results:
                # パス規則（あなたの指定どおり）
                print(cat+"のAPI推薦")
                model_path   = f"/home/nakagawa/study_category_inc_nomashup_model_new_kfold/bert_multilabel_{cat}_model_weights.pth"
                apilist_path = f"/home/nakagawa/api_category_list/category_inc_nomashup_new_kfold/{cat}_api_labels.json"
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