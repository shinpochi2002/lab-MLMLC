import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# ====== 中川さん提供のコア部分 (一切変更なし) ======

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


def load_model(model_class, model_name, num_labels, path=None):
    model = model_class(model_name, num_labels)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model.to('cuda')


def predict(text, model, tokenizer, index_to_category, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to('cuda')
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    predicted_indices = np.where(probs > threshold)[0]
    recommended_categories = [(index_to_category[idx], probs[idx]) for idx in predicted_indices]
    return recommended_categories, probs


def run_model_and_get_results(text, model, tokenizer, index_to_category, threshold=0.5):
    _, probs = predict(text, model, tokenizer, index_to_category)
    category_predictions = {index_to_category[i]: float(prob) for i, prob in enumerate(probs)}
    recommended_categories = [(category, prob) for category, prob in category_predictions.items() if prob >= threshold]
    return recommended_categories


def calculate_metrics(true_set, predicted_set):
    true_set, predicted_set = set(true_set), set(predicted_set)
    tp = len(true_set & predicted_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    precision = tp / len(predicted_set) if predicted_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1, fp, fn


def average_metrics(metrics_list):
    avg_precision = sum(m[0] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_recall = sum(m[1] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_f_score = sum(m[2] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_fp = sum(m[3] for m in metrics_list) / len(metrics_list)
    avg_fn = sum(m[4] for m in metrics_list) / len(metrics_list)
    return avg_precision, avg_recall, avg_f_score, avg_fp, avg_fn


# ====== ここからテスト部分 ======

MODEL_DIR = legacy_path("/home/nakagawa/study_category_inc_negative_1_10model_only_train")
TEST_DIR = legacy_path("/home/nakagawa/test_by_category_inc_negative_1_10")
API_CATEGORY_DIR = legacy_path("/home/nakagawa/api_category_list/category_inc_negative_1_10_only_train")
MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
results_summary = {}

for model_file in sorted(os.listdir(MODEL_DIR)):
    if not model_file.endswith(".pth"):
        continue

    category_name = model_file.replace("bert_multilabel_", "").replace("_model_weights.pth", "")
    test_file = os.path.join(TEST_DIR, f"study_mashup_data_category_split_inc_negative_{category_name}.json")
    label_file = os.path.join(API_CATEGORY_DIR, f"{category_name}_api_labels.json")

    if not os.path.exists(test_file):
        print(f"[WARN] No test file found for {category_name}")
        continue
    if not os.path.exists(label_file):
        print(f"[WARN] No label file found for {category_name}")
        continue

    print(f"\n=== Evaluating category: {category_name} ===")

    # ラベルリストを読み込む（学習時と同じ）
    with open(label_file, "r") as f:
        all_labels = json.load(f)
    category_to_index = {cat: i for i, cat in enumerate(all_labels)}
    index_to_category = {i: cat for cat, i in category_to_index.items()}
    num_labels = len(all_labels)

    # モデル読み込み
    model_path = os.path.join(MODEL_DIR, model_file)
    model = load_model(BertForMultilabelClassification, MODEL_NAME, num_labels, path=model_path)

    # テストデータ読み込み
    with open(test_file, "r") as f:
        data = json.load(f)
    items = data.get("items", [])

    if not items:
        print(f"[WARN] No items in test file: {test_file}")
        continue

    metrics_list = []

    for item in items:
        text = item.get("description", "")
        true_labels = item.get("used_apis", [])
        if not text or not true_labels:
            continue

        results = run_model_and_get_results(text, model, tokenizer, index_to_category)
        predicted_labels = [r[0] for r in results]

        precision, recall, f1, fp, fn = calculate_metrics(true_labels, predicted_labels)
        metrics_list.append((precision, recall, f1, fp, fn))

    avg_p, avg_r, avg_f1, avg_fp, avg_fn = average_metrics(metrics_list)
    results_summary[category_name] = {
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f1,
        "false_positive": avg_fp,
        "false_negative": avg_fn,
    }

# ====== 集計結果出力 ======
print("\n=== 各カテゴリ別 平均結果 ===")
for cat, res in results_summary.items():
    print(f"{cat:25s} | P: {res['precision']:.3f} | R: {res['recall']:.3f} | F1: {res['f1']:.3f}")

# 結果をJSONで保存
output_path = legacy_path("/home/nakagawa/eval_results_summary_inc_negative_1_10.json")
with open(output_path, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✅ 結果を保存しました: {output_path}")
