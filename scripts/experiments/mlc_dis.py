import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, precision_recall_fscore_support, roc_auc_score
import pytorch_lightning as pl

# Load evaluation data from 'mushup_data_new.json'
with open('mashup_data_new_test_data.json', 'r', encoding='utf-8') as f:
    mushup_data = json.load(f)

# Extract evaluation texts and labels from mushup_data_new.json
evaluation_texts = []
evaluation_labels = []
all_categories = set()

# Extract categories and descriptions for evaluation
for api_name, api_info in list(mushup_data.items())[:500]:  # Take the first 500 entries
    categories = api_info[0]  # Extract categories as labels
    description = api_info[1]  # Extract description as the text
    # 数字以外のカテゴリを無視する
    filtered_categories = [category for category in categories if isinstance(category, int)]
    
    # 数字のカテゴリがあれば処理を続行
    if filtered_categories:
        evaluation_texts.append(description)
        evaluation_labels.append(filtered_categories)
        all_categories.update(filtered_categories)

print(len(evaluation_labels))

# Create category to index mappings
category_to_index = {category: idx for idx, category in enumerate(all_categories)}
index_to_category = {idx: category for category, idx in category_to_index.items()}
number_of_labels = len(all_categories)
print(number_of_labels)

# Convert evaluation labels to binary vectors
evaluation_label_indices = []
for label in evaluation_labels:
    indices = [category_to_index[category] for category in label if category in category_to_index]
    label_vector = [0] * number_of_labels
    for idx in indices:
        label_vector[idx] = 1
    evaluation_label_indices.append(label_vector)

# Define a dataset class for evaluation
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels  
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Define the BERT model for multilabel classification
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Pooled output for classification
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss)
        return {'val_loss': loss, 'logits': logits, 'labels': labels}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# Function to load the saved model
def load_model(model_class, model_name, num_labels, path="10category_model_weights.pth"):
    model = model_class(model_name, num_labels)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    return model

def evaluate_model(model, dataloader, tokenizer, index_to_category, threshold=0.5, save_path="detailed_predictions.json"):
    model.eval()
    all_labels = []
    all_preds = []
    detailed_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_labels.extend(labels)
            all_preds.extend(preds)

            # Collect detailed prediction data for saving
            for idx in range(len(labels)):
                true_label = labels[idx]
                predicted_label = preds[idx]
                prob = probs[idx]
                
                # Prepare the detailed prediction result for each text
                true_categories = [index_to_category[i] for i, label in enumerate(true_label) if label == 1]
                predicted_categories = [index_to_category[i] for i, pred in enumerate(predicted_label) if pred == 1]

                # Compare true and predicted labels
                label_status = {}
                for category, true_val in zip(index_to_category.values(), true_label):
                    label_status[category] = {
                        "true": int(true_val),
                        "predicted": int(predicted_label[category_to_index[category]]),
                        "correct": int(true_val) == int(predicted_label[category_to_index[category]])
                    }

                # Add to detailed predictions
                detailed_predictions.append({
                    "text": evaluation_texts[batch_idx * dataloader.batch_size + idx],  # Retrieve the original text
                    "true_categories": true_categories,
                    "predicted_categories": predicted_categories,
                    "label_status": label_status,
                    "probabilities": {index_to_category[i]: float(prob[i]) for i in range(len(prob))}
                })

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Save detailed predictions to a JSON file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_predictions, f, ensure_ascii=False, indent=4)

    # Compute evaluation metrics (the previous code remains the same)
    valid_labels = (all_labels.sum(axis=0) > 0) | (all_preds.sum(axis=0) > 0)
    all_labels = all_labels[:, valid_labels]
    all_preds = all_preds[:, valid_labels]

    if all_labels.shape[1] == 0:
        print("No valid labels remain after filtering.")
        return {
            "hamming_loss": float('nan'),
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "f1_score": float('nan'),
            "roc_auc": float('nan')
        }

    roc_auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            roc_auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            roc_auc_scores.append(roc_auc)
        else:
            roc_auc_scores.append(float('nan'))

    roc_auc = np.nanmean(roc_auc_scores)
    h_loss = hamming_loss(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Hamming Loss: {h_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "hamming_loss": h_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# Main evaluation function
def main_evaluation():
    # Load the tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create the evaluation dataset and dataloader
    evaluation_dataset = TextDataset(evaluation_texts, evaluation_label_indices, tokenizer)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=2, num_workers=8)

    # Load the saved model
    model = load_model(BertForMultilabelClassification, model_name, number_of_labels, path="10category_model_weights.pth")

    # Evaluate the model and compute metrics
    metrics = evaluate_model(model, evaluation_loader, tokenizer, index_to_category)

    # Print evaluation metrics
    print(metrics)

if __name__ == "__main__":
    main_evaluation()
