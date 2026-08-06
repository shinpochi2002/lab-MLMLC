import json
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# JSONファイルのロード
with open('softfill_mashup_data_new_api_4category_inclu_nomash_category2ver_study_data.json', 'r', encoding='utf-8') as f:
    api_data = json.load(f)

# テキストとラベルの抽出
texts = []
labels = []
all_categories = set()

for api_name, api_info in api_data.items():
    apis = api_info[0]
    description = api_info[1]
    texts.append(description)
    labels.append(apis)
    all_categories.update(apis)

# カテゴリをインデックスにマッピング
category_to_index = {category: idx for idx, category in enumerate(all_categories)}
index_to_category = {idx: category for category, idx in category_to_index.items()}

# ラベルをインデックス形式に変換
label_indices = []
for label in labels:
    indices = [category_to_index[category] for category in label if category in category_to_index]
    label_vector = [0] * len(all_categories)
    for idx in indices:
        label_vector[idx] = 1
    label_indices.append(label_vector)

number_of_labels = len(all_categories)

# データセットクラス
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

# トークナイザーの作成
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# データセットの作成
train_texts = texts[:int(0.8*len(texts))]
train_label_indices = label_indices[:int(0.8*len(label_indices))]
val_texts = texts[int(0.8*len(texts)):]
val_label_indices = label_indices[int(0.8*len(label_indices)):]

train_dataset = TextDataset(train_texts, train_label_indices, tokenizer)
val_dataset = TextDataset(val_texts, val_label_indices, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=8,num_workers=8)

# PyTorch Lightning モジュール
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(num_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss, on_epoch=True)
        return {'val_loss': loss, 'logits': logits, 'labels': labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', val_loss)

        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        targets = labels.cpu().numpy()

        acc = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='micro')

        self.log("val_acc", acc)
        self.log("val_f1", f1)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

# モデルの保存
def save_model(model, path):
    torch.save(model.state_dict(),path)


# バッチサイズの自動調整
def main():
    trainer = pl.Trainer(max_epochs=10, gpus=1, auto_scale_batch_size=True)
    model = BertForMultilabelClassification(model_name, num_labels=number_of_labels)
    trainer.fit(model, train_loader, val_loader)
    save_model(model, "new_mashup_softfill_api_4category_category2_inc_nomash_model_weights.pth")  # モデル保存

if __name__ == "__main__":
    main()
