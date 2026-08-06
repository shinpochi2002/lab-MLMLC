import json
import os
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, Dataset



with open('/home/nakagawa/softfill_mashup_data_new_api_inclu_nomash/8category/softfill_mashup_data_new_api_8category_inclu_nomash_category0ver_study_data.json', 'r', encoding='utf-8') as f:
    api_data = json.load(f)

# テキストとラベルの抽出
texts = []
labels = []

# カテゴリとAPIをラベルとして処理
all_categories = set()

# api_dataの処理
for api_name, api_info in api_data.items():
    apis = api_info[0]  # カテゴリ情報を取得
    description = api_info[1]
    texts.append(description)
    labels.append(apis)
    all_categories.update(apis)

# データ数の確認
print(f"Number of texts after extraction: {len(texts)}")
print(f"Number of labels after extraction: {len(labels)}")

# カテゴリをインデックスにマッピング
category_to_index = {category: idx for idx, category in enumerate(all_categories)}
index_to_category = {idx: category for category, idx in category_to_index.items()}

# カテゴリのインデックス確認
print("Category to Index Mapping:")
print(f"Number of categories: {len(category_to_index)}")

# ラベルをインデックス形式に変換
label_indices = []
for label in labels:
    indices = [category_to_index[category] for category in label]
    label_vector = [0] * len(all_categories)
    for idx in indices:
        label_vector[idx] = 1
    label_indices.append(label_vector)

number_of_labels = len(all_categories)

# データセットクラスの定義
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

# データセットの作成
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_texts = texts[:int(0.8*len(texts))]
train_label_indices = label_indices[:int(0.8*len(label_indices))]
val_texts = texts[int(0.8*len(texts)):]
val_label_indices = label_indices[int(0.8*len(label_indices)):]

train_dataset = TextDataset(train_texts, train_label_indices, tokenizer)
val_dataset = TextDataset(val_texts, val_label_indices, tokenizer)

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(val_dataset)}")

# PyTorch Lightningのモジュール定義
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):  # 学習率を調整
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # 損失関数をBCEWithLogitsLossに設定

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # get the pooled output from the second element of the tuple
        logits = self.classifier(pooled_output)  # apply the classifier layer
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)  # 損失を計算
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)  # 損失を計算
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        return {'val_loss': loss, 'logits': logits, 'labels': labels}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)
        print(f"Training Loss after epoch end: {avg_loss.item()}")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        print(f"Validation Loss after epoch end: {avg_loss.item()}")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

def predict(text, model, tokenizer, index_to_category, threshold=0.5):
    # テキストのエンコード
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].flatten().unsqueeze(0)
    attention_mask = encoding['attention_mask'].flatten().unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()  # シグモイド関数を適用して予測を行う
    
    recommended_categories = []
    for idx, prob in enumerate(probs):
        category = index_to_category[idx]
        recommended_categories.append((category, prob))
    recommended_categories.sort(key=lambda x: x[1], reverse=True)
    
    return recommended_categories, probs

def save_model(model, path="mashup_softfill_api_8category_category0_model_weights.pth"):
    torch.save(model.state_dict(), path)


def main1():
    # GPUを指定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # トークナイザーとモデルのロード
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # データセットをトレーニングと検証に分割
    train_texts = texts[:int(0.8*len(texts))]
    train_label_indices = label_indices[:int(0.8*len(label_indices))]
    val_texts = texts[int(0.8*len(texts)):]
    val_label_indices = label_indices[int(0.8*len(label_indices)):]

    # データセットの作成
    train_dataset = TextDataset(train_texts, train_label_indices, tokenizer)
    val_dataset = TextDataset(val_texts, val_label_indices, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    print(f"Number of training examples: {len(train_loader.dataset)}")
    print(f"Number of validation examples: {len(val_loader.dataset)}")

    # モデルのインスタンス作成
    model = BertForMultilabelClassification(model_name, num_labels=number_of_labels)
    
    # トレーナーの定義
    trainer = pl.Trainer(max_epochs=10, gpus=1)

    # トレーニングの実行
    trainer.fit(model, train_loader, val_loader)

    # モデルを保存
    save_model(model)



if __name__ == "__main__": 
    main1()  # 次に保存されたモデルをロードして評価