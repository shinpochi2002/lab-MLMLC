import json
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, Dataset




from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

with open(legacy_path('/home/nakagawa/studydata/layer2/layer1_6/softfill_mashup_data_new_api_category_inclu_nomash_study_data_2layer_6_study_data.json'), 'r', encoding='utf-8') as f:
    api_data = json.load(f)

# テキストとラベルの抽出
texts = []
labels = []

# カテゴリとAPIをラベルとして処理
all_categories = set()

# api_dataの処理
for api_name,api_info in api_data.items():
    category = api_info[2]  # カテゴリ情報を取得
    description = api_info[1]
    texts.append(description)
    labels.append(category)
    all_categories.update(category)


# データ数の確認
print(f"Number of texts after extraction: {len(texts)}")
print(f"Number of labels after extraction: {len(labels)}")
print(all_categories)

# APIリストをJSON形式で保存
with open(legacy_path("/home/nakagawa/api_category_list/layer2_6/oftfill_inc_categorylist_layer2_6.json"), 'w', encoding='utf-8') as file:
    json.dump(list(all_categories), file, ensure_ascii=False, indent=4)

# カテゴリをインデックスにマッピング
category_to_index = {category: idx for idx, category in enumerate(all_categories)}
index_to_category = {idx: category for category, idx in category_to_index.items()}

# カテゴリのインデックス確認
print("Category to Index Mapping:")
print(f"Number of categories: {len(category_to_index)}")
print(category_to_index)

# ラベルをインデックス形式に変換
label_indices = []
for label in labels:
    # 単一カテゴリの場合の処理
    indices = [category_to_index[category] for category in label]
    label_vector = [0] * len(all_categories)
    for idx in indices:
        label_vector[idx] = 1
    label_indices.append(label_vector)

number_of_labels = len(all_categories)

# pos_weight の計算
positive_counts = torch.tensor([sum(l[i] for l in label_indices) for i in range(number_of_labels)])
negative_counts = torch.tensor([len(label_indices) - p for p in positive_counts])
pos_weight = negative_counts / (positive_counts + 1e-6)


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
        self.loss_fn = (
            torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if pos_weight is not None else
            torch.nn.BCEWithLogitsLoss()
        )
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # get the pooled output from the second element of the tuple
        logits = self.classifier(pooled_output)  # apply the classifier layer
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)  # 損失を計算
        self.train_losses.append(loss.detach())
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(logits, labels)  # 損失を計算
        self.val_losses.append(loss.detach())
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        return {'val_loss': loss, 'logits': logits, 'labels': labels}

    def on_train_epoch_end(self):
        if self.train_losses:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log('avg_train_loss', avg_loss)
            print(f"Training Loss after epoch end: {avg_loss.item()}")
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = torch.stack(self.val_losses).mean()
            self.log('avg_val_loss', avg_loss)
            print(f"Validation Loss after epoch end: {avg_loss.item()}")
            self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}


def save_model(model, path=legacy_path("/home/nakagawa/layer2_6/mashup_softfill_api_all_layer2_6_model_weights.pth")):
    
    # state_dict を取得
    state_dict = model.state_dict()
    
    # 不要なキーを削除
    state_dict = {k: v for k, v in state_dict.items() if "loss_fn" not in k}

    torch.save(state_dict, path)


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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=8)

    print(f"Number of training examples: {len(train_loader.dataset)}")
    print(f"Number of validation examples: {len(val_loader.dataset)}")

    # モデルのインスタンス作成
    model = BertForMultilabelClassification(model_name, num_labels=number_of_labels)
    
    # **EarlyStoppingの定義**
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # モニターする値（検証損失）
        patience=3,          # 改善しないエポック数の許容（例: 3エポック）
        verbose=True,        # ログ出力
        mode="min"           # 損失なので「最小化」を目指す
    )
    
    # トレーナーの定義
    trainer = pl.Trainer(
        max_epochs=50,       # エポック数は大きめに設定（早期終了があるので）
        accelerator="gpu",           # GPU を使う明示的指定
        devices=1,
        callbacks=[early_stop_callback]  # EarlyStoppingをコールバックとして追加
    )

    # トレーニングの実行
    trainer.fit(model, train_loader, val_loader)

    # モデルを保存
    save_model(model)



if __name__ == "__main__": 
    main1()  # 次に保存されたモデルをロードして評価