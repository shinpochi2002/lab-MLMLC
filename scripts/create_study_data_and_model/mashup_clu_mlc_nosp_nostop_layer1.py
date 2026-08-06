import json
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from pytorch_lightning.callbacks import EarlyStopping


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import PROJECT_ROOT, legacy_path

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


class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, pos_weight, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        self.train_losses.append(loss.detach())
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        self.val_losses.append(loss.detach())
        self.log('val_loss', loss, on_epoch=True)
        return {'val_loss': loss}

    def on_train_epoch_end(self):
        if self.train_losses:
            self.log('avg_train_loss', torch.stack(self.train_losses).mean())
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        if self.val_losses:
            self.log('avg_val_loss', torch.stack(self.val_losses).mean())
            self.val_losses.clear()

    def configure_optimizers(self):
        return {"optimizer": AdamW(self.parameters(), lr=self.lr)}

# ====== モデル保存（ファイル名にカテゴリ名を含める） ======
def save_model(model: pl.LightningModule, out_path: str):
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if "loss_fn" not in k}
    torch.save(state_dict, out_path)

def process_file(index):
    
    base_path = str(PROJECT_ROOT)
    input_file = os.path.join(base_path, legacy_path(f"studydata/layer4/layer4_8_3_2_3/softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_8_3_2_3_study_data.json"))
    output_model = os.makedirs(legacy_path(f"/home/nakagawa/lab-MLMLC/layer4_8_3_2_3"), exist_ok=True) or legacy_path(f"/home/nakagawa/lab-MLMLC/layer4_8_3_2_3/mashup_softfill_api_all_layer4_model_weights.pth")
    category_list_output = os.makedirs(legacy_path(f"/home/nakagawa/lab-MLMLC/api_category_list/layer4_8_3_2_3"), exist_ok=True) or legacy_path(f"/home/nakagawa/lab-MLMLC/api_category_list/layer4_8_3_2_3/softfill_inc_categorylist_layer4_8.json")
    target = "bigcategories"
    
    if not os.path.exists(input_file):
        print(f"[{index}] ファイルが存在しません。スキップします。")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)
    
    texts = [item["description"] for item in api_data]
    raw_labels = [item[target] for item in api_data]

    all_categories = sorted(set(cat for sublist in raw_labels for cat in sublist))
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}

    labels = []
    for cats in raw_labels:
        label_vector = [0] * len(all_categories)
        for cat in cats:
            label_vector[category_to_index[cat]] = 1
        labels.append(label_vector)

    # pos_weight計算
    positive_counts = torch.tensor([sum(l[i] for l in labels) for i in range(len(all_categories))])
    negative_counts = torch.tensor([len(labels) - p for p in positive_counts])
    pos_weight = negative_counts / (positive_counts + 1e-6)

    # データ分割
    split = int(0.8 * len(texts))
    train_dataset = TextDataset(texts[:split], labels[:split], tokenizer)
    val_dataset = TextDataset(texts[split:], labels[split:], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)

    model = BertForMultilabelClassification("bert-base-uncased", num_labels=len(all_categories), pos_weight=pos_weight)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, train_loader, val_loader)
    save_model(model, output_model)

    with open(category_list_output, "w", encoding="utf-8") as f:
        json.dump(all_categories, f, ensure_ascii=False, indent=4)

    print(f"[{index}] モデル学習と保存完了: {output_model}")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for idx in range(0, 1):  
        process_file(idx)
