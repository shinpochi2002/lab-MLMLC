import json
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, AdamW
from pytorch_lightning.callbacks import EarlyStopping


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        encoding = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        self.input_ids = encoding["input_ids"]
        self.attention_masks = encoding["attention_mask"]
        self.labels = [torch.tensor(label, dtype=torch.float) for label in labels]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': self.labels[idx]
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

BASE_ROOT = "/home/nakagawa/lab-MLMLC"
TARGET_PATTERN = "8_3_2_3"
LAYER1_NUM = "8"

def process_file(a, b, c):
    base_path = f"{BASE_ROOT}/studydata/layer4/layer4_{TARGET_PATTERN}"
    input_file = os.path.join(
        base_path,
        f"softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_{TARGET_PATTERN}_{LAYER1_NUM}_{a}_{b}_{c}_study_data.json",
    )

    model_dir = f"{BASE_ROOT}/layer4_{TARGET_PATTERN}"
    os.makedirs(model_dir, exist_ok=True)
    output_model = f"{model_dir}/mashup_softfill_api_all_layer4_{LAYER1_NUM}_{a}_{b}_{c}_model_weights.pth"

    api_dir = f"{BASE_ROOT}/api_category_list/layer4_{TARGET_PATTERN}"
    os.makedirs(api_dir, exist_ok=True)
    category_list_output = f"{api_dir}/softfill_inc_categorylist_layer4_{LAYER1_NUM}_{a}_{b}_{c}.json"
    
    
    if not os.path.exists(input_file):
        print(f"[{a}-{b}-{c}] ファイルが存在しません。スキップします。")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    texts = [item["description"] for item in api_data]
    raw_labels = [item["categories"] for item in api_data]

    all_categories = sorted(set(cat for sublist in raw_labels for cat in sublist))
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}

    labels = []
    for cats in raw_labels:
        label_vector = [0] * len(all_categories)
        for cat in cats:
            label_vector[category_to_index[cat]] = 1
        labels.append(label_vector)

    positive_counts = torch.tensor([sum(l[i] for l in labels) for i in range(len(all_categories))])
    negative_counts = torch.tensor([len(labels) - p for p in positive_counts])
    pos_weight = negative_counts / (positive_counts + 1e-6)

    split = int(0.8 * len(texts))
    train_dataset = TextDataset(texts[:split], labels[:split], tokenizer)
    val_dataset = TextDataset(texts[split:], labels[split:], tokenizer)

    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": 8,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = BertForMultilabelClassification("bert-base-uncased", num_labels=len(all_categories), pos_weight=pos_weight)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)
    save_model(model, output_model)

    with open(category_list_output, "w", encoding="utf-8") as f:
        json.dump(all_categories, f, ensure_ascii=False, indent=4)

    print(f"[{a}-{b}-{c}] モデル学習と保存完了: {output_model}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    max_a = 13
    max_b = 4
    max_c = 4

    for a in range(max_a + 1):
        for b in range(max_b + 1):
            for c in range(max_c + 1):
                process_file(a, b, c)
