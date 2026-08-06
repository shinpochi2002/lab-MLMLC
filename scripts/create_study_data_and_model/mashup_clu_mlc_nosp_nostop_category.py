import os
import json
import re
import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# ====== パス設定（必要に応じて変更） ======
ROOT_DIR = legacy_path("/home/nakagawa/out_by_category_inc_nomashup_negative_1_3")  # カテゴリ別のstudydataが入っているディレクトリ
LABEL_OUT_DIR = legacy_path("/home/nakagawa/api_category_list/category_inc_nomashup_negative_1_3_new")  # APIラベル一覧の出力先
MODEL_OUT_DIR = legacy_path("/home/nakagawa/study_category_inc_nomashup_negative_1_3_model_new")            # 学習済みモデルの出力先
os.makedirs(LABEL_OUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# ====== ユーティリティ ======
def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-一-龯ぁ-んァ-ン]", "_", s)
    return s[:120]

# ====== Dataset ======
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
        label_vec = self.labels[idx]
        enc = self.tokenizer.encode_plus(
            text if isinstance(text, str) else "",
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(label_vec, dtype=torch.float),
        }

# ====== Lightning Module ======
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, pos_weight=None, lr=2e-5):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])  # 軽量に
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
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

# ====== モデル保存（ファイル名にカテゴリ名を含める） ======
def save_model(model: pl.LightningModule, out_path: str):
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if "loss_fn" not in k}
    torch.save(state_dict, out_path)
# ====== 8:2 split（trainに全ラベルを含める） ======
def split_with_all_labels(texts, label_indices, api_list, train_ratio=0.8, seed=42):
    """
    学習データにすべてのラベル(API)を含めるように分割する
    """
    random.seed(seed)
    n = len(texts)
    indices = list(range(n))
    random.shuffle(indices)

    # まず通常の8:2分割を試みる
    split = int(train_ratio * n)
    train_idx = set(indices[:split])
    val_idx = set(indices[split:])

    # 各ラベルがtrain側に出ているか確認
    num_labels = len(api_list)
    label_in_train = [False] * num_labels

    for i in train_idx:
        for j, v in enumerate(label_indices[i]):
            if v == 1:
                label_in_train[j] = True

    # trainに出ていないラベルを確認
    missing_labels = [j for j, present in enumerate(label_in_train) if not present]

    if missing_labels:
        print(f"[INFO] {len(missing_labels)} ラベルがtrainに未出現。valから移動します。")

        for j in missing_labels:
            # val側でこのラベルを含むサンプルを探す
            candidates = [i for i in val_idx if label_indices[i][j] == 1]
            if not candidates:
                continue
            # 1件をtrainへ移動
            move_idx = random.choice(candidates)
            train_idx.add(move_idx)
            val_idx.remove(move_idx)

    # リスト再構成
    train_texts = [texts[i] for i in train_idx]
    val_texts   = [texts[i] for i in val_idx]
    train_labels = [label_indices[i] for i in train_idx]
    val_labels   = [label_indices[i] for i in val_idx]
    
    # --- NEW: val最低1件ガード（EarlyStopping対策）
    if len(val_texts) == 0 and len(train_texts) >= 2:
        val_texts.append(train_texts[-1]); val_labels.append(train_labels[-1])
        train_texts = train_texts[:-1];     train_labels = train_labels[:-1]

    return train_texts, val_texts, train_labels, val_labels    

# ====== 1カテゴリ分を学習 ======
def train_one_category(cat_json_path: str, tokenizer_name="bert-base-uncased"):
    with open(cat_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category = data.get("category", "unknown")
    count = int(data.get("count", 0))
    items = data.get("items", [])


    # 入力テキスト（description）とマルチラベル（used_apis）を作る
    texts = []
    labels_apis = []
    all_apis = set()

    for it in items:
        desc = it.get("description", "") or ""
        apis = it.get("used_apis", []) or []
        # 空は除外（学習に寄与しないため）
        if not desc.strip() or not apis:
            continue
        texts.append(desc.strip())
        labels_apis.append(list(apis))
        all_apis.update(apis)

    # ラベル空間：このカテゴリ内で出現した全API
    api_list = list(all_apis)
    api_to_index = {api: i for i, api in enumerate(api_list)}
    num_labels = len(api_list)
    
    cat_slug = slugify(category)
    # count==1 はスキップ
    if count <= 1 or len(items) <= 1:
        print(f"[SKIP] {category}: count={count}")
        label_path = os.path.join(LABEL_OUT_DIR, f"{cat_slug}_api_labels.json")
         # ラベルマップ保存（API名一覧）
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(api_list, f, ensure_ascii=False, indent=2)
        return
    
    if len(texts) <= 1 or len(all_apis) == 0:
        print(f"[SKIP] {category}: 有効データ不足 texts={len(texts)} apis={len(all_apis)}")
        return

    # ラベルベクトル化
    label_indices = []
    for apis in labels_apis:
        vec = [0] * num_labels
        for a in apis:
            idx = api_to_index.get(a)
            if idx is not None:
                vec[idx] = 1
        label_indices.append(vec)

    # pos_weight 計算
    positive_counts = torch.tensor(
        [sum(l[i] for l in label_indices) for i in range(num_labels)],
        dtype=torch.float,
    )
    negative_counts = torch.tensor([len(label_indices) - p for p in positive_counts], dtype=torch.float)
    pos_weight = negative_counts / (positive_counts + 1e-6)

    # 8:2 split
    n = len(texts)
    split = int(0.8 * n)
    train_texts = texts[:split]
    val_texts = texts[split:] if split < n else texts[-max(1, n//5):]
    train_labels = label_indices[:split]
    val_labels = label_indices[split:] if split < n else label_indices[-max(1, n//5):]

    # tokenizer / dataset / loader
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds = TextDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=8)

    print(f"[TRAIN] {category}: train={len(train_ds)} val={len(val_ds)} labels={num_labels}")

    # モデル
    model = BertForMultilabelClassification(
        model_name=tokenizer_name,
        num_labels=num_labels,
        pos_weight=pos_weight,
        lr=2e-5,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop],
        log_every_n_steps=10,
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, train_loader, val_loader)

    # 出力ファイル（カテゴリ名入り）
    label_path = os.path.join(LABEL_OUT_DIR, f"{cat_slug}_api_labels.json")
    model_path = os.path.join(MODEL_OUT_DIR, f"bert_multilabel_{cat_slug}_model_weights.pth")

    # ラベルマップ保存（API名一覧）
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(api_list, f, ensure_ascii=False, indent=2)

    # モデル保存
    save_model(model, model_path)

    print(f"[DONE] {category}: model -> {model_path}")
    print(f"[DONE] {category}: labels -> {label_path}")

# ====== ループ実行 ======
def main():
    # GPU 指定（必要なら）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    files = [os.path.join(ROOT_DIR, f) for f in os.listdir(ROOT_DIR) if f.endswith(".json")]
    files.sort()

    if not files:
        print(f"対象JSONが見つかりません: {ROOT_DIR}")
        return

    for fp in files:
        try:
            train_one_category(fp, tokenizer_name="bert-base-uncased")
        except Exception as e:
            print(f"[ERROR] {fp}: {e}")

if __name__ == "__main__":
    main()