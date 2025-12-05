import os
import json
import random
import re
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset
import hashlib  
import numpy as np 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# ====== パス設定（必要に応じて変更） ======
ROOT_DIR = "/home/nakagawa/out_by_category_inc_nomashup_negative_1_3"  # カテゴリ別のstudydataが入っているディレクトリ
LABEL_OUT_DIR = "/home/nakagawa/api_category_list/category_inc_nomashup_negative_1_3_new_kfold"  # APIラベル一覧の出力先
MODEL_OUT_DIR = "/home/nakagawa/study_category_inc_nomashup_negative_1_3_model_new_kfold"            # 学習済みモデルの出力先
CV_OUT_DIR = "/home/nakagawa/study_category_inc_nomashup_negative_1_3_cv"
os.makedirs(CV_OUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ---（少データでKFoldに切替える閾値など）
K_FOLD_TRIGGER_N = 20     # サンプルがこれ未満ならKFoldに切替
K_FOLDS = 5               # KFoldの分割数（自動で min(K_FOLDS, n, >=2) に調整）
LR_CANDIDATES = (2e-5, 3e-5)  # KFoldで試す学習率の候補
BATCH_SIZE = 8
MAX_LENGTH = 512
ES_PATIENCE = 3
RANDOM_SEED = 42

# ====== ユーティリティ ======
def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-一-龯ぁ-んァ-ン]", "_", s)
    return s[:120]

def category_uid(cat_json_path: str) -> str:  # --- NEW ---
    return hashlib.md5(os.path.basename(cat_json_path).encode()).hexdigest()[:8]

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


# ====== 前処理：JSON → (category, texts, labels(one-hot), api_list) ======  # --- NEW ---
def load_category(cat_json_path: str):
    with open(cat_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category = data.get("category", "unknown")
    count = int(data.get("count", 0))
    items = data.get("items", [])

    texts = []
    labels_apis = []
    all_apis = set()
    for it in items:
        desc = (it.get("description", "") or "").strip()
        apis = it.get("used_apis", []) or []
        if not desc or not apis:
            continue
        texts.append(desc)
        labels_apis.append(list(apis))
        all_apis.update(apis)

    api_list = sorted(all_apis)
    api_to_index = {api: i for i, api in enumerate(api_list)}
    labels = []
    for apis in labels_apis:
        vec = [0] * len(api_list)
        for a in apis:
            idx = api_to_index.get(a)
            if idx is not None:
                vec[idx] = 1
        labels.append(vec)

    return category, count, items, texts, labels, api_list 


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

# ====== pos_weight ヘルパ ======  # --- NEW ---
def compute_pos_weight_from_labels(label_list, max_clip=100.0):
    if not label_list:
        return None
    arr = np.array(label_list, dtype=np.int64)
    pos = torch.tensor(arr.sum(axis=0), dtype=torch.float)
    neg = torch.tensor(len(label_list), dtype=torch.float) - pos
    pw = torch.clamp(neg / (pos + 1e-6), max=max_clip)
    return pw

# ====== 通常ルート（8:2） ======  # --- NEW（安定化済み） ---
def train_holdout(category, texts, labels, api_list, tokenizer_name="bert-base-uncased"):
    num_labels = len(api_list)
    train_texts, val_texts, train_labels, val_labels = split_with_all_labels(
        texts, labels, api_list, train_ratio=0.8
    )

    # pos_weightはtrainで再計算
    pos_weight = compute_pos_weight_from_labels(train_labels)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length=MAX_LENGTH)
    val_ds   = TextDataset(val_texts,  val_labels,   tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=0)

    print(f"[TRAIN] {category}: train={len(train_ds)} val={len(val_ds)} labels={num_labels}")

    model = BertForMultilabelClassification(
        model_name=tokenizer_name, num_labels=num_labels,
        pos_weight=pos_weight, lr=2e-5
    )
    
    if len(val_ds) == 0:
        # 検証データがない場合：EarlyStoppingなし・val_loaderなしで学習
        print(f"[WARN] {category}: val dataset is empty. Training without validation / early stopping.")
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            log_every_n_steps=10,
        )
        trainer.fit(model, train_loader)  # val_loader を渡さない
    else:
        # 検証データがある場合：今まで通り EarlyStopping を使う
        early_stop = EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, mode="min", verbose=True)
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

    return model
# ====== K-Fold → ベスト設定選択 → 全データ最終学習 ======  # --- NEW ---
def train_kfold_then_final(cat_json_path, category, texts, labels, api_list, tokenizer_name="bert-base-uncased"):
    uid = category_uid(cat_json_path)
    num_labels = len(api_list)
    n = len(texts)

    print(f"[KFold] {category}: n={n} labels={num_labels}")

    # Y: (N, L) 0/1行列
    Y = np.array(labels, dtype=np.int64)
    k = max(2, min(K_FOLDS, n))  # 安全に調整
    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # 学習率候補からベストを選ぶ（val_loss最小、同値ならF1高い）
    results = []  # (lr, mean_val_loss, std_val_loss, mean_micro_f1)
    for lr in LR_CANDIDATES:
        fold_val_losses, fold_f1s = [], []
        for fold, (tr_idx, va_idx) in enumerate(mskf.split(np.arange(n), Y), 1):
            tr_texts = [texts[i] for i in tr_idx]
            va_texts = [texts[i] for i in va_idx]
            tr_labels = [labels[i] for i in tr_idx]
            va_labels = [labels[i] for i in va_idx]

            pos_weight = compute_pos_weight_from_labels(tr_labels)

            train_ds = TextDataset(tr_texts, tr_labels, tokenizer, max_length=MAX_LENGTH)
            val_ds   = TextDataset(va_texts,  va_labels,   tokenizer, max_length=MAX_LENGTH)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=0)

            model = BertForMultilabelClassification(
                model_name=tokenizer_name, num_labels=num_labels,
                pos_weight=pos_weight, lr=lr
            )
            callbacks = [EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, mode="min", verbose=False)]
            trainer = pl.Trainer(
                max_epochs=50, accelerator="gpu", devices=1,
                callbacks=callbacks, logger=False, enable_checkpointing=False,
                log_every_n_steps=10, num_sanity_val_steps=2
            )
            print(f"[CV FOLD {fold}/{k}] {category}: train={len(train_ds)} val={len(val_ds)} lr={lr}")
            trainer.fit(model, train_loader, val_loader)

            vloss = float(trainer.callback_metrics.get("val_loss", torch.tensor(float('nan'))))
            fold_val_losses.append(vloss)

            # 簡易micro-F1(th=0.5)
            model.eval()
            with torch.no_grad():
                all_logits, all_true = [], []
                for batch in val_loader:
                    logits = model(batch["input_ids"].to(model.device), batch["attention_mask"].to(model.device))
                    all_logits.append(torch.sigmoid(logits).cpu())
                    all_true.append(batch["labels"])
                if all_logits:
                    pred = (torch.vstack(all_logits) >= 0.5)
                    true = torch.vstack(all_true).bool()
                    tp = (pred & true).sum().item()
                    fp = (pred & (~true)).sum().item()
                    fn = ((~pred) & true).sum().item()
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall    = tp / (tp + fn) if (tp + fn) else 0.0
                    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
                else:
                    f1 = float("nan")
            fold_f1s.append(f1)

        mean_v = float(np.nanmean(fold_val_losses))
        std_v  = float(np.nanstd(fold_val_losses))
        mean_f = float(np.nanmean(fold_f1s))
        results.append((lr, mean_v, std_v, mean_f))
        print(f"[CV] {category}: lr={lr}  val_loss mean={mean_v:.4f}±{std_v:.4f}  micro-F1 mean={mean_f:.4f}")

    results.sort(key=lambda x: (x[1], -x[3]))  # val_loss小→同値はF1高
    best_lr, best_mean_v, best_std_v, best_mean_f1 = results[0]

    # CV結果を保存
    cv_out = {
        "category": category, "uid": uid,
        "n_samples": n, "n_labels": num_labels,
        "results": [
            {"lr": lr, "mean_val_loss": mv, "std_val_loss": sv, "mean_micro_f1": mf}
            for (lr, mv, sv, mf) in results
        ],
        "selected": {
            "lr": best_lr,
            "mean_val_loss": best_mean_v,
            "std_val_loss": best_std_v,
            "mean_micro_f1": best_mean_f1
        }
    }
    cv_path = os.path.join(CV_OUT_DIR, f"cv_{slugify(category)}_{uid}.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_out, f, ensure_ascii=False, indent=2)
    print(f"[CV SAVED] {cv_path}")

    # ---- 配布用：全データで最終学習 ----
    print(f"[FINAL TRAIN] {category}: n={n} labels={num_labels} lr={best_lr}")
    pos_weight_all = compute_pos_weight_from_labels(labels)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    ds = TextDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = BertForMultilabelClassification(
        model_name=tokenizer_name, num_labels=num_labels,
        pos_weight=pos_weight_all, lr=best_lr
    )
    trainer = pl.Trainer(
        max_epochs=50, accelerator="gpu", devices=1,
        logger=False, enable_checkpointing=False,
        log_every_n_steps=10, num_sanity_val_steps=0
    )
    trainer.fit(model, loader)

    # 保存（uid付きで衝突回避）
    cat_slug = slugify(category)
    label_path = os.path.join(LABEL_OUT_DIR, f"{cat_slug}_api_labels.json")
    model_path = os.path.join(MODEL_OUT_DIR, f"bert_multilabel_{cat_slug}_model_weights.pth")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(api_list, f, ensure_ascii=False, indent=2)
    save_model(model, model_path)
    print(f"[FINAL DONE] {category}: model -> {model_path}")
    print(f"[FINAL DONE] {category}: labels -> {label_path}")

# ====== 自動ルータ：少データならKFold、そうでなければ8:2 ======  # --- NEW ---
def train_one_category_auto(cat_json_path: str, tokenizer_name="bert-base-uncased"):
    category, count, items, texts, labels, api_list = load_category(cat_json_path)

    # ラベル定義が空/テキスト0件ならスキップ（旧仕様踏襲）
    if len(items) == 0:
        print(f"[SKIP] {category}: count={count}")
        cat_slug = slugify(category)
        label_path = os.path.join(LABEL_OUT_DIR, f"{cat_slug}_api_labels.json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(api_list, f, ensure_ascii=False, indent=2)
        return
    if len(texts) == 0 or len(api_list) == 0:
        print(f"[SKIP] {category}: 有効データ不足 texts={len(texts)} apis={len(api_list)}")
        return
    
    num_labels = len(api_list)
    
    if len(texts) < K_FOLD_TRIGGER_N and num_labels > 1:
        # 少データ：KFold評価→全データ最終学習（配布用）
        train_kfold_then_final(cat_json_path, category, texts, labels, api_list, tokenizer_name)
    else:
        # 通常：8:2で学習（val≥1ガード入り）
        model = train_holdout(category, texts, labels, api_list, tokenizer_name)
        # 保存（旧命名互換）
        cat_slug = slugify(category)
        label_path = os.path.join(LABEL_OUT_DIR, f"{cat_slug}_api_labels.json")
        model_path = os.path.join(MODEL_OUT_DIR, f"bert_multilabel_{cat_slug}_model_weights.pth")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(api_list, f, ensure_ascii=False, indent=2)
        save_model(model, model_path)
        print(f"[DONE] {category}: model -> {model_path}")
        print(f"[DONE] {category}: labels -> {label_path}")



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
        if not desc.strip():
            continue
        texts.append(desc.strip())
        labels_apis.append(list(apis))
        all_apis.update(apis)

    # ラベル空間：このカテゴリ内で出現した全API
    api_list = list(all_apis)
    api_to_index = {api: i for i, api in enumerate(api_list)}
    num_labels = len(api_list)
    
    cat_slug = slugify(category)
    if len(items) == 0:
        print(f"[SKIP] {category}: count={count}")
        label_path = os.path.join(LABEL_OUT_DIR, f"{cat_slug}_api_labels.json")
        # ラベルマップ保存（API名一覧）
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(api_list, f, ensure_ascii=False, indent=2)
        return
    
    if len(texts) == 0 or len(all_apis) == 0:
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

    # 8:2 split（学習データに全ラベルを含めるように分割）
    train_texts, val_texts, train_labels, val_labels = split_with_all_labels(
        texts, label_indices, api_list, train_ratio=0.8
    )
    
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
        lr=2e-5
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
            train_one_category_auto(fp, tokenizer_name="bert-base-uncased")
        except Exception as e:
            import traceback
            print(f"[ERROR] {fp}: {type(e).__name__}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()