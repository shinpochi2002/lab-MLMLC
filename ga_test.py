# ----------------- 基本処理 -----------------
import os
import json
import glob
import re
import random
import numpy as np
from collections import defaultdict
import traceback

# ----------------- Word2Vec & BERT -----------------
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances
from transformers import BertTokenizer, BertModel
import gensim

# ----------------- クラスタリング -----------------
from sklearn.cluster import KMeans

# Balanced KMeans（独自実装）
# ※balanced_kmeans_clustering() は別途実装済み関数を使う

# ----------------- PyTorch Lightning 学習 -----------------
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from pytorch_lightning.callbacks import EarlyStopping

# ----------------- 遺伝的アルゴリズム（GA） -----------------
from deap import base, creator, tools, algorithms

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_individual(min_layers=2, max_layers=4, min_clusters=2, max_clusters=20):
    num_layers = random.randint(min_layers, max_layers)
    individual = [random.randint(min_clusters, max_clusters) for _ in range(num_layers)]
    return individual


def get_category_vectors(categories, model):
    def get_category_vector(category_name):
        words = category_name.split("-")
        word_vectors = [model[word] for word in words if word in model]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    
    return np.array([get_category_vector(category) for category in categories])


def balanced_kmeans_clustering(vectors, num_clusters=10, max_iter=100, random_state=None):
    """
    Balanced k-means clustering with k-means++ initialization.

    Parameters:
        vectors (array-like): Input vectors, shape (n_samples, n_features)
        num_clusters (int): Number of clusters (k)
        max_iter (int): Maximum number of iterations
        random_state (int or None): Random seed for initialization

    Returns:
        labels (array): Cluster labels for each data point, shape (n_samples,)
    """
    np.random.seed(random_state)
    n_samples = vectors.shape[0]
    cluster_size = n_samples // num_clusters  # Target size for each cluster

    # k-means++ initialization
    centers = []
    centers.append(vectors[np.random.choice(n_samples)])  # First center randomly chosen
    for _ in range(1, num_clusters):
        distances = np.min(euclidean_distances(vectors, np.array(centers)), axis=1)
        squared_distances = distances ** 2
        total = np.sum(squared_distances)

        if total == 0 or np.isnan(total):
            probabilities = np.ones(n_samples) / n_samples
        else:
            probabilities = squared_distances / total

        next_center_idx = np.random.choice(n_samples, p=probabilities)
        centers.append(vectors[next_center_idx])
    centers = np.array(centers)

    for iteration in range(max_iter):
        # Step 1: Assign points to nearest cluster
        distances = euclidean_distances(vectors, centers)
        initial_labels = np.argmin(distances, axis=1)

        # Step 2: Balance cluster sizes
        sorted_indices = np.argsort(distances[np.arange(n_samples), initial_labels])
        labels = -1 * np.ones(n_samples, dtype=int)
        remaining_indices = sorted_indices[num_clusters * cluster_size:]  # Excess data points

        for cluster_idx in range(num_clusters):
            start_idx = cluster_idx * cluster_size
            if cluster_idx == num_clusters - 1:  # Last cluster takes the remaining data
                end_idx = n_samples
            else:
                end_idx = (cluster_idx + 1) * cluster_size
            labels[sorted_indices[start_idx:end_idx]] = cluster_idx

        # Assign remaining data points to the nearest cluster
        remaining_distances = distances[remaining_indices]
        for idx, point_idx in enumerate(remaining_indices):
            nearest_cluster = np.argmin(remaining_distances[idx])
            labels[point_idx] = nearest_cluster

        # Step 3: Recalculate cluster centers
        new_centers = np.array([vectors[labels == i].mean(axis=0) for i in range(num_clusters)])

        # Check for convergence
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # Final check: ensure no points remain unassigned
    if np.any(labels == -1):
        raise ValueError("Some data points were not assigned to any cluster.")

    return labels


def perform_hierarchical_clustering(categories, num_layers, cluster_counts, vector_model):
    """
    階層的にBalanced KMeansでクラスタリングする。各層のクラスタIDは親クラスタIDを継承（例: 3 → 3-2 → 3-2-4）。

    Args:
        categories (list): smallカテゴリ（全カテゴリ）のリスト
        num_layers (int): 層の数（例: 3なら layer1, layer2, layer3）
        cluster_counts (list): 各層のクラスタ数（例: [12, 6, 3]）
        vector_model: Word2VecモデルやGoogleNewsベクトル

    Returns:
        dict: 各layerの {クラスタID（階層表現）: [カテゴリ名]} の辞書
    """
    from collections import defaultdict

    # 初期状態（最上位層）では全カテゴリが1つの親クラスタに所属
    current_clusters = {"": categories}
    all_layer_clusters = {}

    for layer_idx in range(num_layers):
        next_clusters = defaultdict(list)
        layer_result = {}

        for parent_id, cat_list in current_clusters.items():
            if len(cat_list) == 0:
                continue

            vectors = get_category_vectors(cat_list, vector_model)
            num_clusters = min(cluster_counts[layer_idx], len(cat_list))  # 安全策：カテゴリ数より多くクラスタしない
            labels = balanced_kmeans_clustering(vectors, num_clusters=num_clusters, random_state=42)

            for cat, label in zip(cat_list, labels):
                # 新しいクラスタIDは親に"-N"を加えた形式
                new_cluster_id = f"{parent_id}-{label}" if parent_id else f"{label}"
                next_clusters[new_cluster_id].append(cat)

        all_layer_clusters[f"layer{layer_idx+1}"] = dict(next_clusters)
        current_clusters = next_clusters

    return all_layer_clusters

def assign_layer_labels(original_data, layer_clusters):
    """
    smallカテゴリに対して、各layerのクラスタ番号（ラベル）を付与する。

    Args:
        original_data (list of dict): 元のサービスデータ
        layer_clusters (dict): perform_hierarchical_clustering() の出力

    Returns:
        list of dict: 各サービスにlayer1, layer2などのラベルを追加したデータ
    """
    # カテゴリ名から各レイヤーのクラスタを引くための辞書
    category_to_layer = {}
    for layer_name, clusters in layer_clusters.items():
        for cluster_id, cat_list in clusters.items():
            for cat in cat_list:
                if cat not in category_to_layer:
                    category_to_layer[cat] = {}
                category_to_layer[cat][layer_name] = cluster_id

    updated_data = []
    for item in original_data:
        category = item["category"]
        if category not in category_to_layer:
            print(f"[警告] カテゴリが階層クラスタに見つかりません: {category}")
            continue

        new_item = item.copy()
        for layer_name, cluster_id in category_to_layer[category].items():
            new_item[layer_name + "categories"] = [cluster_id]
        updated_data.append(new_item)

    return updated_data

def split_dataset_by_clusters(data, layer_key, output_dir, test_ratio=0.2):
    """
    指定したレイヤーのクラスタごとにデータをファイルに分割し、学習・テストデータを保存。
    同じ name を持つデータはマージされる。

    Returns:
        list: 各クラスタに対応する (train_data, test_data) のリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    layer_field = layer_key
    
   # クラスタごとにデータを分ける（複数クラスタ対応）
    grouped = defaultdict(list)
    for item in data:
        if layer_field in item and item[layer_field]:
            for cluster_id in item[layer_field]:  # 複数クラスタ対応
                grouped[cluster_id].append(item)
        else:
            print(f"[警告] {layer_key} が存在しない or 空: {item['name']}")

    split_data = []

    # 各クラスタごとに処理
    for cluster_id, items in grouped.items():
        # 🔁 name ベースでマージ
        merged_data = {}
        for item in items:
            name = item["name"]
            if name not in merged_data:
                merged_data[name] = item.copy()
                # category list化
                if isinstance(item["category"], str):
                    merged_data[name]["category"] = [item["category"]]
                for k in item:
                    if k.startswith("layer") and k.endswith("categories"):
                        merged_data[name][k] = list(item[k]) if isinstance(item[k], list) else [item[k]]
            else:
                if item["category"] not in merged_data[name]["category"]:
                    merged_data[name]["category"].append(item["category"])
                for k in item:
                    if k.startswith("layer") and k.endswith("categories"):
                        if k not in merged_data[name]:
                            merged_data[name][k] = []
                        for val in item[k]:
                            if val not in merged_data[name][k]:
                                merged_data[name][k].append(val)

        merged_list = list(merged_data.values())

        # 🔁 train/test 分割
        random.shuffle(merged_list)
        test_size = int(len(merged_list) * test_ratio)
        test_data = merged_list[:test_size]
        train_data = merged_list[test_size:]

        # 保存
        train_path = os.path.join(output_dir, f"{layer_key}_{cluster_id}_train.json")
        test_path = os.path.join(output_dir, f"{layer_key}_{cluster_id}_test.json")

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        split_data.append((cluster_id, train_data, test_data))

    print(f"[確認] split_dataの件数: {len(split_data)}")
    for cid, train, test in split_data:
        print(f"[クラスタID: {cid}] train={len(train)}, test={len(test)}")

    return split_data


def merge_by_name(data):
    merged_data = {}
    for item in data:
        name = item["name"]
        if name not in merged_data:
            merged_data[name] = item.copy()
            if isinstance(item["category"], str):
                merged_data[name]["category"] = [item["category"]]
            for k in item:
                if k.startswith("layer") and k.endswith("categories"):
                    merged_data[name][k] = list(item[k]) if isinstance(item[k], list) else [item[k]]
        else:
            if item["category"] not in merged_data[name]["category"]:
                merged_data[name]["category"].append(item["category"])
            for k in item:
                if k.startswith("layer") and k.endswith("categories"):
                    if k not in merged_data[name]:
                        merged_data[name][k] = []
                    for val in item[k]:
                        if val not in merged_data[name][k]:
                            merged_data[name][k].append(val)
    return list(merged_data.values())

def generate_data_with_layer_labels(original_data, layer_cluster_maps):
    """
    original_data: 元データ（1カテゴリだけ持つ辞書のリスト）
    layer_cluster_maps: { "layer1": {cluster_id: [category...]}, ... }

    return: 各layerのラベルが追加されたデータリスト
    """
    # 各layerのカテゴリ→クラスタ辞書を構築
    reverse_maps = {}
    for layer_name, cluster_map in layer_cluster_maps.items():
        reverse_map = {}
        for cluster_id, cats in cluster_map.items():
            for cat in cats:
                reverse_map[cat] = str(cluster_id)
        reverse_maps[layer_name] = reverse_map

    # 元データに各layerのラベルを付加
    updated_data = []
    for item in original_data:
        base = {
            "name": item["name"],
            "description": item["description"],
            "categories": [item["category"]]
        }
        for layer_name, reverse_map in reverse_maps.items():
            cat = item["category"]
            label = reverse_map.get(cat)
            if label is None:
                print(f"[警告] {cat} に対応する {layer_name} が見つかりません")
                break
            base.setdefault(layer_name + "categories", []).append(label)
        else:
            updated_data.append(base)

    return updated_data


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
        
        
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, pos_weight, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask,token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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

def build_index_maps(save_dir):
    """
    save_dir 内の label_map ファイルを読み込み、index_maps を構築する。
    ファイル名は以下の形式を想定：
      e.g., layer1_0_label_map.json
    """
    index_maps = {}

    # すべての label_map ファイルを取得
    for filepath in glob.glob(os.path.join(save_dir, "layer*_*_label_map.json")):
        filename = os.path.basename(filepath)
        
        # 正規表現で layerX, cluster_id を抽出
        match = re.match(r"layer(\d+)_([^_]+)_label_map\.json", filename)
        if not match:
            continue
        
        layer_num = int(match.group(1))
        cluster_id = match.group(2)
        layer_key = f"layer{layer_num}"

        with open(filepath, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        if layer_key not in index_maps:
            index_maps[layer_key] = {}
        index_maps[layer_key][cluster_id] = label_map

    return index_maps

def calculate_metrics(true_set, predicted_set):
    """
    適合率、再現率、Fスコアを計算する。
    """
    true_set = set(true_set)
    predicted_set = set(predicted_set)
    true_positive = len(true_set.intersection(predicted_set))
    precision = true_positive / len(predicted_set) if predicted_set else 0
    recall = true_positive / len(true_set) if true_set else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    print(f"Precision: {precision}, Recall: {recall}, F-Score: {f_score}")
    return precision, recall, f_score

def average_metrics(metrics_list):
    
    """
    渡されたリストから適合率、再現率、Fスコアの平均を計算する。
    """
    avg_precision = sum(m[0] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_recall = sum(m[1] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_f_score = sum(m[2] for m in metrics_list) / len(metrics_list) if metrics_list else 0
    print(f"Precision: {avg_precision}, Recall: {avg_recall}, F-Score: {avg_f_score}")
    return avg_precision, avg_recall, avg_f_score



def train_model_per_cluster(train_file, model_save_path, catlist_save_path, model_name="bert-base-uncased",label_key="categories"):

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["description"] for item in data]
    raw_labels = [item[label_key] for item in data]

    # ラベル辞書の作成
    all_categories = sorted(set(cat for cats in raw_labels for cat in cats))
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}
    index_to_category = {i: cat for cat, i in category_to_index.items()}  # ← 🔧 ここで定義

    # ラベルのベクトル化
    labels = []
    for cats in raw_labels:
        vector = [0] * len(all_categories)
        for cat in cats:
            vector[category_to_index[cat]] = 1
        labels.append(vector)

    # pos_weight の計算
    pos_counts = torch.tensor([sum(l[i] for l in labels) for i in range(len(all_categories))])
    neg_counts = torch.tensor([len(labels) - p for p in pos_counts])
    pos_weight = neg_counts / (pos_counts + 1e-6)

    # データ分割
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = TextDataset(texts, labels, tokenizer)
    split = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    # モデル構築と学習
    model = BertForMultilabelClassification(model_name, len(all_categories), pos_weight)
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        enable_checkpointing=False, 
        logger=False
    )
    trainer.fit(model, train_loader, val_loader)

    # モデル保存
    torch.save(model.state_dict(), model_save_path)

    # 🔽 index_maps 用のラベルマップ保存
    label_map = {
        "index_to_category": {str(k): v for k, v in index_to_category.items()},
        "category_to_index": category_to_index
    }
    with open(catlist_save_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"[モデル学習完了] モデル保存: {model_save_path}")
    print(f"[ラベルマップ保存] {catlist_save_path}")



def predict_labels(model, tokenizer, text, threshold=0.5, device="cuda"):
    """
    description に対するマルチラベル予測（インデックス形式）
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).squeeze()
        probs = torch.sigmoid(logits)
    if probs.ndim == 0:
        return [0] if probs > threshold else []
    return (probs > threshold).nonzero(as_tuple=True)[0].tolist()


def load_model(model_class, model_name, num_labels, path, device="cuda"):
    """
    モデルのロード（保存されたstate_dictから）
    """
    model = model_class(model_name, num_labels, pos_weight=torch.ones(num_labels)).to(device)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_solution_hierarchical(
    test_data,
    model_class,
    tokenizer,
    model_name,
    index_maps,
    save_dir,
    max_layer=3,
    threshold=0.5
):
    """
    階層的にsmallカテゴリを予測し、true categoriesと比較してF1を評価。

    Args:
        test_data: [{"description": ..., "categories": [...]}]
        model_class: BertForMultilabelClassificationなど
        tokenizer: トークナイザ
        model_name: モデル名
        index_maps: 各レイヤー/クラスタIDごとの index_to_category / category_to_index を持つ辞書
        save_dir: 保存されたモデルのディレクトリ
        max_layer: 階層の最大数（GAで個体から決まる）
    Returns:
        avg_precision, avg_recall, avg_f1
    """
    results = []
    
    print(index_maps['layer1'].keys())

    for idx,item in enumerate(test_data):
        text = item["description"]
        true_labels = item["categories"]
        print(f"Processing text {idx + 1}/{len(test_data)}")
        print(text[:100])

        current_cluster_ids = []
        final_predicted_labels = []

        for layer in range(1, max_layer + 1):
            next_cluster_ids = []           
            layer_key = f"layer{layer}"
            
             # クラスタIDの初期化
            if layer == 1:
                # layer1だけはクラスタ分岐なしの特別処理
                model_path = os.path.join(save_dir, f"{layer_key}_all_model.pth")
                label_map_path = os.path.join(save_dir, f"{layer_key}_all_label_map.json")

                if not os.path.exists(model_path) or not os.path.exists(label_map_path):
                    print(f"[SKIP] layer1 model or map missing")
                    break

                with open(label_map_path, "r", encoding="utf-8") as f:
                    label_map = json.load(f)

                index_to_category = label_map["index_to_category"]
                category_to_index = label_map["category_to_index"]

                model = load_model(model_class, model_name, len(index_to_category), model_path)
                pred_indices = predict_labels(model, tokenizer, text, threshold)
                pred_categories = [index_to_category[str(i)] for i in pred_indices]

                print(f"[Layer1 Predict] {pred_categories}")
                current_cluster_ids = pred_categories  # layer2以降のクラスタ候補に使う
                continue

            for cluster_id in current_cluster_ids:
                # 対応するモデル・マップの読み込み
                if cluster_id not in index_maps.get(layer_key,{}):
                    print(f"[SKIP] {layer_key} に cluster_id {cluster_id} が存在しません")
                    continue

                label_map = index_maps[layer_key][cluster_id]
                index_to_category = label_map["index_to_category"]
                category_to_index = label_map["category_to_index"]

                model_path = os.path.join(save_dir, f"{layer_key}_{cluster_id}_model.pth")
                if not os.path.exists(model_path):
                    print(f"[SKIP] モデルファイルが存在しません: {model_path}")
                    continue

                model = load_model(model_class, model_name, len(index_to_category), model_path)

                pred_indices = predict_labels(model, tokenizer, text, threshold)
                key_type = type(next(iter(index_to_category.keys())))
                pred_categories = [index_to_category[key_type(i)] for i in pred_indices]
                print(f"[{layer_key}_{cluster_id} Predict] {pred_categories}")

                if layer == max_layer:
                    # 最終レイヤー：smallカテゴリを収集
                    final_predicted_labels = pred_categories
                    print(final_predicted_labels)
                else:
                    # 中間レイヤー：次のクラスタIDとして使用
                    next_cluster_ids.extend(pred_categories)

            current_cluster_ids = next_cluster_ids

        # 最終予測結果がなければスキップ
        if not final_predicted_labels:
            results.append((0, 0, 0))
            print("NO Result")
        else:
            precision, recall, f1 = calculate_metrics(true_labels, final_predicted_labels)
            results.append((precision, recall, f1))

    return average_metrics(results)


vector_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def fitness_function(individual, categories, all_data, model_class, tokenizer, model_name, vector_model=vector_model):
    """
    個体（階層構造案）に対する適応度（F1スコア）を計算する。

    Args:
        individual (list): [num_layers, k1, k2, ..., kn] の形式
        categories (list): smallカテゴリのリスト
        all_data (list): 学習用の元データ
        model_class: BertForMultilabelClassification クラス
        tokenizer: BERT トークナイザ
        model_name (str): 事前学習済みモデル名
        seed (int): ランダムシード
    Returns:
        float: F1スコア（適応度）
    """
    num_layers = individual[0]
    cluster_counts = individual[1:num_layers + 1]  # 各レイヤのクラスタ数（例: [12, 6, 2]）
    save_dir = f"./ga_saved_models/{'-'.join(map(str, individual))}"
    os.makedirs(save_dir, exist_ok=True)
    

    try:
        # ステップ 1: 階層クラスタリング（クラスタ構造を構築）
        layer_cluster_map = perform_hierarchical_clustering(categories, num_layers, cluster_counts, vector_model)

        # ステップ 2: データに階層カテゴリ（layer1, layer2, ...）を付与
        labeled_data = assign_layer_labels(all_data, layer_cluster_map)

        # ステップ 3: 最終層のlayerキー（例：layer3）
        final_layer_key = f"layer{num_layers}"

        # テンポラリ保存用ディレクトリ（個体IDで区別）
        output_dir = f"./temp_split_{'-'.join(map(str, individual))}"

        for layer_id in range(1, num_layers + 1):  # 1, 2, ..., num_layers
            layer_key = f"layer{layer_id}"
            output_dir = f"./temp_split_{'-'.join(map(str, individual))}_{layer_key}"
            
            if layer_id == num_layers:
                label_key = "category"  # ← 最終層はカテゴリそのもの
            else:
                label_key = f"{layer_key}categories"

             # === 分割対象の設定 ===
            if layer_id == 1:
                # layer1 は全体データ（分割なし）
                os.makedirs(output_dir, exist_ok=True)  # これを追加
                merge_labeled_data = merge_by_name(labeled_data)
                random.shuffle(merge_labeled_data)
                test_size = int(len(merge_labeled_data) * 0.2)
                test_data = merge_labeled_data[:test_size]
                train_data = merge_labeled_data[test_size:]

                # 保存（オプション）
                train_path = os.path.join(output_dir, f"{final_layer_key}_all_train.json")
                test_path = os.path.join(output_dir, f"{final_layer_key}_all_test.json")
                with open(train_path, "w", encoding="utf-8") as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=2)
                with open(test_path, "w", encoding="utf-8") as f:
                    json.dump(test_data, f, ensure_ascii=False, indent=2)

                split_data = [("all", train_data, test_data)]
                print(f"[INFO] 層数1のためクラスタ分割なしで train/test 分割: train={len(train_data)}, test={len(test_data)}")
            else:
                prev_layer_key = f"layer{layer_id - 1}categories"
                split_data = split_dataset_by_clusters(
                    labeled_data,
                    layer_key=prev_layer_key,  # 一つ前の層のクラスタで分割
                    output_dir=output_dir,
                    test_ratio=0.2
                )

            for cluster_id, train_data, test_data in split_data:
                model_save_path = os.path.join(save_dir, f"{layer_key}_{cluster_id}_model.pth")
                label_map_path = os.path.join(save_dir, f"{layer_key}_{cluster_id}_label_map.json")
                temp_train_file = os.path.join(output_dir, f"{layer_key}_{cluster_id}_train_temp.json")

                with open(temp_train_file, "w", encoding="utf-8") as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=2)

                train_model_per_cluster(
                    train_file=temp_train_file,
                    model_save_path=model_save_path,
                    catlist_save_path=label_map_path,
                    model_name=model_name,
                    label_key=label_key
                )

                
        # ステップ 4: モデル学習・評価（評価対象は最終層）
        test_data = load_json("/home/nakagawa/collaboration_test_data.json")  # あなたの検証データ

        index_maps = build_index_maps(save_dir)  # モデル保存先にある label_map から自動構築

        avg_precision, avg_recall, avg_f1 = evaluate_solution_hierarchical(
            test_data=test_data,
            model_class=model_class,
            tokenizer=tokenizer,
            model_name=model_name,
            index_maps=index_maps,
            save_dir=save_dir,
            max_layer=num_layers  # 例：3
        )
        return avg_f1 

    except Exception as e:
        print(f"[エラー] 個体 {individual} の評価中に例外が発生: {e}")
        traceback.print_exc()
        return 0.0  # 失敗した個体の適応度は 0


with open("/home/nakagawa/studydata/original/softfill_mashup_data_new_api_category_inclu_nomash_split_all_study_data.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

# unique なカテゴリを抽出
categories = sorted(set(item["category"] for item in all_data))


# -----------------------------
# パラメータ設定
# -----------------------------
MIN_LAYERS = 2
MAX_LAYERS = 4
MIN_CLUSTERS = 2
MAX_CLUSTERS = 76
POP_SIZE = 10
NGEN = 5
CXPB = 0.7  # 交叉確率
MUTPB = 0.3  # 突然変異確率

# -----------------------------
# DEAP セットアップ
# -----------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化問題（F1スコア）
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_individual():
    total_categories = len(categories)  # 例: 152
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    remaining = total_categories
    min_per_cluster = 2
    clusters = []

    for i in range(num_layers):
        # 最大でも「カテゴリ数 // 2」、最小でも 2クラスタ
        max_clusters = max(2, remaining // min_per_cluster)
        # ただし現在のレイヤ数 - i で割り切れるような余地を残す
        remaining_layers = num_layers - (i + 1)
        if remaining_layers > 0:
            # 最低限、次層でもクラスタ数2以上を保証するカテゴリ数が残るようにする
            max_clusters = min(max_clusters, remaining // (2 ** (remaining_layers)))

        # この層のクラスタ数（必ず2以上）
        k = random.randint(2, max_clusters)
        clusters.append(k)
        remaining = max(1, remaining // k)

    return [num_layers] + clusters

toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 評価関数（前に定義した fitness_function を使う）
toolbox.register("evaluate", fitness_function, categories=categories, all_data=all_data,
                 model_class=BertForMultilabelClassification,
                 tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
                 model_name="bert-base-uncased")

# 選択・交叉・突然変異
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def mutate_individual(individual):
    total_categories = len(categories)  # smallカテゴリの総数（例: 152）
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    remaining = total_categories
    min_per_cluster = 2
    clusters = []

    for i in range(num_layers):
        max_clusters = max(2, remaining // min_per_cluster)
        remaining_layers = num_layers - (i + 1)
        if remaining_layers > 0:
            max_clusters = min(max_clusters, remaining // (2 ** remaining_layers))

        k = random.randint(2, max_clusters)
        clusters.append(k)
        remaining = max(1, remaining // k)

    individual[:] = [num_layers] + clusters
    return individual,


toolbox.register("mutate", mutate_individual)

def save_log(best_solution, log_history):
    os.makedirs("ga_logs", exist_ok=True)

    with open("ga_logs/best_solution.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_genes": best_solution[:],
            "best_f1_score": best_solution.fitness.values[0]
        }, f, indent=2)

    with open("ga_logs/generation_history.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    print("✅ GAのログを ga_logs/ に保存しました。")

# -----------------------------
# GA 実行部分
# -----------------------------
def run_ga():
    population = toolbox.population(n=POP_SIZE)
    log_history = []  # 世代ごとの記録を保持

    print("Start evolution...")
    for gen in range(NGEN):
        print(f"\n--- Generation {gen + 1} ---")
        generation_log = {"generation": gen + 1, "individuals": []}

        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = (fit,)
            generation_log["individuals"].append({
                "genes": ind[:],
                "f1_score": fit
            })

        population = toolbox.select(offspring, k=len(population))
        best_ind = tools.selBest(population, k=1)[0]
        print(f"Best individual: {best_ind}, Fitness (F1): {best_ind.fitness.values[0]}")
        generation_log["best_individual"] = best_ind[:]
        generation_log["best_f1"] = best_ind.fitness.values[0]
        log_history.append(generation_log)

    # 最終ベスト個体
    best_solution = tools.selBest(population, k=1)[0]
    print(f"\n=== 最終ベスト個体: {best_solution} ===")

    # 保存処理
    save_log(best_solution, log_history)

    return best_solution

# 実行
best_solution = run_ga()
print(f"\n=== 最終ベスト個体: {best_solution} ===")