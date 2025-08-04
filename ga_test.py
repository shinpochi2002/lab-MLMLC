# ----------------- 基本処理 -----------------
import os
import json
import random
import numpy as np
from collections import defaultdict

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
        probabilities = distances**2 / np.sum(distances**2)
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

def hierarchical_clustering_by_layers(categories, individual, vectorizer, seed=42):
    from collections import defaultdict

    current_categories = categories
    full_structure = {}
    parent_paths = ['']  # 初期は全体を1つのグループとする

    for depth, num_clusters in enumerate(individual):
        layer_name = f"layer{depth + 1}"
        full_structure[layer_name] = {}
        next_parent_paths = []

        for path in parent_paths:
            # クラスタリング対象のカテゴリ
            if path == '':
                subcategories = current_categories
            else:
                subcategories = full_structure[f"layer{depth}"][path]

            if len(subcategories) < num_clusters:
                print(f"[警告] {path}のカテゴリ数がクラスタ数{num_clusters}未満のためスキップ")
                continue

            vectors = vectorizer(subcategories)
            labels = balanced_kmeans_clustering(vectors, num_clusters, random_state=seed)

            for i, label in enumerate(labels):
                new_path = f"{path}-{label}" if path != '' else str(label)
                full_structure[layer_name].setdefault(new_path, []).append(subcategories[i])
                next_parent_paths.append(new_path)

        parent_paths = next_parent_paths

    return full_structure


def perform_hierarchical_clustering(categories, num_layers, cluster_counts, vector_model):
    """
    階層的にBalanced KMeansでクラスタリングする。

    Args:
        categories (list): smallカテゴリ（全カテゴリ）のリスト
        num_layers (int): 層の数（例: 2なら layer1, layer2）
        cluster_counts (list): 各層でのクラスタ数（例: [12, 6]）
        vector_model: Word2VecモデルやGoogleNewsベクトル

    Returns:
        dict: 各layerの {クラスタID: [カテゴリ名]} の辞書
    """
    current_categories = categories
    all_layer_clusters = {}

    for layer_idx in range(num_layers):
        vectors = get_category_vectors(current_categories, vector_model)
        labels = balanced_kmeans_clustering(vectors, num_clusters=cluster_counts[layer_idx], random_state=42)

        # クラスタごとにカテゴリをまとめる
        clusters = defaultdict(list)
        for cat, label in zip(current_categories, labels):
            clusters[f"{layer_idx}-{label}"].append(cat)

        all_layer_clusters[layer_idx] = clusters

        # 次のレイヤーの入力として使うカテゴリを更新
        current_categories = []
        for cluster in clusters.values():
            current_categories.extend(cluster)

    layer_clusters_strkey = {f"layer{idx+1}": clusters for idx, clusters in all_layer_clusters.items()}
    return layer_clusters_strkey


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
    for layer_idx, clusters in layer_clusters.items():
        for cluster_id, cat_list in clusters.items():
            for cat in cat_list:
                if cat not in category_to_layer:
                    category_to_layer[cat] = {}
                category_to_layer[cat][f"layer{layer_idx + 1}"] = cluster_id

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

    Args:
        data (list of dict): 各レイヤー付きのサービスデータ
        layer_key (str): "layer1" や "layer2" など
        output_dir (str): 出力先ディレクトリ
        test_ratio (float): テストデータの割合

    Returns:
        list: 各クラスタに対応する (train_data, test_data) のリスト
    """
    os.makedirs(output_dir, exist_ok=True)

    grouped = defaultdict(list)
    for item in data:
        if layer_key in item:
            cluster_id = item[layer_key][0]
            grouped[cluster_id].append(item)
        else:
            print(f"[警告] {layer_key} が存在しない項目: {item['name']}")

    split_data = []

    for cluster_id, items in grouped.items():
        random.shuffle(items)
        test_size = int(len(items) * test_ratio)
        test_data = items[:test_size]
        train_data = items[test_size:]

        train_path = os.path.join(output_dir, f"{layer_key}_{cluster_id}_train.json")
        test_path = os.path.join(output_dir, f"{layer_key}_{cluster_id}_test.json")

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        split_data.append((train_data, test_data))

    return split_data


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


def split_data_by_layer_category(data, layer_name, output_dir):
    """
    data: ラベル付与済みデータ
    layer_name: "layer1", "layer2" など
    output_dir: 保存ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    grouped = defaultdict(list)
    key = f"{layer_name}categories"

    for item in data:
        if key not in item:
            print(f"[警告] {key} が存在しないデータ: {item.get('name', '')}")
            continue
        cluster_id = item[key][0]  # 1カテゴリだけなので0番目だけ
        grouped[cluster_id].append(item)

    for cluster_id, items in grouped.items():
        filename = os.path.join(output_dir, f"{layer_name}_{cluster_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"[出力] {filename} ({len(items)} 件)")


def split_train_test_data_per_file(input_dir, output_dir, test_ratio=0.2):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        total_count = len(all_data)
        test_size = int(total_count * test_ratio)

        all_names = [item["name"] for item in all_data]
        test_names = set(random.sample(all_names, test_size))

        test_data = [item for item in all_data if item["name"] in test_names]
        train_data = [item for item in all_data if item["name"] not in test_names]

        base_name = filename.replace(".json", "")
        test_path = os.path.join(output_dir, f"{base_name}_test_data.json")
        train_path = os.path.join(output_dir, f"{base_name}_study_data.json")

        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        print(f"[分割完了] {filename} → 学習: {len(train_data)} 件, テスト: {len(test_data)} 件")


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



def train_model_per_cluster(train_file, model_save_path, catlist_save_path, model_name="bert-base-uncased"):

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["description"] for item in data]
    raw_labels = [item["middlecategories"] for item in data]

    all_categories = sorted(set(cat for cats in raw_labels for cat in cats))
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}

    labels = []
    for cats in raw_labels:
        vector = [0] * len(all_categories)
        for cat in cats:
            vector[category_to_index[cat]] = 1
        labels.append(vector)

    pos_counts = torch.tensor([sum(l[i] for l in labels) for i in range(len(all_categories))])
    neg_counts = torch.tensor([len(labels) - p for p in pos_counts])
    pos_weight = neg_counts / (pos_counts + 1e-6)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = TextDataset(texts, labels, tokenizer)
    split = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = BertForMultilabelClassification(model_name, len(all_categories), pos_weight)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")]
    )
    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), model_save_path)

    with open(catlist_save_path, "w", encoding="utf-8") as f:
        json.dump(all_categories, f, ensure_ascii=False, indent=2)

    print(f"[モデル学習完了] 保存: {model_save_path}")


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



def evaluate_model_per_cluster(cluster_id, model_class, tokenizer, model_name, train_data, test_data, layer_label_key):
    """
    指定されたクラスタのモデルを訓練し、テストデータで精度を評価する。
    Args:
        cluster_id (int): クラスタ番号
        model_class: BertForMultilabelClassification クラス
        tokenizer: BERTトークナイザ
        model_name (str): 事前学習モデル名
        train_data (list): 学習データ（dict形式）
        test_data (list): テストデータ（dict形式）
        layer_label_key (str): ラベルキー（例: "layer2categories"）
    Returns:
        tuple: (avg_precision, avg_recall, avg_f1)
    """
    # ラベルリストの抽出とインデックス化
    raw_labels = [item[layer_label_key] for item in train_data]
    all_categories = sorted(set(cat for cats in raw_labels for cat in cats))
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}
    index_to_category = {i: cat for cat, i in category_to_index.items()}

    # ラベルベクトル化
    def encode_labels(data):
        encoded = []
        for item in data:
            vec = [0] * len(all_categories)
            for cat in item[layer_label_key]:
                if cat in category_to_index:
                    vec[category_to_index[cat]] = 1
            encoded.append(vec)
        return encoded

    train_texts = [item["description"] for item in train_data]
    train_labels = encode_labels(train_data)

    test_texts = [item["description"] for item in test_data]
    test_labels = encode_labels(test_data)

    # pos_weight の計算
    label_tensor = torch.tensor(train_labels)
    pos_count = label_tensor.sum(dim=0)
    neg_count = len(train_labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-6)

    # データローダー作成
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # モデル構築と学習
    model = model_class(model_name, len(all_categories), pos_weight)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        enable_progress_bar=False,
        logger=False
    )

    trainer.fit(model, train_loader, val_loader)

    # 推論と評価
    metrics_list = []

    for item in test_data:
        text = item["description"]
        true_labels = item[layer_label_key]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).squeeze()
            probs = torch.sigmoid(logits)

        predicted_indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [index_to_category[i] for i in predicted_indices]

        precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
        metrics_list.append((precision, recall, f1))

    avg_precision, avg_recall, avg_f1 = average_metrics(metrics_list)
    return avg_precision, avg_recall, avg_f1


def evaluate_solution(clustered_datasets, model_class, tokenizer, model_name, layer_label_key):
    """
    すべてのクラスタ（モデル）に対してモデルを訓練し評価する。
    Args:
        clustered_datasets (dict): { cluster_id: {"train": [...], "test": [...]} } の形式
        model_class: BertForMultilabelClassification クラス
        tokenizer: BERTトークナイザ
        model_name (str): 事前学習モデル名
        layer_label_key (str): 評価対象ラベルキー（例: "layer2categories"）
    Returns:
        tuple: 全体の (avg_precision, avg_recall, avg_f1)
    """
    all_metrics = []

    for cluster_id, datasets in clustered_datasets.items():
        print(f"Evaluating cluster {cluster_id}...")
        train_data = datasets["train"]
        test_data = datasets["test"]

        if not train_data or not test_data:
            print(f"  [警告] cluster {cluster_id} のデータが不足しています。スキップします。")
            continue

        precision, recall, f1 = evaluate_model_per_cluster(
            cluster_id,
            model_class,
            tokenizer,
            model_name,
            train_data,
            test_data,
            layer_label_key
        )
        all_metrics.append((precision, recall, f1))

    # 全体の平均を集計
    avg_precision, avg_recall, avg_f1 = average_metrics(all_metrics)
    return avg_precision, avg_recall, avg_f1


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
    

    try:
        # ステップ 1: 階層クラスタリング（クラスタ構造を構築）
        layer_cluster_map = perform_hierarchical_clustering(categories, num_layers, cluster_counts, vector_model)

        # ステップ 2: データに階層カテゴリ（layer1, layer2, ...）を付与
        labeled_data = assign_layer_labels(all_data, layer_cluster_map)

        # ステップ 3: 最終層のlayerキー（例：layer3）
        final_layer_key = f"layer{num_layers}"

        # テンポラリ保存用ディレクトリ（個体IDで区別）
        output_dir = f"./temp_split_{'-'.join(map(str, individual))}"

        # クラスタごとに分割（train/test）して保存し、リストで受け取る
        split_data = split_dataset_by_clusters(
            labeled_data,
            layer_key=final_layer_key,
            output_dir=output_dir,
            test_ratio=0.2
        )

        # clustered_datasets形式に変換（evaluate_solution用）
        clustered_datasets = {
            f"{final_layer_key}_{i}": {
                "train": train_data,
                "test": test_data
            }
            for i, (train_data, test_data) in enumerate(split_data)
        }

        # ステップ 4: モデル学習・評価（評価対象は最終層）
        final_layer_key = f"layer{num_layers}categories"
        avg_precision, avg_recall, avg_f1 = evaluate_solution(
            clustered_datasets, model_class, tokenizer, model_name, final_layer_key
        )

        return avg_f1  # 適応度として F1 スコアを返す

    except Exception as e:
        print(f"[エラー] 個体 {individual} の評価中に例外が発生: {e}")
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
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    return [num_layers] + [random.randint(MIN_CLUSTERS, MAX_CLUSTERS) for _ in range(num_layers)]

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
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    new_clusters = [random.randint(MIN_CLUSTERS, MAX_CLUSTERS) for _ in range(num_layers)]
    individual[:] = [num_layers] + new_clusters
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