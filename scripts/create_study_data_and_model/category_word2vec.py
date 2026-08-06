from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_vectors
import gensim
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
import json
from collections import defaultdict
import torch
from torch.nn.functional import normalize
from transformers import BertTokenizer, BertModel
import torch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# カテゴリー名のリスト
categories =  [
            "documents",
            "images",
            "sales",
            "database",
            "accounting",
            "blogging",
            "standards",
            "translation",
            "storage",
            "geography",
            "collaboration",
            "chat",
            "movies",
            "search",
            "games",
            "hosting",
            "politics",
            "monitoring",
            "tagging"
        ]
# カテゴリーをWord2Vecでベクトル化
def split_categories(categories):
    return [category.split("-") for category in categories]

def train_word2vec(categories):
    split_categories_data = split_categories(categories)
    model = Word2Vec(split_categories_data, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_category_vectors(categories, model):
    def get_category_vector(category_name):
        words = category_name.split("-")
        word_vectors = [model[word] for word in words if word in model]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    
    return np.array([get_category_vector(category) for category in categories])

# BERTモデルをロードする関数
def load_bert_model(model_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

def get_category_embeddings_list(categories, tokenizer, model):
    """
    カテゴリリストをベクトル化する。
    Args:
        categories (list): カテゴリ名のリスト
        tokenizer: BERTのトークナイザ
        model: BERTモデル
    Returns:
        tuple: (ベクトルのNumPy配列, カテゴリ名リスト)
               - ベクトルはKMeansなどで利用可能な形式
    """
    embeddings = []
    for category in categories:
        # トークン化とベクトル化
        inputs = tokenizer(category, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 埋め込み（[CLS]トークンのベクトルを利用）
        last_hidden_state = outputs[0]  # outputs.last_hidden_state
        category_embedding = last_hidden_state.mean(dim=1).squeeze(0).numpy()
        embeddings.append(category_embedding)
    
    return np.array(embeddings)

model1 = gensim.models.KeyedVectors.load_word2vec_format(legacy_path('/home/nakagawa/lab-MLMLC/create_study_data_and_model/GoogleNews-vectors-negative300.bin'), binary=True)

category_vectors1 = get_category_vectors(categories, model1)


# 各クラスタリング手法の関数
def kmeans_clustering(vectors, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1)  # n_initを明示的に設定
    labels = kmeans.fit_predict(vectors)
    return labels


def hierarchical_clustering(vectors, num_clusters=10):
    agglo = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')  # affinityをmetricに変更
    labels = agglo.fit_predict(vectors)
    return labels


def dbscan_clustering(vectors, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(vectors)
    return labels

def mean_shift_clustering(vectors):
    mean_shift = MeanShift()
    labels = mean_shift.fit_predict(vectors)
    return labels

def gmm_clustering(vectors, num_components=10):
    gmm = GaussianMixture(n_components=num_components)
    gmm.fit(vectors)
    labels = gmm.predict(vectors)
    return labels

def spectral_clustering(vectors, num_clusters=10):
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors',random_state=42)
    labels = spectral.fit_predict(vectors)
    return labels

def affinity_propagation_clustering(vectors):
    affinity_propagation = AffinityPropagation()
    labels = affinity_propagation.fit_predict(vectors)
    return labels


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



# クラスタリング結果表示関数
def display_results(categories, labels):
    for category, label in zip(categories, labels):
        print(f"Category: {category} -> Cluster: {label}")

# クラスタリング結果をクラスタごとにまとめてJSONファイルに保存する関数
def display_results_to_file(categories, labels, filename):
    # クラスタごとにカテゴリーを分類
    cluster_dict = defaultdict(list)
    for category, label in zip(categories, labels):
        cluster_dict[label].append(category)
    
    # データを辞書形式に変換
    results = {
        "clusters": {str(cluster): categories for cluster, categories in cluster_dict.items()}
    }
    
    # JSONファイルに保存
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)



# # 実行例：K-meansクラスタリング
# kmeans_labels = kmeans_clustering(category_vectors1, num_clusters=8)
# display_results_to_file(categories, kmeans_labels, "category_soft_clu_kmeans_google_results_8ver.json")

# # 実行例：K-meansクラスタリング
# kmeans_labels = kmeans_clustering(category_vectors2, num_clusters=8)
# display_results_to_file(categories, kmeans_labels, "category_soft_clu_kmeans_fast_results_8ver.json")

# # 実行例：K-meansクラスタリング
# kmeans_labels = kmeans_clustering(category_vectors3, num_clusters=8)
# display_results_to_file(categories, kmeans_labels, "category_soft_clu_kmeans_bert_results_8ver.json")

# # 実行例：Mean Shiftクラスタリング
# mean_shift_labels = mean_shift_clustering(category_vectors)
# display_results_to_file(categories, mean_shift_labels, "softclu_mean_shift_results.txt")

# # 実行例：階層型クラスタリング
# hierarchical_labels = hierarchical_clustering(category_vectors, num_clusters=3)
# display_results_to_file(categories, hierarchical_labels, "category_Hcategory1_SPcategory1_soft_clu_hierarchical_results_3ver.json")

# # 実行例：DBSCANクラスタリング
# dbscan_labels = dbscan_clustering(category_vectors, eps=0.5, min_samples=5)
# display_results_to_file(categories, dbscan_labels, "softclu_dbscan_results.txt")

# # 実行例：Gaussian Mixture Modelクラスタリング
# gmm_labels = gmm_clustering(category_vectors, num_components=10)
# display_results_to_file(categories, gmm_labels, "softclu_gmm_results.txt")

# # 実行例：Spectral Clustering
# spectral_labels = spectral_clustering(category_vectors, num_clusters=3)
# display_results_to_file(categories, spectral_labels, "category_Hcategory1_SPcategory1_soft_clu_spectral_results_3ver.json")

# # 実行例：Affinity Propagationクラスタリング
# affinity_labels = affinity_propagation_clustering(category_vectors)
# display_results_to_file(categories, affinity_labels, "softclu_affinity_propagation_results.txt")

def load_clusters_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["clusters"]

cluster_json_path = legacy_path("/home/nakagawa/lab-MLMLC/cluster/8_3_2_3/category_soft_clu_balanced_balanced_kmeans+_results_8_7_layer4_8_3_2_3ver.json")  # ← ここをJSONのファイルパスに

clusters_dict = load_clusters_from_json(cluster_json_path)

# クラスタごとに処理
for cluster_id, category_list in clusters_dict.items():
    print(f"Processing cluster {cluster_id} with {len(category_list)} categories...")
    
    # ここでクラスタごとのベクトルを作成（例：Word2Vec）
    category_vectors = get_category_vectors(category_list, model1)  # Word2Vecでベクトル化

    # クラスタリング（例：Balanced KMeans）
    labels = balanced_kmeans_clustering(category_vectors, num_clusters=2, random_state=42)

    # 保存ファイル名にクラスタ番号をつける
    filename = legacy_path(f"/home/nakagawa/lab-MLMLC/cluster/8_3_2_3/category_soft_clu_balanced_balanced_kmeans+_results_8_7_{cluster_id}_layer4_8_3_2_3ver.json")
    display_results_to_file(category_list, labels, filename)

