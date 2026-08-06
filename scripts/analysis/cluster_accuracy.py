from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# -*- coding: utf-8 -*-
# clusters.json を読み込み、カテゴリ名を GoogleNews ベクトルに変換して
# Silhouette / Calinski–Harabasz / Davies–Bouldin を算出するシンプルスクリプト

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
import gensim

# -----------------------------
# 1) 入力
# -----------------------------
CLUSTERS_JSON = legacy_path("/home/nakagawa/cluster/4cluster/category_soft_clu_balanced_balanced_kmeans+_results_4ver.json")  # あなたが貼った形式のJSON
W2V_PATH = legacy_path("GoogleNews-vectors-negative300.bin")  # GoogleNewsベクトル

# -----------------------------
# 2) カテゴリ名 -> ベクトル化（あなたの関数を少し堅牢化）
# -----------------------------
def get_category_vectors(categories, model):
    """
    category-name を "-" で分割し、語の平均でカテゴリベクトルを作る。
    OOV (語彙外) は無視。全語OOVの場合は ゼロベクトル を返す。
    """
    def get_category_vector(category_name):
        words = category_name.split("-")
        word_vecs = []
        for w in words:
            if w in model:
                word_vecs.append(model[w])
        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(model.vector_size, dtype=np.float32)

    vecs = np.vstack([get_category_vector(cat) for cat in categories])
    return vecs

# -----------------------------
# 3) JSON 読み込み → X, y の作成
# -----------------------------
def load_clusters_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = data["clusters"]  # { "0": [...], "1": [...], ... }
    # フラット化
    categories, labels = [], []
    for cid_str, cats in clusters.items():
        cid = int(cid_str)
        for c in cats:
            categories.append(c)
            labels.append(cid)
    return categories, np.array(labels, dtype=int), clusters

# -----------------------------
# 4) メイン：内部評価
# -----------------------------
def evaluate_clustering_internal(X, y, metric_silhouette="cosine"):
    """
    X: (n_samples, n_features) ベクトル
    y: (n_samples,) クラスタID
    """
    # クラスタが2未満、またはサンプルが少なすぎる場合は指標が定義されない
    n_clusters = len(np.unique(y))
    if n_clusters < 2 or len(y) < 3:
        return {
            "Silhouette": np.nan,
            "Calinski-Harabasz": np.nan,
            "Davies-Bouldin": np.nan,
            "note": "Need at least 2 clusters and >= 3 samples."
        }

    # Silhouette（cosine距離で評価）
    sil = silhouette_score(X, y, metric=metric_silhouette)

    # CH/DB は特徴空間上の分散ベース（実質的にユークリッド想定）
    # スケールの影響を減らすため L2 正規化した空間で計算（慣例的にOK）
    X_norm = normalize(X, norm="l2")
    ch = calinski_harabasz_score(X_norm, y)
    db = davies_bouldin_score(X_norm, y)

    return {"Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db}

# -----------------------------
# 5) 実行
# -----------------------------
if __name__ == "__main__":
    # JSON 読み込み
    categories, labels, clusters_dict = load_clusters_json(CLUSTERS_JSON)
    print(f"#samples={len(categories)}, #clusters={len(clusters_dict)}")

    # Word2Vec 読み込み（かなり重いので最初の1回に時間がかかります）
    model = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)

    # ベクトル化
    X_all = get_category_vectors(categories, model)

    # ゼロベクトルの扱い（全語OOVのカテゴリ）
    is_zero = np.linalg.norm(X_all, axis=1) == 0
    n_zero = int(is_zero.sum())
    if n_zero > 0:
        zero_names = [categories[i] for i in np.where(is_zero)[0]]
        print(f"[WARN] {n_zero} categories had no in-vocab words and became zero vectors.")
        # 必要なら：print(zero_names)
        # ゼロベクトルは評価を歪める可能性があるので除外推奨
        X = X_all[~is_zero]
        y = labels[~is_zero]
        kept = len(y)
        print(f"Filtered to {kept} samples after removing zero vectors.")
    else:
        X, y = X_all, labels

    # 各クラスタのサイズを表示（バランス確認用）
    unique, counts = np.unique(y, return_counts=True)
    size_info = dict(zip(unique.tolist(), counts.tolist()))
    print("Cluster sizes (after filter):", size_info)

    # 評価
    scores = evaluate_clustering_internal(X, y, metric_silhouette="cosine")
    print("=== Internal Validation Scores ===")
    for k, v in scores.items():
        print(f"{k}: {v}")
