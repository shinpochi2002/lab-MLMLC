# MLMLC: Hierarchical Multi-Label API Recommendation

## Overview

This repository contains research code for API recommendation from mashup requirement text. The main approach uses BERT-based multi-label classifiers over API/category labels, including hierarchical category patterns such as layer-1, layer-2, layer-3, and layer-4 models. The repository also includes TF-IDF, Sentence-BERT, cross-encoder, and One-vs-Rest baselines for comparison.

The project is organized around the following directories:

- `data/`: raw data, processed JSON files, category label lists, cluster definitions, splits, CV files, and fixed test subsets.
- `scripts/`: preprocessing, training, inference/evaluation, analysis, baseline, and utility scripts.
- `models/`: local BERT assets, embeddings, and trained checkpoint locations.
- `results/`: generated tables, logs, hierarchy outputs, and baseline outputs.
- `docs/`: supplementary notes and experiment reports.

## Requirements

- Python 3.10 or later
- CUDA-capable GPU for BERT training/evaluation runs
- Local BERT model files under `models/base/bert-base-uncased/`, or access to the Hugging Face model cache

Python dependencies are unified in the root `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Dataset

The main dataset files are stored under `data/`.

- Raw mashup/API data:
  - `data/raw/mashup_data/`
  - `data/raw/new_api_data/`
- Processed train/test JSON:
  - `data/processed/root_json/study_data_mashup_apis.json`
  - `data/processed/root_json/test_data_mashup_apis.json`
  - `data/processed/root_json/api_dataset_like_manifest.json`
- Category and hierarchy definitions:
  - `data/api_category_list/`
  - `data/cluster_definitions/`
- Fixed test subsets:
  - `data/test_subsets/`

The baseline experiments use a closed API catalog from `api_dataset_like_manifest.json`. Historical reports use 909 API candidates and 336 evaluation records. The current fixed split uses **100 validation records for hierarchy selection and the remaining 236 for test**, with seed 42 and the same mashups across all hierarchy/masking variants. The original 336-record files are retained. See [the split and selection workflow](docs/validation_test_split.md).

## Installation

Clone the repository, create an environment, and install the dependencies:

```bash
cd /home/nakagawa/lab-MLMLC
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use the local BERT assets, keep them at:

```text
models/base/bert-base-uncased/
```

## Usage

### Preprocessing

Create hierarchy-enriched test data from a cluster definition:

```bash
python scripts/data_prep/create_hierarchical_testdata.py \
  --cluster-dir data/cluster_definitions/cluster/16cluster \
  --mashup-json data/processed/root_json/test_data_mashup_apis.json \
  --out-base results/hierarchy/output
```

The script writes hierarchy definitions and enriched test JSON files under `results/hierarchy/output/<cluster_tag>/`.

### Training

Train category-specific API recommendation models:

```bash
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category.py
```

Train hierarchy models for specific layer patterns:

```bash
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer2.py
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer3.py
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer4.py
```

These training scripts save model weights under `models/checkpoints/` through the path migration helper in `scripts/common/project_paths.py`. Some scripts still contain experiment-specific constants, so check the input path and target pattern before long runs.

### Evaluation

Model directory names follow the [inference-based naming rules](docs/hierarchy_model_naming.md); experiment identifiers and E-calculation labels are separate.

Create (or verify) the fixed validation/test split:

```bash
python scripts/data_prep/create_validation_test_split.py
```

List the validation hierarchy tasks without running models:

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked \
  --split validation \
  --layers 2,3,4 \
  --dry-run
```

Evaluate the hierarchy candidates on validation:

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked \
  --split validation \
  --layers 2,3,4
```

Use `scripts/analysis/select_hierarchy_from_validation.py --results-dir results/hierarchy/hierarchy_existing_models_api_name_masked/validation` to save the best validation-F1 choice, then run `--split test --patterns SELECTED_PATTERN` for that choice. See the [complete workflow](docs/validation_test_split.md). The runner defaults to `--split test`; historical 336-record evaluations require `--split full`.

Run a TF-IDF baseline:

```bash
python scripts/baselines/run_tfidf_baseline.py \
  --top-k 10 \
  --run-name tfidf_top10_909api
```

Run a supervised TF-IDF One-vs-Rest baseline:

```bash
python scripts/baselines/run_tfidf_ovr_baseline.py \
  --train_path data/processed/root_json/study_data_mashup_apis.json \
  --test_path data/splits/model_selection/test/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json \
  --model_type logistic \
  --threshold 0.5 \
  --output_dir results/baselines/tfidf_ovr_logistic_test236_masked
```

## Experimental Settings

- Encoder: `bert-base-uncased`
- Task: multi-label API/category prediction from mashup requirement descriptions
- Loss: `BCEWithLogitsLoss`
- Default threshold: `0.5`
- Training split: typically 80/20 train/validation in the training scripts
- Batch size: commonly `8` for BERT training
- Max epochs: commonly `50` with early stopping on validation loss
- Metrics: sample-level average precision, recall, and F1
- Baselines: TF-IDF cosine, Sentence-BERT cosine, BERT cross-encoder, TF-IDF OvR, and Sentence-BERT OvR

## Expected Output

Preprocessing outputs:

- `results/hierarchy/output/<cluster_tag>/category_hierarchy_<cluster_tag>.json`
- `results/hierarchy/output/<cluster_tag>/test_data_*_with_hierarchy_<cluster_tag>.json`
- `results/hierarchy/output/<cluster_tag>/unknown_categories_<cluster_tag>.json`

Training outputs:

- `models/checkpoints/hierarchy/`
- `models/checkpoints/category_kfold/`
- `data/api_category_list/`

Evaluation outputs:

- `results/hierarchy/hierarchy_existing_models*/`
- `results/hierarchy/hierarchy_existing_models*/test_data_manifest.json`
- `results/hierarchy/hierarchy_existing_models*/run_summary.json`
- `results/hierarchy/hierarchy_existing_models*/*_results.csv`

Baseline outputs:

- `results/baselines/*.json`
- `results/baselines/*.csv`
- `results/baselines/<run_name>/metrics.json`
- `results/baselines/<run_name>/predictions.json`

## Citation


## License


# MLMLC: 階層型マルチラベルAPI推薦

## 概要

このリポジトリは、マッシュアップの要求文から利用すべきAPIを推薦する研究コードです。主手法では、BERTベースのマルチラベル分類器を用いて、APIラベルやカテゴリラベルを予測します。layer-1からlayer-4までの階層カテゴリ構造を使ったモデルに加えて、比較用のTF-IDF、Sentence-BERT、cross-encoder、One-vs-Rest baselineも含まれています。

主なディレクトリ構成は次の通りです。

- `data/`: raw data、processed JSON、カテゴリラベル一覧、クラスタ定義、分割データ、CVファイル、固定テストサブセット
- `scripts/`: 前処理、学習、推論/評価、分析、baseline、ユーティリティスクリプト
- `models/`: ローカルBERTモデル、埋め込み、学習済みcheckpointの配置先
- `results/`: 生成された表、ログ、階層モデル出力、baseline出力
- `docs/`: 補足資料、整理メモ、実験レポート

## 必要要件

- Python 3.10以上
- BERTの学習・評価を行う場合はCUDA対応GPUを推奨
- `models/base/bert-base-uncased/` 配下のローカルBERTモデル、またはHugging Faceのモデルキャッシュ

Python依存関係はルートの `requirements.txt` に統一しています。

```bash
pip install -r requirements.txt
```

## データセット

主要なデータセットは `data/` 配下に配置しています。

- raw mashup/API data:
  - `data/raw/mashup_data/`
  - `data/raw/new_api_data/`
- processed train/test JSON:
  - `data/processed/root_json/study_data_mashup_apis.json`
  - `data/processed/root_json/test_data_mashup_apis.json`
  - `data/processed/root_json/api_dataset_like_manifest.json`
- カテゴリ・階層定義:
  - `data/api_category_list/`
  - `data/cluster_definitions/`
- 固定テストサブセット:
  - `data/test_subsets/`

baseline実験では、`api_dataset_like_manifest.json` に含まれる909件のAPIカタログを閉じた候補集合として使います。現在は、従来の評価用336件を**階層構造選定用validation 100件と、残りのtest 236件**に分割しています。seedは42に固定し、全階層・マスク有無で同じサンプルを分割先に割り当てます。元の336件も保持しています。[分割と選定の手順](docs/validation_test_split.md)を参照してください。

## インストール

リポジトリに移動し、仮想環境を作成して依存関係をインストールします。

```bash
cd /home/nakagawa/lab-MLMLC
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

ローカルのBERTモデルを使う場合は、次の場所に配置します。

```text
models/base/bert-base-uncased/
```

## 使い方

### 前処理

クラスタ定義から、階層ラベル付きのテストデータを作成します。

```bash
python scripts/data_prep/create_hierarchical_testdata.py \
  --cluster-dir data/cluster_definitions/cluster/16cluster \
  --mashup-json data/processed/root_json/test_data_mashup_apis.json \
  --out-base results/hierarchy/output
```

出力は `results/hierarchy/output/<cluster_tag>/` 配下に保存されます。

### 学習

カテゴリ別のAPI推薦モデルを学習します。

```bash
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category.py
```

特定の階層パターンに対応するモデルを学習します。

```bash
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer2.py
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer3.py
python scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer4.py
```

これらの学習スクリプトは、`scripts/common/project_paths.py` のパス移行ヘルパーを通じて `models/checkpoints/` 配下にモデル重みを保存します。一部のスクリプトには実験ごとの固定パスや対象パターンが残っているため、長時間実行する前に入力パスと対象パターンを確認してください。

### 評価

モデルフォルダ名は[推論コードに基づく命名ルール](docs/hierarchy_model_naming.md)に従います。既存の構成識別名とE計算用の表記は区別します。

固定したvalidation/test分割を生成（または再確認）します。

```bash
python scripts/data_prep/create_validation_test_split.py
```

validation用の階層評価タスクを、モデル実行なしで確認します。

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked \
  --split validation \
  --layers 2,3,4 \
  --dry-run
```

階層構成の候補をvalidationで評価します。

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked \
  --split validation \
  --layers 2,3,4
```

`scripts/analysis/select_hierarchy_from_validation.py --results-dir results/hierarchy/hierarchy_existing_models_api_name_masked/validation`でvalidation F1が最大の構成を記録し、その構成を`--split test --patterns SELECTED_PATTERN`で評価します。[詳しい手順](docs/validation_test_split.md)を参照してください。ランナーの既定値は`--split test`で、旧336件を使う場合は`--split full`を明示します。

TF-IDF baselineを実行します。

```bash
python scripts/baselines/run_tfidf_baseline.py \
  --top-k 10 \
  --run-name tfidf_top10_909api
```

教師ありTF-IDF One-vs-Rest baselineを実行します。

```bash
python scripts/baselines/run_tfidf_ovr_baseline.py \
  --train_path data/processed/root_json/study_data_mashup_apis.json \
  --test_path data/splits/model_selection/test/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json \
  --model_type logistic \
  --threshold 0.5 \
  --output_dir results/baselines/tfidf_ovr_logistic_test236_masked
```

## 実験設定

- エンコーダ: `bert-base-uncased`
- タスク: マッシュアップ要求文からのマルチラベルAPI/カテゴリ予測
- 損失関数: `BCEWithLogitsLoss`
- 既定の閾値: `0.5`
- 学習分割: 多くの学習スクリプトで80/20のtrain/validation分割
- バッチサイズ: BERT学習では主に `8`
- 最大epoch数: 主に `50`、validation lossによるearly stoppingあり
- 評価指標: サンプル単位平均のprecision、recall、F1
- baseline: TF-IDF cosine、Sentence-BERT cosine、BERT cross-encoder、TF-IDF OvR、Sentence-BERT OvR

## 期待される出力

前処理の出力:

- `results/hierarchy/output/<cluster_tag>/category_hierarchy_<cluster_tag>.json`
- `results/hierarchy/output/<cluster_tag>/test_data_*_with_hierarchy_<cluster_tag>.json`
- `results/hierarchy/output/<cluster_tag>/unknown_categories_<cluster_tag>.json`

学習の出力:

- `models/checkpoints/hierarchy/`
- `models/checkpoints/category_kfold/`
- `data/api_category_list/`

評価の出力:

- `results/hierarchy/hierarchy_existing_models*/`
- `results/hierarchy/hierarchy_existing_models*/test_data_manifest.json`
- `results/hierarchy/hierarchy_existing_models*/run_summary.json`
- `results/hierarchy/hierarchy_existing_models*/*_results.csv`

baselineの出力:

- `results/baselines/*.json`
- `results/baselines/*.csv`
- `results/baselines/<run_name>/metrics.json`
- `results/baselines/<run_name>/predictions.json`

## 引用


## ライセンス


