# API Recommendation Baselines

このディレクトリには、既存の API 推薦実験と同じサンプル単位の `precision / recall / F1` 定義で比較する baseline を置いています。

## 入出力

- デフォルト test data: `codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`
- デフォルト train data: `study_data_mashup_apis.json`
- API 説明文 catalog: `api_dataset_like_manifest.json`
- 出力先: `codex/baseline/results/`

`api_dataset_like_manifest.json` は 909 件の API catalog です。この 909 件は `study_data_mashup_apis.json` と `test_data_mashup_apis.json` に実際に出現する API の union と一致します。指定 test data 自体に出現する unique API は 244 件ですが、候補集合は「filtered mashup 全体で使われた API」の閉世界として 909 件を使います。

各 JSON / CSV には、要件文ごとに `mashup_id`, `mashup_name`, `true_apis`, `recommended_apis`, `scores`, `precision`, `recall`, `f1` を保存します。JSON には全体平均 `average_metrics` も含めます。

## 推薦の切り方

デフォルトは従来通り `--selection-mode topk --top-k 10` です。提案手法のように閾値 0.5 で切る場合は、各 baseline に次を付けます。

```bash
--selection-mode threshold --threshold 0.5
```

スコアの意味は手法ごとに違います。TF-IDF は `tfidf_cosine_similarity`、Sentence-BERT は `sbert_cosine_similarity`、cross-encoder は sigmoid 後の `cross_encoder_sigmoid_probability` です。提案手法の閾値 0.5 と一番直接対応するのは cross-encoder です。TF-IDF/SBERT の 0.5 は cosine 類似度に対する閾値として扱います。

## train data の扱い

TF-IDF cosine、Sentence-BERT cosine、ServiceBERT baseline は教師あり学習をしないため、train data は使いません。API catalog の説明文をベクトル化し、test 要件文との cosine 類似度で ranking します。

`run_tfidf_ovr_baseline.py` は教師あり multi-label baseline です。`study_data_mashup_apis.json` などの train data から TF-IDF 特徴量と API multi-hot ラベルを作り、`OneVsRestClassifier(LogisticRegression)` または `OneVsRestClassifier(LinearSVC)` を学習します。予測 API 集合は Top-k ではなく global threshold で決めます。Logistic Regression は `predict_proba`、Linear SVM は `decision_function` を使い、SVM score は確率化しません。

`run_sbert_ovr_baseline.py` は、上の教師あり multi-label baseline のベクトル化部分だけを Sentence-BERT embedding に置き換えたものです。Sentence-BERT 自体は fine-tuning せず、train data の要件文を固定 embedding に変換してから `OneVsRestClassifier(LogisticRegression)` または `OneVsRestClassifier(LinearSVC)` を学習します。

BERT cross-encoder baseline は `study_data_mashup_apis.json` を train data として使います。正例は mashup が実際に使った API、負例は 909 件 API catalog の未使用 API から sampling します。test data の正解ペアは fine-tuning には使いません。

## 依存関係

```bash
pip install -r codex/baseline/requirements-baseline.txt
```

この環境では `/home/nakagawa/venvs/swallow-py312/bin/python` に `scikit-learn`, `torch`, `transformers`, `sentence-transformers` が入っています。Sentence-BERT baseline を実行する場合はこの venv を使うのが確実です。

## 実行例

TF-IDF cosine Top-k baseline:

```bash
/home/nakagawa/venvs/swallow-py312/bin/python codex/baseline/run_tfidf_baseline.py --top-k 10
```

TF-IDF cosine threshold-0 baseline:

```bash
/home/nakagawa/venvs/swallow-py312/bin/python codex/baseline/run_tfidf_baseline.py \
  --selection-mode threshold \
  --threshold 0 \
  --threshold-strict \
  --run-name tfidf_threshold0_909api
```

TF-IDF + One-vs-Rest Logistic Regression baseline:

```bash
python codex/baseline/run_tfidf_ovr_baseline.py \
  --train_path study_data_mashup_apis.json \
  --test_path codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
  --text_key description \
  --label_key apis \
  --model_type logistic \
  --threshold 0.5 \
  --output_dir codex/baseline/results/tfidf_ovr_logistic
```

TF-IDF + One-vs-Rest Logistic Regression baseline with threshold tuning:

```bash
python codex/baseline/run_tfidf_ovr_baseline.py \
  --train_path study_data_mashup_apis.json \
  --test_path codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
  --model_type logistic \
  --tune_threshold \
  --output_dir codex/baseline/results/tfidf_ovr_logistic_tuned
```

TF-IDF + One-vs-Rest Linear SVM baseline:

```bash
python codex/baseline/run_tfidf_ovr_baseline.py \
  --train_path study_data_mashup_apis.json \
  --test_path codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
  --model_type svm \
  --threshold 0.0 \
  --output_dir codex/baseline/results/tfidf_ovr_svm
```

`run_tfidf_ovr_baseline.py` は `metrics.json`, `predictions.json`, `labels.json`, `config.json`, `model.joblib`, `vectorizer.joblib` を `--output_dir` に保存します。`--data_path` だけを指定した場合は 80/20 の random split を使います。

Sentence-BERT baseline:

```bash
python codex/baseline/run_sbert_baseline.py \
  --top-k 10 \
  --model-name sentence-transformers/all-MiniLM-L6-v2
```

Sentence-BERT + One-vs-Rest Logistic Regression baseline:

```bash
python codex/baseline/run_sbert_ovr_baseline.py \
  --train_path study_data_mashup_apis.json \
  --test_path codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
  --text_key description \
  --label_key apis \
  --model_type logistic \
  --threshold 0.5 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_dir codex/baseline/results/sbert_ovr_logistic
```

Sentence-BERT + One-vs-Rest Logistic Regression baseline with threshold tuning:

```bash
python codex/baseline/run_sbert_ovr_baseline.py \
  --train_path study_data_mashup_apis.json \
  --test_path codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json \
  --model_type logistic \
  --tune_threshold \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_dir codex/baseline/results/sbert_ovr_logistic_tuned
```

`run_sbert_ovr_baseline.py` は `metrics.json`, `predictions.json`, `labels.json`, `config.json`, `encoder_config.json`, `model.joblib` を `--output_dir` に保存します。`model.joblib` は One-vs-Rest 分類器だけを保存し、Sentence-BERT の重みは保存しません。再現には `encoder_config.json` の `model_name` を使います。

ServiceBERT / Hugging Face encoder baseline:

```bash
python codex/baseline/servicebert/run_servicebert_baseline.py \
  --top-k 10 \
  --model-name /path/to/servicebert-checkpoint \
  --run-name servicebert_top10_909api
```

BERT cross-encoder fine-tuning + TF-IDF candidate rerank:

```bash
python codex/baseline/run_cross_encoder_baseline.py \
  --train \
  --top-k 10 \
  --candidate-method tfidf \
  --candidate-k 200 \
  --negative-ratio 4 \
  --epochs 1 \
  --model-name bert-base-uncased
```

学習済み cross-encoder を再利用して評価だけ行う場合:

```bash
python codex/baseline/run_cross_encoder_baseline.py \
  --top-k 10 \
  --candidate-method tfidf \
  --candidate-k 200 \
  --model-dir codex/baseline/models/cross_encoder_bert_base_uncased
```

## よく使うオプション

- `--top-k`: `--selection-mode topk` のときの推薦 API 数を変更します。
- `--selection-mode threshold --threshold 0.5`: score が 0.5 以上の API を全て推薦します。
- `run_tfidf_baseline.py --threshold-strict`: TF-IDF cosine の threshold 推薦で `score > threshold` を使います。`--threshold 0 --threshold-strict` なら、類似度が非ゼロの API だけを推薦します。
- `--eval-path`: 既存提案手法や LLM 評価と同じ test JSON を明示できます。
- `--api-catalog-path`: API 説明文 catalog を差し替えます。
- `--train-path`: cross-encoder の fine-tuning 用、または TF-IDF / Sentence-BERT + One-vs-Rest の教師あり学習用 mashup JSON を差し替えます。TF-IDF cosine / Sentence-BERT cosine / ServiceBERT では使いません。
- `--output-dir`: 結果 JSON / CSV の保存先を変更します。
- `--limit`: 少数サンプルで smoke test します。
- `--run-name`: 出力ファイル名の stem を指定します。
- `--candidate-k 0`: cross-encoder で全 API を scoring します。ただし計算量は大きくなります。
- `--tune_threshold`: TF-IDF / Sentence-BERT + One-vs-Rest baseline で validation set 上の macro F1 が最大になる global threshold を選びます。
- `--save_all_scores`: TF-IDF / Sentence-BERT + One-vs-Rest baseline の `predictions.json` に全 API score を保存します。指定しない場合は正解 API と予測 API の score のみ保存します。
