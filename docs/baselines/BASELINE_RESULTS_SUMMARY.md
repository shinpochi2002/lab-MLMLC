# Historical Baseline Results Summary (336 samples)

作成日: 2026-06-14
更新日: 2026-06-14

本書は2026年6月の旧336件・API名マスクなしの評価記録です。数値は当時の結果を保持しています。現在のtest236件での成績ではありません。現在の手順と事後分割の制約は[validation/test分割](../validation_test_split.md)を参照してください。以下の出力名は`results/baselines/`からの相対パスです。

## 概要

`scripts/baselines/` では、API 推薦タスクに対して以下の baseline が実行されています。

- TF-IDF cosine similarity による Top-k / threshold 推薦
- Sentence-BERT cosine similarity による Top-k 推薦
- BERT cross-encoder の fine-tuning + 全 909 API scoring
- TF-IDF + One-vs-Rest multi-label classifier
- Sentence-BERT embedding + One-vs-Rest multi-label classifier

主要な full evaluation はすべて同じ test data を使っています。

- Test data: `results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`
- Test records: 336
- Test true API mentions: 717
- Average true APIs per record: 2.134
- API catalog candidates for ranking baselines: 909
- Metrics: sample-level average precision / recall / F1

教師あり OvR 系だけは train data に出現したラベルを候補にするため、候補 API 数は 754 です。test には train に出ていない API が 91 種、96 mentions、54 records に含まれるため、OvR 系はそれらを原理的に予測できません。

## 実行内容

### TF-IDF cosine Top-k / threshold

- Script: `run_tfidf_baseline.py`
- Train: なし
- Scoring: requirement text と API description の TF-IDF cosine similarity
- API candidates: `data/processed/root_json/api_dataset_like_manifest.json` の 909 API
- Vectorizer: unigram, `min_df=1`, `stop_words=none`
- Selection: Top-1 / Top-5 / Top-10 / threshold 0 strict
- `threshold 0 strict` は `score > 0` の API を全て推薦します。
- Outputs:
  - `tfidf_top1_909api.json`
  - `tfidf_top5_909api.json`
  - `tfidf_top10_909api.json`
  - `tfidf_top10_default.json`
  - `tfidf_threshold0_909api.json`

### Sentence-BERT cosine Top-k

- Script: `run_sbert_baseline.py`
- Train: なし
- Scoring: requirement embedding と API description embedding の cosine similarity
- API candidates: 909
- Models:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-mpnet-base-v2`
- Selection: Top-1 / Top-5 / Top-10
- Outputs:
  - `sbert_top5_909api.json`
  - `sbert_top10_909api.json`
  - `sbert_top1_909api_mpnet.json`
  - `sbert_top5_909api_mpnet.json`
  - `sbert_top10_909api_mpnet.json`

### BERT cross-encoder

- Script: `run_cross_encoder_baseline.py`
- Base model: `bert-base-uncased`
- Train data: `data/processed/root_json/study_data_mashup_apis.json`
- Fine-tuning: positive pair = used API, negative pair = unused API sampled from 909 candidates
- Negative ratio: 4
- Epochs: 1
- Saved model: `models/baselines/cross_encoder_bert_base_uncased_909api_all`
- Evaluation: all 909 APIs are scored per test record (`candidate_method=all`, `candidate_k=0`)
- Score: sigmoid probability from cross-encoder
- Selection:
  - Top-10 run trained the model in that run
  - Top-5 / Top-1 / threshold 0.5 reused the saved model
- Outputs:
  - `cross_encoder_all909_top1.json`
  - `cross_encoder_all909_top5.json`
  - `cross_encoder_all909_top10.json`
  - `cross_encoder_all909_threshold05.json`

### TF-IDF + OvR classifier

- Script: `run_tfidf_ovr_baseline.py`
- Train data: `data/processed/root_json/study_data_mashup_apis.json`
- Train records loaded: 3651
- Test records: 336
- Labels: 754 train labels
- Feature: TF-IDF
- Classifier:
  - `OneVsRestClassifier(LogisticRegression)`
  - `OneVsRestClassifier(LinearSVC)`
- Selection: global threshold
- Outputs:
  - `tfidf_ovr_logistic/`
  - `tfidf_ovr_logistic_unigram/`
  - `tfidf_ovr_svm_unigram/`

### Sentence-BERT + OvR classifier

- Script: `run_sbert_ovr_baseline.py`
- Train data: `data/processed/root_json/study_data_mashup_apis.json`
- Train records loaded: 3651
- Test records: 336
- Labels: 754 train labels
- Feature: fixed `sentence-transformers/all-MiniLM-L6-v2` embedding, 384 dimensions
- Classifier: `OneVsRestClassifier(LogisticRegression)`
- Selection: threshold 0.5
- Output: `sbert_ovr_logistic/`

## Results

Full evaluation runs sorted by F1:

| Rank | Run | Method | Selection | Candidates | Avg pred APIs | Precision | Recall | F1 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `cross_encoder_all909_top1.json` | Cross Encoder | Top-1 | 909 | 1.000 | 0.7708 | 0.5072 | 0.5759 |
| 2 | `tfidf_ovr_logistic_unigram/` | TF-IDF OvR Logistic | threshold 0.5 | 754 | 1.241 | 0.6292 | 0.5549 | 0.5538 |
| 3 | `tfidf_ovr_logistic/` | TF-IDF OvR Logistic | threshold 0.5 | 754 | 1.060 | 0.6405 | 0.5237 | 0.5422 |
| 4 | `tfidf_ovr_svm_unigram/` | TF-IDF OvR SVM | threshold 0.0 | 754 | 0.994 | 0.6263 | 0.5092 | 0.5305 |
| 5 | `sbert_ovr_logistic/` | SBERT OvR Logistic | threshold 0.5 | 754 | 3.051 | 0.3789 | 0.6120 | 0.4252 |
| 6 | `cross_encoder_all909_top5.json` | Cross Encoder | Top-5 | 909 | 5.000 | 0.2732 | 0.7622 | 0.3752 |
| 7 | `tfidf_top1_909api.json` | TF-IDF cosine | Top-1 | 909 | 1.000 | 0.3571 | 0.2311 | 0.2662 |
| 8 | `cross_encoder_all909_threshold05.json` | Cross Encoder | threshold 0.5 | 909 | 11.429 | 0.1570 | 0.8428 | 0.2506 |
| 9 | `cross_encoder_all909_top10.json` | Cross Encoder | Top-10 | 909 | 10.000 | 0.1512 | 0.8086 | 0.2414 |
| 10 | `tfidf_top5_909api.json` | TF-IDF cosine | Top-5 | 909 | 5.000 | 0.1506 | 0.4418 | 0.2109 |
| 11 | `sbert_top1_909api_mpnet.json` | SBERT cosine, mpnet | Top-1 | 909 | 1.000 | 0.2411 | 0.1582 | 0.1812 |
| 12 | `sbert_top5_909api_mpnet.json` | SBERT cosine, mpnet | Top-5 | 909 | 5.000 | 0.1214 | 0.3725 | 0.1726 |
| 13 | `tfidf_top10_909api.json` | TF-IDF cosine | Top-10 | 909 | 10.000 | 0.0964 | 0.5387 | 0.1557 |
| 14 | `tfidf_top10_default.json` | TF-IDF cosine | Top-10 | 909 | 10.000 | 0.0964 | 0.5387 | 0.1557 |
| 15 | `sbert_top5_909api.json` | SBERT cosine, MiniLM | Top-5 | 909 | 5.000 | 0.0988 | 0.3003 | 0.1403 |
| 16 | `sbert_top10_909api_mpnet.json` | SBERT cosine, mpnet | Top-10 | 909 | 10.000 | 0.0777 | 0.4619 | 0.1271 |
| 17 | `sbert_top10_909api.json` | SBERT cosine, MiniLM | Top-10 | 909 | 10.000 | 0.0664 | 0.3884 | 0.1083 |
| 18 | `tfidf_threshold0_909api.json` | TF-IDF cosine | threshold 0 strict | 909 | 875.920 | 0.0024 | 0.9891 | 0.0048 |

OvR runs also saved micro metrics:

| Run | Micro Precision | Micro Recall | Micro F1 | Zero-pred records |
|---|---:|---:|---:|---:|
| `tfidf_ovr_logistic_unigram/` | 0.7434 | 0.4324 | 0.5467 | 72 |
| `tfidf_ovr_logistic/` | 0.8146 | 0.4045 | 0.5405 | 81 |
| `tfidf_ovr_svm_unigram/` | 0.8204 | 0.3821 | 0.5214 | 87 |
| `sbert_ovr_logistic/` | 0.3366 | 0.4812 | 0.3961 | 10 |

## 読み取り

- 最も高い sample-level F1 は `cross_encoder_all909_top1.json` の 0.5759 でした。Top-1 なので平均推薦数が 1 件に固定され、precision が 0.7708 と高く出ています。
- 教師あり OvR 系では `tfidf_ovr_logistic_unigram/` が F1 0.5538 で最良です。bigram ありの `tfidf_ovr_logistic/` より recall が高く、F1 も少し上です。
- `tfidf_ovr_logistic/` と `tfidf_ovr_svm_unigram/` は micro precision が 0.8 前後と高い一方、平均推薦数が約 1 件で、zero-pred records も多めです。
- `sbert_ovr_logistic/` は recall 0.6120 と OvR 系で最も高いですが、平均 3.05 件推薦するため precision が下がり、F1 は 0.4252 です。
- Cross Encoder の Top-5 / Top-10 / threshold 0.5 は recall が高い一方で、推薦数が増えるほど precision が大きく落ちます。特に threshold 0.5 は平均 11.43 件推薦し、recall 0.8428、F1 0.2506 でした。
- `tfidf_threshold0_909api.json` は cosine 類似度が非ゼロの API を全て推薦する実行です。平均 875.92 件を推薦するため recall は 0.9891 と非常に高いですが、precision は 0.0024、F1 は 0.0048 まで落ちます。実用比較用というより、閾値を極端に緩くしたときの上限 recall 確認に近い結果です。
- 単純な cosine ranking では TF-IDF が SBERT より強く、SBERT は MiniLM より mpnet の方が改善しています。ただし mpnet Top-5 の F1 0.1726 でも TF-IDF Top-5 の F1 0.2109 には届いていません。
- `tfidf_top10_default.json` と `tfidf_top10_909api.json` は同じ設定・同じ結果です。

## Smoke Runs

`smoke/` は full evaluation ではなく動作確認用の少数サンプル実行です。

| Run | Records | Selection | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|
| `smoke_tfidf_top5.json` | 3 | Top-5 | 0.2000 | 0.2143 | 0.2037 |
| `smoke_tfidf_threshold0_strict.json` | 3 | threshold 0 strict | 0.0059 | 1.0000 | 0.0117 |
| `smoke_tfidf_threshold05.json` | 3 | threshold 0.5 | 0.0000 | 0.0000 | 0.0000 |
| `smoke_cross_threshold05.json` | 1 | threshold 0.5 | 0.0000 | 0.0000 | 0.0000 |

## まとめ

比較用の主 baseline としては、目的によって次を代表値にすると見やすいです。

- 高 precision / 1件推薦の代表: `cross_encoder_all909_top1.json`、F1 0.5759
- 教師あり multi-label baseline の代表: `tfidf_ovr_logistic_unigram/`、F1 0.5538
- Top-5 推薦の代表: `cross_encoder_all909_top5.json`、F1 0.3752
- Top-10 推薦の代表: `cross_encoder_all909_top10.json`、F1 0.2414
- 単純 cosine baseline の代表: `tfidf_top5_909api.json` または `tfidf_top10_909api.json`
- Recall upper-bound check: `tfidf_threshold0_909api.json`、Recall 0.9891、F1 0.0048

LLM 系の推薦数が可変または threshold 的に出る場合は、Top-k baseline だけでなく、平均推薦数が近い OvR や threshold run と比較するのが妥当です。Top-1 の F1 が最も高い点は重要ですが、推薦数が 1 件に固定されているため、Top-5/Top-10 や可変推薦とは別枠で読むのが安全です。
