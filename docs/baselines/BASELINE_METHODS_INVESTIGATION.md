# Baseline Methods Investigation

作成日: 2026-06-14

## Scope

このメモは、ローカル環境で確認できた baseline 実験条件を、論文の Baseline Methods 節に使える形で整理したものです。

- コード変更、再学習、新規 LLM API 呼び出し、既存出力の上書きは行っていない。
- LLM baseline は別サーバで実行されていたため、このローカル環境の `MatsumotoProgram/output/...` は採用根拠から除外する。
- ここで確認済みとして扱うのは Single-layer BERT、TF-IDF Cosine、SBERT+OvR Logistic Regression、Cross-Encoder の4系統。
- 不明な点は不明として残す。

## Common Evaluation Setting

主要な非LLM full evaluation は、同じ 336 件の test data を使っている。

- Test data: `codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`
- Test records: 336
- True API mentions: 717
- Average true APIs per record: 2.134
- Ranking candidates: 909 APIs
- Metrics: sample-level average precision / recall / F1

根拠:

- `BASELINE_RESULTS_SUMMARY.md:16-25`
- `baseline_utils.py:19-27`
- `baseline_utils.py:168-195`

評価定義は各サンプルで `true_apis` と `predicted_apis` を set 化し、precision / recall / F1 を計算したあと、全サンプルで算術平均する。予測が空の場合の precision は 0。

## Results Confirmed Locally

| Baseline | Output / source | Records | Candidates / labels | Avg pred APIs | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Single-layer BERT | `shorter_desc/layer1_direct_api_results.csv` + `longer_desc/layer1_direct_api_results.csv` | 336 | 755 labels | 102.604 | 0.0197 | 0.6549 | 0.0376 |
| TF-IDF Cosine | `results/tfidf_threshold0_909api.json` | 336 | 909 APIs | 875.920 | 0.0024 | 0.9891 | 0.0048 |
| SBERT+OvR Logistic Regression | `results/sbert_ovr_logistic/` | 336 | 754 train labels | 3.051 | 0.3789 | 0.6120 | 0.4252 |
| Cross-Encoder | `results/cross_encoder_all909_threshold05.json` | 336 | 909 APIs | 11.429 | 0.1570 | 0.8428 | 0.2506 |

追加確認値:

- Single-layer BERT: min predicted APIs 45, max 628, zero-pred records 0.
- TF-IDF Cosine: min 202, max 909, zero-pred records 0.
- SBERT+OvR Logistic Regression: min 0, max 10, zero-pred records 10.
- Cross-Encoder: min 3, max 31, zero-pred records 0.

## Single-layer BERT

### Confirmed Facts

- Evaluation script: `mashup_dis_nosp_1layer_api.py`.
- Model architecture in evaluation script: `BertModel` followed by a linear multilabel classification layer.
- Base model path used at inference: `/home/nakagawa/lab-MLMLC/models/bert-base-uncased/`.
- Checkpoint path used at inference: `/home/nakagawa/lab-MLMLC/layer1/mashup_softfill_api_all_layer1_model_weights.pth`.
- Label list path used at inference: `/home/nakagawa/lab-MLMLC/api_category_list/layer1/softfill_inc_categorylist_layer1.json`.
- The label list has 755 labels, not 909.
- Tokenizer max length at prediction time is 512.
- Final CSV output uses labels whose sigmoid probability is `>= 0.5`.
- The result above is obtained by combining two condition-analysis outputs:
  - `shorter_desc`: 171 records.
  - `longer_desc`: 165 records.
  - Total: 336 records.

根拠:

- `mashup_dis_nosp_1layer_api.py:33-49`
- `mashup_dis_nosp_1layer_api.py:58-70`
- `mashup_dis_nosp_1layer_api.py:73-96`
- `mashup_dis_nosp_1layer_api.py:116-145`
- `mashup_dis_nosp_1layer_api.py:204-205`
- `mashup_dis_nosp_1layer_api.py:230-284`
- `codex/results/hierarchy_existing_models_condition_2x2/run_summary.json:3-28`
- `logs/nohup_layer1_20260219_171839.log:1471`

### Notes / Unknowns

- The checkpoint save log confirms that `/home/nakagawa/lab-MLMLC/layer1/mashup_softfill_api_all_layer1_model_weights.pth` was saved in this environment.
- However, the exact training script and launch command for that checkpoint were not confirmed from the available log. The log shows model structure and training progress, but not the command line.
- The script has an internal minor inconsistency: `predict()` first uses `probs > threshold`, but `run_model_and_get_results()` reconstructs the final recommendation list using `prob >= threshold`. The CSV output follows the latter.

## TF-IDF Cosine

### Confirmed Facts

- Script: `codex/baseline/run_tfidf_baseline.py`.
- No supervised training on mashup labels.
- API descriptions from `api_dataset_like_manifest.json` are vectorized.
- Requirement text is transformed by the same vectorizer.
- Scores are cosine similarities between requirement TF-IDF vectors and API-description TF-IDF vectors.
- Vectorizer settings recorded in the output:
  - `max_features`: null
  - `min_df`: 1
  - `ngram_range`: `[1, 1]`
  - `stop_words`: `none`
- Selection condition for the reported run:
  - `selection_mode`: `threshold`
  - `threshold`: `0.0`
  - `threshold_strict`: true
  - Therefore recommended APIs are those with `score > 0`.

根拠:

- `run_tfidf_baseline.py:21-39`
- `run_tfidf_baseline.py:57-69`
- `run_tfidf_baseline.py:81-93`
- `baseline_utils.py:198-243`
- `results/tfidf_threshold0_909api.json:1-28`

## SBERT+OvR Logistic Regression

### Confirmed Facts

- Script: `codex/baseline/run_sbert_ovr_baseline.py`.
- Feature extractor: fixed Sentence-BERT embeddings.
- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- Embedding dimension: 384.
- Embeddings are L2-normalized unless `--no_normalize_embeddings` is set. The saved config confirms normalization was enabled.
- Classifier: One-vs-Rest Logistic Regression.
- Logistic regression settings in saved config:
  - `C`: 1.0
  - `max_iter`: 1000
  - `class_weight`: `balanced`
  - `solver`: `liblinear`
  - `n_jobs`: -1
- Score type: `predict_proba`.
- Threshold: 0.5.
- Train/test were explicit files, not a random split.
- Train loaded: 3651 records.
- Test loaded: 336 records.
- Number of labels: 754 train labels.
- Test set contains 91 API label types unseen in train, corresponding to 96 mentions and 54 records. These are not predictable by this OvR model.

根拠:

- `run_sbert_ovr_baseline.py:42-114`
- `run_sbert_ovr_baseline.py:164-190`
- `run_sbert_ovr_baseline.py:216-232`
- `run_sbert_ovr_baseline.py:399-449`
- `results/sbert_ovr_logistic/config.json:1-55`
- `results/sbert_ovr_logistic/metrics.json:1-22`
- `BASELINE_RESULTS_SUMMARY.md:25`

## Cross-Encoder

### Confirmed Facts

- Script: `codex/baseline/run_cross_encoder_baseline.py`.
- Base model: `bert-base-uncased`.
- Architecture: `AutoModelForSequenceClassification(..., num_labels=1)` used as a binary cross-encoder over `(requirement text, API description)` pairs.
- Training pair construction in code:
  - Positive pair: mashup requirement text paired with a used API description.
  - Negative pair: same requirement text paired with randomly sampled unused API descriptions.
  - Negative ratio: 4 negatives per positive.
  - Seed: 42 by default.
- Training defaults in script:
  - epochs: 1
  - learning rate: 2e-5
  - train batch size: 16
  - eval batch size: 64
  - tokenizer max length: 256
- Reported threshold result:
  - Output: `results/cross_encoder_all909_threshold05.json`
  - Model dir: `models/cross_encoder_bert_base_uncased_909api_all`
  - `trained_this_run`: false
  - `candidate_method`: `all`
  - `candidate_k`: 0
  - All 909 API candidates were scored for each test record.
  - Score: sigmoid probability from the cross-encoder.
  - Selection: `score >= 0.5`.

根拠:

- `run_cross_encoder_baseline.py:31-74`
- `run_cross_encoder_baseline.py:85-123`
- `run_cross_encoder_baseline.py:139-198`
- `run_cross_encoder_baseline.py:201-217`
- `run_cross_encoder_baseline.py:326-357`
- `run_cross_encoder_baseline.py:360-418`
- `results/cross_encoder_all909_threshold05.json:1-26`
- `BASELINE_RESULTS_SUMMARY.md:62-80`

## LLM Baselines

LLM baselines are intentionally not summarized here.

Reason: the user confirmed that the LLM runs were executed on another server. Therefore, this local workspace cannot provide verified run conditions, logs, prompts, model versions, decoding settings, or result files for the LLM baselines. Any local `MatsumotoProgram/output/...` files should be treated as non-authoritative for the final paper table unless separately matched to the external server run.

Status:

- Zero-shot: unknown in this local investigation.
- Zero-shot-CoT: unknown in this local investigation.
- Few-shot: unknown in this local investigation.
- Few-shot-CoT: unknown in this local investigation.
- Zero-shot-infe: unknown in this local investigation.
- Few-shot-infe: unknown in this local investigation.
- Plan-and-Solve: unknown in this local investigation.

## Fairness / Leakage Notes

- TF-IDF Cosine, SBERT+OvR, and Cross-Encoder results above are evaluated on the same 336-record test file.
- Single-layer BERT result is not from a single 336-row output file. It is reconstructed by combining `shorter_desc` and `longer_desc`, which partition the 336 records by description length.
- TF-IDF Cosine uses the 909 API catalog descriptions as candidate documents and does not use test labels for training.
- SBERT+OvR uses only training labels as output classes. Because 91 test API label types are unseen in train, recall is capped for those labels.
- Cross-Encoder threshold run reuses a saved model. The threshold result file itself records `trained_this_run: false`; training conditions are inferred from the training script and baseline summary, not from that threshold output file alone.
- LLM fairness and leakage cannot be assessed from this local environment because the authoritative LLM runs are on another server.

## English Summary for Paper Draft

- We evaluated non-LLM baselines on a common test set of 336 mashups containing 717 ground-truth API mentions.
- TF-IDF Cosine represents API descriptions with unigram TF-IDF and recommends all APIs with cosine similarity greater than zero.
- SBERT+OvR uses fixed `all-MiniLM-L6-v2` sentence embeddings and trains a one-vs-rest logistic regression classifier over 754 API labels observed in the training set.
- The Cross-Encoder baseline fine-tunes a BERT binary classifier over requirement/API-description pairs and scores all 909 API candidates at evaluation time.
- The Single-layer BERT baseline loads a multilabel BERT classifier checkpoint and predicts API labels with a global sigmoid threshold of 0.5.
- LLM baselines were executed on a separate server and were not verified in this local investigation.

