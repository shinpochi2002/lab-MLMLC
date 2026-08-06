# Path Migration Plan

このドキュメントは、ディレクトリ整理後に動かなくなる旧パス参照を修正するための作業方針です。

## 目的

- 旧ディレクトリ直指定による `FileNotFoundError` を減らす。
- `/home/nakagawa/...` のような個人環境依存を減らし、cloneした人でも構成を追いやすくする。
- データ、モデル、結果の置き場所を `docs/FOLDER_REORGANIZATION_PLAN.md` と揃える。

## 基本ルール

1. 新しいコードでは `scripts/common/project_paths.py` の定義を使う。
2. 入力データは `data/`、モデル重みは `models/`、出力は `results/` に分ける。
3. 旧 `codex/` パスは残さない。
4. スクリプトから生成されるCSV、log、jsonlは原則 `results/` 配下へ出す。
5. 大容量モデルや学習済み重みはGit管理しないが、ローカル実行時の参照先は `models/` に統一する。
6. 既存の環境変数指定がある場合は残し、デフォルト値だけ新パスへ変える。
7. 一括置換後は `python -m py_compile` と旧パス検索で確認する。

## 旧パスから新パスへの対応

| 旧パス | 新パス |
| --- | --- |
| `api_category_list/` | `data/api_category_list/` |
| `studydata/` | `data/processed/studydata/` |
| `mashup_data/` | `data/raw/mashup_data/` |
| `new_api_data/` | `data/raw/new_api_data/` |
| `softfill_mashup_data_new_api/` | `data/processed/softfill_mashup_data_new_api/` |
| `softfill_mashup_data_new_api_inclu_nomash/` | `data/processed/softfill_mashup_data_new_api_inclu_nomash/` |
| `out_by_category*` | `data/splits/by_category/` |
| `train_by_category*` | `data/splits/train/` |
| `test_by_category*` | `data/splits/test/` |
| `study_category*_cv/` | `data/cv/` |
| `cluster/` | `data/cluster_definitions/` |
| `models/bert-base-uncased/` | `models/base/bert-base-uncased/` |
| `layer1/` | `models/checkpoints/hierarchy/layer1/` |
| `layer2_*/` | `models/checkpoints/hierarchy/layer2/layer2_*/` |
| `layer3_*/` | `models/checkpoints/hierarchy/layer3/layer3_*/` |
| `layer4_*/` | `models/checkpoints/hierarchy/layer4/layer4_*/` |
| `study_category_inc_nomashup_model_new_kfold/` | `models/checkpoints/category_kfold/` |
| `GoogleNews-vectors-negative300.bin` | `models/embeddings/GoogleNews-vectors-negative300.bin` |
| `codex/output/` | `results/hierarchy/output/` または `data/test_subsets/` |
| `codex/results/` | `results/hierarchy/` |
| `csvs/` | `results/tables/` |
| `logs/` | `results/logs/` |

## 修正優先度

1. `scripts/inference/`
2. `scripts/create_study_data_and_model/`
3. `scripts/data_prep/`
4. `scripts/utils/`
5. `scripts/analysis/` と `analysis/`
6. `scripts/experiments/`
7. `scripts/baselines/`

## 検証

```bash
python -m py_compile scripts/common/project_paths.py
python -m py_compile scripts/inference/*.py
rg -n '/home/nakagawa|codex/|api_category_list/|studydata/|csvs/|logs/' scripts analysis
```
