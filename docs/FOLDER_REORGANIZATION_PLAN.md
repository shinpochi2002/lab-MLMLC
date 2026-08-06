# Folder Reorganization Plan

このドキュメントは、`lab-MLMLC` を他人にも使いやすい構成へ整理するための方針と移動候補表です。
既存ファイルの中身は書き換えず、まずは「どこに置くべきか」を決めるための作業メモとして使います。

## 目的

- 初めて見る人が、コード・データ・モデル・結果をすぐ探せるようにする。
- ルート直下に実験生成物や一時ファイルが増えないようにする。
- 今後Codex等でリポジトリ全体を扱う前提にし、`codex/` という専用作業フォルダは残さない。
- 既存の実行環境をなるべく壊さず、必要に応じて旧パス互換を残す。

## 基本ルール

1. 既存ファイルの中身は変更しない。
2. 移動前に `git status --short --ignored` を確認する。
3. Git管理中のファイルは `git mv` で移動する。
4. 未追跡ファイルやignore対象の大容量ファイルは通常の `mv` で移動する。
5. 大容量モデル・チェックポイントは最後に移動する。
6. ハードコードされたパスがあるため、移動直後は必要に応じて旧パスにシンボリックリンクを置く。
7. 新しいトップレベルディレクトリは原則作らない。追加物はこのドキュメントの分類に従って置く。
8. 新しいデータや結果を追加したら、READMEまたはdocs内の対応表も更新する。

## Git管理ルール

| 種類 | Git管理 | 理由 |
| --- | --- | --- |
| Pythonコード、Notebook、Markdown | 管理する | 再利用・説明・再現に必要。 |
| 小さな固定JSON、APIカテゴリ定義、設定表 | 管理する | 他人がclone後にすぐ使えるようにする。 |
| 学習済みモデル重み、BERT本体、word2vec等 | 管理しない | 大容量で再取得/再生成対象。 |
| ログ、CSV結果、中間出力 | 原則管理しない | 実験ごとに増える生成物。最終結果だけ別途判断。 |
| 図表、PDF、最終サマリ | 要判断 | 論文・発表で使う最終版は管理候補。中間版は管理しない。 |
| 個人環境ファイル | 管理しない | `.envrc`, `.lesshst`, `.selected_editor`, `venv_active.txt` など。 |

## 目標トップレベル構成

```text
lab-MLMLC/
  README.md
  docs/
  scripts/
  notebooks/
  data/
  models/
  results/
  analysis/
  tools/
  archive/
```

## 目標ディレクトリの役割

| ディレクトリ | 役割 | 置くもの |
| --- | --- | --- |
| `docs/` | 説明・整理方針・再現手順 | README補助、移動計画、データ説明、実験メモ |
| `scripts/` | 実行コード | データ作成、推論、評価、集計、ユーティリティ |
| `notebooks/` | Notebook | `.ipynb` |
| `data/` | Git管理候補のデータ | raw, processed, splits, cv, APIカテゴリ定義 |
| `models/` | Git管理しないモデル置き場 | BERT本体、学習済み重み、埋め込み |
| `results/` | Git管理しない結果置き場 | CSV, logs, figures, summaries, experiment outputs |
| `analysis/` | 分析コードと分析用補助資料 | layer分析、図表生成コード |
| `tools/` | 補助ツール | server monitor等 |
| `archive/` | 退避・旧構成 | 古いコピー、由来不明、移行記録 |

## `data/` 配下ルール

```text
data/
  raw/                  # 元データ、外部から取得した加工前データ
  processed/            # 前処理済みデータ
  splits/               # train/test/category分割
  cv/                   # cross validation用データ
  api_category_list/    # APIカテゴリ/ラベル定義
  cluster_definitions/  # クラスタ/階層定義
  test_subsets/         # 固定テストサブセット
```

- モデル入力として必要な小さめのJSONは `data/` に置く。
- 同じ意味のデータが複数バージョンある場合は、名前に条件を残す。
- 再生成できる巨大な中間データは `data/processed/` に置いてもGit管理しない。

## `scripts/` 配下ルール

```text
scripts/
  data_prep/        # データ作成・前処理
  inference/        # 推論・評価実行
  analysis/         # 集計・分析
  baselines/        # ベースライン手法
  utils/            # リネーム、分割、変換などの補助
```

- 新しいPythonファイルはルート直下に置かない。
- 実験専用スクリプトでも、まずは目的別に `scripts/` へ置く。
- 汎用性が低く古いものは `archive/legacy_scripts/` に置く。

## `results/` 配下ルール

```text
results/
  tables/       # CSV/表形式結果
  figures/      # PNG/PDF等の図表
  logs/         # 実行ログ
  summaries/    # 最終結果サマリ
  baselines/    # baseline実験結果
  hierarchy/    # hierarchy model評価結果
```

- 原則として `results/` はGit管理しない。
- 論文・発表・READMEで参照する最終成果だけGit管理を検討する。
- 中間CSV、試行錯誤ログ、キャッシュはGit管理しない。

## `models/` 配下ルール

```text
models/
  base/         # bert-base-uncased等の外部モデル
  embeddings/   # GoogleNews word2vec等
  checkpoints/
    hierarchy/
    category_kfold/
```

- モデル重みはGit管理しない。
- どのモデルが必要かはREADMEまたは `docs/MODELS.md` に記載する。
- ダウンロード元・生成方法・必要容量をドキュメント化する。

## トップレベルフォルダ移動候補

| 現在の場所 | 移動先候補 | Git方針 | メモ |
| --- | --- | --- | --- |
| `api_category_list/` | `data/api_category_list/` | track候補 | コードから参照される固定データ。ignoreから外した。 |
| `mashup_data/` | `data/raw/mashup_data/` | track候補 | 元データ系。 |
| `new_api_data/` | `data/raw/new_api_data/` | track候補 | 追加APIデータ。 |
| `softfill_mashup_data_new_api/` | `data/processed/softfill_mashup_data_new_api/` | track候補 | 前処理済みデータ。 |
| `softfill_mashup_data_new_api_inclu_nomash/` | `data/processed/softfill_mashup_data_new_api_inclu_nomash/` | track候補 | no-mashupを含む前処理済みデータ。 |
| `studydata/` | `data/processed/studydata/` | 要判断 | 大きめ。全体を残すか最終版だけ残すか決める。 |
| `out_by_category*/` | `data/splits/by_category/` | track候補 | カテゴリ別分割データ。 |
| `train_by_category*/` | `data/splits/train/` | track候補 | 学習用分割。 |
| `test_by_category*/` | `data/splits/test/` | track候補 | テスト用分割。 |
| `study_category*_cv/` | `data/cv/` | track候補 | CV用データ。 |
| `cluster/` | `data/cluster_definitions/` | track候補 | 入力定義ならdata。結果ならresultsへ分離。 |
| `models/` | `models/base/` | ignore | 再取得可能な外部モデル。 |
| `layer1/` | `models/checkpoints/hierarchy/layer1/` | ignore | 学習済み重み。 |
| `layer2_*/` | `models/checkpoints/hierarchy/layer2/` | ignore | 学習済み重み。 |
| `layer3_*/` | `models/checkpoints/hierarchy/layer3/` | ignore | 学習済み重み。 |
| `layer4_*/` | `models/checkpoints/hierarchy/layer4/` | ignore | 学習済み重み。 |
| `study_category_inc_nomashup_model_new_kfold/` | `models/checkpoints/category_kfold/` | ignore | カテゴリ別k-foldモデル重み。 |
| `csvs/` | `results/tables/csvs/` | mixed | CSVは結果、`csv_process.py`は`scripts/utils/`候補。 |
| `logs/` | `results/logs/experiment_logs/` | ignore | 実行ログ。 |
| `layer_ana/` | `analysis/layer_ana/` | mixed | 解析コードと図表が混在。後で図表を`results/figures/`へ分離。 |
| `create_study_data_and_model/` | `scripts/create_study_data_and_model/` | mixed | コード中心。巨大binは`models/embeddings/`候補。 |
| `server_monitor/` | `tools/server_monitor/` | track候補 | 補助ツール。 |
| `Desktop/` | `archive/local/Desktop/` | ignore/要確認 | ローカル混入と思われる。 |
| `lab-MLMLC/` (入れ子) | `archive/legacy/nested-lab-MLMLC/` | ignore/要確認 | 同名入れ子。現状は`csvs/`のみ。 |
| `__pycache__/` | 移動しない | ignore | キャッシュ。削除するなら別途確認。 |

## `codex/` 解体ルール

今後はリポジトリ全体をCodex等で扱うため、`codex/` という専用フォルダは残さない。
中身は用途別に分解して移動する。

| 現在の場所 | 移動先候補 | Git方針 | メモ |
| --- | --- | --- | --- |
| `codex/scripts/` | `scripts/data_prep/` または `scripts/analysis/` | track | 内容ごとに分類。Codex名は残さない。 |
| `codex/run_all_existing_hierarchy_patterns.py` | `scripts/inference/run_all_existing_hierarchy_patterns.py` | track | hierarchy評価実行スクリプト。 |
| `codex/baseline/*.py` | `scripts/baselines/` | track | baseline実験コード。 |
| `codex/baseline/test_*.py` | `scripts/baselines/tests/` | track | baseline用テスト。 |
| `codex/baseline/requirements-baseline.txt` | `scripts/baselines/requirements-baseline.txt` | track | baseline依存関係。 |
| `codex/baseline/*.md` | `docs/baselines/` | track | baseline説明資料。 |
| `codex/baseline/api_name_masking*.json` | `data/processed/api_name_masking/` | track候補 | 固定辞書ならGit管理。生成ログならresultsへ。 |
| `codex/baseline/*mask*.csv` | `results/baselines/tables/` | ignore/要判断 | 結果CSV。最終版だけ管理候補。 |
| `codex/baseline/models/` | `models/base/baselines/` | ignore | baseline用外部モデル。 |
| `codex/baseline/results/` | `results/baselines/` | ignore | baseline結果。 |
| `codex/analysis_test_subset_conditions/*.md` | `docs/test_subset_conditions/` | track | 条件分析レポート。 |
| `codex/analysis_test_subset_conditions/*.json` | `data/test_subsets/condition_analysis/` | track/要判断 | 固定メタデータなら管理。バックアップはignore。 |
| `codex/analysis_test_subset_conditions/*.csv` | `results/tables/test_subset_conditions/` | ignore/要判断 | 分析表。最終表だけ管理候補。 |
| `codex/analysis_test_subset_conditions/subset_candidates/` | `data/test_subsets/candidates/` | track候補 | 固定サブセット候補。 |
| `codex/analysis_test_subset_conditions/test_subsets/` | `data/test_subsets/condition_analysis/` | track候補 | 固定テストサブセット。 |
| `codex/test_subsets/` | `data/test_subsets/` | track候補 | 固定テストサブセット。 |
| `codex/results/hierarchy_existing_models_*` | `results/hierarchy/` | ignore/要判断 | hierarchy評価結果。最終サマリだけ管理候補。 |
| `codex/output/` | `results/hierarchy/output/` | ignore | 実行出力。 |
| `codex/organized/` | `archive/organized_experiments/` | ignore/要判断 | 一時的に集約した実験構成。再利用するものだけ後で分解。 |
| `codex/__pycache__/`, `codex/**/__pycache__/` | 移動しない | ignore | キャッシュ。 |
| `codex/**/*.orig`, `codex/**/*.bak_*` | `archive/backups/` または移動しない | ignore | バックアップ。必要性を確認。 |

## トップレベルファイル移動候補

| 種類 | 移動先候補 | Git方針 | メモ |
| --- | --- | --- | --- |
| `mashup_dis_nosp*.py` | `scripts/inference/` | track | 推論・評価系。 |
| `create_*.py`, `cre_*.py` | `scripts/data_prep/` | track | データ作成系。 |
| `count_*.py`, `data_count.py`, `category_*count*.py` | `scripts/analysis/` | track | 集計・分析系。 |
| `file_rename*.py`, `split_data_cluster.py`, `cluster_to_json.py`, `test_split*.py` | `scripts/utils/` | track | 補助スクリプト。 |
| `ga_test.py`, `mlc_dis.py`, `mashup_data_mlc.py`, `new_mashup_data_mlc.py` | `scripts/experiments/` | track | 実験系。 |
| `original_bertmultilabel.ipynb` | `notebooks/` | track | Notebook。 |
| 主要な`.json`データ | `data/processed/root_json/` | track/要判断 | 固定データは管理。生成物はresultsへ。 |
| 結果サマリ`.json` | `results/summaries/` | track/要判断 | 最終結果なら管理候補。 |
| `*.log`, `rsync*.log` | `results/logs/root/` or `archive/sync_logs/` | ignore | ログは基本管理しない。 |
| `nas_files.txt`, `nas2_files.txt` | `archive/sync_logs/` | 要判断 | NAS移行記録ならarchive。 |
| `.lesshst`, `.selected_editor`, `.envrc`, `venv_active.txt` | 移動しない | ignore | 個人環境ファイル。 |

## 今後ファイルが増えた時の判断フロー

1. それは他人が再現に必要な固定情報か。
   - はい: `data/`, `scripts/`, `docs/` のどれかへ置く。
   - いいえ: 次へ。
2. それは実行すると再生成できる結果か。
   - はい: `results/` へ置き、原則Git管理しない。
   - いいえ: 次へ。
3. それは学習済みモデル・外部モデル・キャッシュか。
   - はい: `models/` へ置き、Git管理しない。
   - いいえ: 次へ。
4. それは一時的な作業ファイル・古いコピーか。
   - はい: `archive/` へ置くか、不要なら削除候補にする。
   - いいえ: READMEまたはこのドキュメントに用途を追記してから置く。

## 実施順序案

1. このドキュメントの移動候補表を確認し、保留対象を決める。
2. `README.md` と `docs/` を整備する。
3. 空の目標ディレクトリを作る。
4. 小さい固定データから移動する: `api_category_list/`, `mashup_data/`, `new_api_data/`。
5. 分割データを移動する: `out_by_category*`, `train_by_category*`, `test_by_category*`, `study_category*_cv/`。
6. `codex/` を用途別に解体する。
7. `layer_ana/`, `csvs/`, `logs/` を移動する。
8. トップレベルPython/Notebookを `scripts/` と `notebooks/` へ移動する。
9. 大容量モデルを最後に移動する。
10. 必要なら旧パス互換シンボリックリンクを置く。
11. `git status --short --ignored`、ファイル数、容量を確認する。
12. READMEに最終構成と使い方を反映する。

## 保留判断が必要なもの

- `studydata/`: 全体をGit管理するか、最終版だけ残すか。
- `cluster/`: 入力定義か実験結果か。
- `layer_ana/`: コードと図表をどこまで分離するか。
- `codex/organized/`: 再利用する構成か、単なる一時集約か。
- 旧パス互換: シンボリックリンクを置くか、READMEで新パスだけ案内するか。
