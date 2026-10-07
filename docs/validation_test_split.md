# 階層構造選定用 validation と最終 test の分離

現在の比較実験で使用している336件を、validation 100件とtest 236件に分割する。
validationはベースライン比較に採用する階層構造の選定に使い、選定した構成をtestで評価する。
元のprocessed test 632件ではなく、unknown category除外後の336件を分割対象とする。

## 固定した分割

| 項目 | 設定 |
|---|---|
| validation | 100件、階層構造の選定用 |
| test | 残り236件、選定後の評価用 |
| 学習データ | 既存3,652件（変更なし） |
| 抽出 | サンプル名をソートし、`random.Random(42).sample(..., 100)`で無作為抽出 |
| seed | 42 |
| 共通分割 | masked/unmasked、layer1、全階層パターンの44ファイル |
| 分割記録 | `data/splits/model_selection/manifest.json` |

サンプル名を基準に全形式へ同じ分割を適用する。レコードの本文、正解API、ID、階層ラベル、各元ファイル内の順序は保持する。
validation/test間は重複なし、和集合は元の336件と一致し、学習データとのサンプル名の重複も0件。
各ファイルのSHA-256を記録し、評価前に分割と元データの変更を検出する。

再生成・検証はリポジトリ直下から実行する（Python標準ライブラリのみ）。

```bash
python scripts/data_prep/create_validation_test_split.py
```

同じ入力での再実行は既存ファイルを変更しない。別のseedや件数を試して既存の分割を上書きすることはできない。
別実験として作る場合のみ`--output-dir`を変える。今回の選定ではseedと100件を固定し、評価値を見て分割を変更しない。

## 保存先

`data/splits/model_selection/{validation,test}/`以下に、元の`results/hierarchy/output/`と同じ相対パスを保存する。

- 共通baseline用（マスクあり）: `{validation,test}/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json`
- 共通baseline用（マスクなし）: `{validation,test}/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`
- 階層モデル用: `{validation,test}/by_pattern/<pattern>/<元のファイル名>.json`

元の336件や過去の結果は保持する。従来の条件別サブセットも336件由来なので、今回のtest236件として使用しない。

## validationで最良階層を選ぶ

まず対象と件数を確認する。

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked --split validation --layers 2,3,4 --dry-run
```

全16階層構成を同じ100件で評価する。

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked --split validation --layers 2,3,4
```

生成されたvalidation結果だけを使い、最終API推薦のサンプル単位F1の算術平均が最大の構成を選ぶ。
これは既存レポートの`Macro_F1平均`に相当し、APIラベル単位のmacro-F1とは異なる。
完全同点なら構成名の昇順で決める。

```bash
python scripts/analysis/select_hierarchy_from_validation.py \
  --results-dir results/hierarchy/hierarchy_existing_models_api_name_masked/validation
```

`selection.json`に選定構成とvalidation成績、分割の識別情報を保存する。
入力データの分割、CSV件数、行の対応を検証し、旧336件の結果やtest結果では選定しない。

## 選定後にtestで評価する

次の`SELECTED_PATTERN`を`selection.json`の`selected_pattern`へ置き換える。

```bash
python scripts/inference/run_all_existing_hierarchy_patterns.py \
  --dataset masked --split test --patterns SELECTED_PATTERN
```

階層ランナーの`--split`既定値は`test`。標準の結果保存先は`validation/`と`test/`で分かれる。
`--results-dir`を指定する場合も、分割ごとに異なるディレクトリを指定する。
旧336件の実験を再現する場合だけ`--split full`を明示する。
`--condition-2x2`との併用も`--split full`が必要。

## 他のbaselineも同じ236件で比較する

baseline共通ローダーの既定値は、マスクなしのtest236件。
`--eval-path` / `--test_path`や`EVALUATION_DATA_PATH`の明示指定は優先されるため、以前の336件のパスを指定しない。
masked階層モデルと比較するときは、baseline側にも同じmasked testを渡す。

```bash
python scripts/baselines/run_tfidf_ovr_baseline.py \
  --train_path data/processed/root_json/study_data_mashup_apis.json \
  --test_path data/splits/model_selection/test/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json \
  --model_type logistic --threshold 0.5 \
  --output_dir results/baselines/tfidf_ovr_logistic_test236_masked
```

TF-IDF/SBERT cosine、cross-encoderは同じパスを`--eval-path`へ渡す。
SBERT OvRは`--test_path`を使用する。

この100件は階層構造の選定用で、学習時のearly stopping用validationとは別。
OvRの`--valid_path`は閾値調整用であり、`--tune_threshold`時にはtrain+validで再学習するため、今回の100件を自動では渡さない。

## 実験記録上の扱い

元の336件では既に過去の構成比較を行っているため、今回は既存評価データの事後的な再分割として記録する。
新しいtest236件を「過去に一度も参照していないデータ」とは扱わない。
今後の選定ではvalidationだけで構成を固定し、testの成績を見て選び直さない。
