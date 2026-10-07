# 推論コードに基づく階層モデルのフォルダ命名ルール

## 基準と適用範囲

階層モデルのフォルダ名は、`scripts/inference/` の推論スクリプトが明示するモデル格納先を基準にする。
特に、一括ランナーが呼び出す `mashup_dis_nosp_{1,2,3,4}layer_api.py` のパス構築と、`run_all_existing_hierarchy_patterns.py` の引数設定を基準とする。
既存の構成名は保存済みモデルを識別する名前として保持する。各数字が現在の枝数・カテゴリ数に厳密に一致することは、命名の要件にしない。

- モデル参照の根拠: 推論コードのフォルダ指定と、そのフォルダに対応する重み・カテゴリリスト。
- 実構造の根拠: カテゴリリストと階層定義。構成名の数字から実構造を算出しない。
- E計算用表記: 分析コードで管理する別の対応表。モデルフォルダの改名・選択には使用しない。

このルールにおいて `layer3_4_4_8` は有効な既存識別名である。実際のカテゴリ数やE計算用表記を理由に `layer3_4_4_10` へ自動変更しない。

## 層ごとのモデル格納先

基点を `models/checkpoints/hierarchy/` とする。
`K` は `LAYER1_CATEGORY_NUM`、`T` は推論スクリプトへ渡す `TARGET` を表す。

| 推論スクリプト | コード内の旧フォルダ指定 | 現在の保存先 |
|---|---|---|
| `mashup_dis_nosp_1layer_api.py` | `layer1/` | `layer1/` |
| `mashup_dis_nosp_2layer_api.py` | `layer2_{K}/` | `layer2/layer2_{K}/` |
| `mashup_dis_nosp_3layer_api.py` | `layer3_{T}/` | `layer3/layer3_{T}/` |
| `mashup_dis_nosp_4layer_api.py` | `layer4_{T}/` | `layer4/layer4_{T}/` |

旧環境の絶対パス `<旧リポジトリルート>/layerN_.../` は `scripts/common/project_paths.py` の `legacy_path()` で現在の保存先へ変換する。
明示した構成フォルダを優先し、重みファイル名から別の構成フォルダを推測して置き換えない。

2層の `TARGET` は評価データ側の識別名であり、モデルフォルダ名と同一ではない。
例えば `TARGET=4-43`、`LAYER1_CATEGORY_NUM=4` のモデルフォルダは `layer2_4/`。
2層スクリプトを直接起動する場合は `LAYER1_CATEGORY_NUM` を明示する。
省略時には `TARGET` の末尾トークンを使うため、`TARGET=4-43` だけでは意図した4にならない。一括ランナーは両方を設定する。
3層・4層では、`LAYER1_CATEGORY_NUM` 省略時に `TARGET` の先頭トークンを使う。

## 構成名と重みファイルの枝ID

フォルダは構成全体、ファイルはその構成の各分類器を識別する。
同じ構成の上位・下位分類器は同じ構成フォルダに置く。

例: `TARGET=4_4_8`、`K=4`、第1層の予測 `l1_cat=3`、第2層の予測 `l2_cat=3-0` の場合。

```text
models/checkpoints/hierarchy/layer3/layer3_4_4_8/
  mashup_softfill_api_all_layer3_4_model_weights.pth
  mashup_softfill_api_all_layer3_4_3_model_weights.pth
  mashup_softfill_api_all_layer3_4_3_3_0_model_weights.pth
```

最後のファイル名は `K=4`、`l1_cat=3`、`l2_cat_norm=3_0` の連結。
`4_3_3_0` は構成名ではなく、`layer3_4_3_3/` というフォルダを導出する根拠にはならない。

| 構成 | ファイル名の数値部分（`layerN_` の後） |
|---|---|
| 2層 | 上位: `K`、下位: `K_l1` |
| 3層 | 上位: `K`、中位: `K_l1`、下位: `K_l1_l2norm` |
| 4層 | 上位から順に `K`、`K_l1`、`K_l2norm`、`K_l3norm` |

`l2norm` / `l3norm` は、経路形式のラベル中の `-` を `_` に置換した値。
3層ではファイル名に親IDが重複するが、既存の保存形式なので省略しない。
旧形式の `bigclusters` / `mid...clusters` / `small...clusters` 等は、既存の `legacy_path()` の明示的な対応を使用する。
構成フォルダのない古いファイル名の互換変換は、新しい命名の根拠にはしない。

## 既存モデルと評価データの対応

2026-09-30時点。一括ランナーの `discover_tasks(dataset="masked")` と実在フォルダを照合した。
モデルフォルダは `models/checkpoints/hierarchy/` からの相対パス。
評価フォルダは `results/hierarchy/output/` 以下の `layer1_direct_api/` または `by_pattern/` 以下の名前。

| タスク名 | TARGET | K | モデルフォルダ | 評価フォルダ |
|---|---|---:|---|---|
| `layer1_direct_api` | `—` | — | `layer1/` | `layer1_direct_api` |
| `layer2_4_43` | `4-43` | 4 | `layer2/layer2_4/` | `4-43` |
| `layer2_6_25` | `6-25` | 6 | `layer2/layer2_6/` | `6-25` |
| `layer2_8_19` | `8-19` | 8 | `layer2/layer2_8/` | `8-19` |
| `layer2_12_13` | `12_13` | 12 | `layer2/layer2_12/` | `12_13` |
| `layer2_15_10` | `15-10` | 15 | `layer2/layer2_15/` | `15-10` |
| `layer2_24_7` | `24-7` | 24 | `layer2/layer2_24/` | `24-7` |
| `layer2_33_4` | `33-4` | 33 | `layer2/layer2_33/` | `33-4` |
| `layer2_51_3` | `51-3` | 51 | `layer2/layer2_51/` | `51-3` |
| `layer3_4_4_8` | `4_4_8` | 4 | `layer3/layer3_4_4_8/` | `4cluster` |
| `layer3_8_4_4` | `8_4_4` | 8 | `layer3/layer3_8_4_4/` | `8_4_4` |
| `layer3_12_2_6` | `12_2_6` | 12 | `layer3/layer3_12_2_6/` | `12_2_6` |
| `layer3_12_4_3` | `12_4_3` | 12 | `layer3/layer3_12_4_3/` | `12_4_3` |
| `layer3_12_6_2` | `12_6_2` | 12 | `layer3/layer3_12_6_2/` | `12_6_2` |
| `layer3_16_4_3` | `16_4_3` | 16 | `layer3/layer3_16_4_3/` | `16cluster` |
| `layer4_8_3_2_3` | `8_3_2_3` | 8 | `layer4/layer4_8_3_2_3/` | `8_3_2_3` |
| `layer4_12_2_2_3` | `12_2_2_3` | 12 | `layer4/layer4_12_2_2_3/` | `12_2_3_2` |

評価フォルダの別名はそのまま保持する。例えば `4cluster` と `4_4_8`、`12_2_3_2` と `12_2_2_3` は文字列が異なる。
一括ランナーは対応する `layerN_{T}` を含む評価ファイル名を検索する。評価フォルダ名だけからモデルフォルダを推測しない。
タスク名では `-` を `_` に置換するため、例として `TARGET=4-43` のタスク名は `layer2_4_43` になる。

validation / test は入力・結果の保存先で分離し、モデルフォルダ名には付けない。同じ構成の重みを両分割で共用する。
マスク有無も、この既存階層ランナーでは評価データと結果保存先を切り替えるものであり、モデルフォルダ名には追加しない。

## 実構造・E計算用表記との区別

`layer3_4_4_8` の現カテゴリ定義は、第1層4、第2層計16、第3層計152。
16枝のカテゴリ数は9個が12枝、11個が4枝であり、末尾8を実カテゴリ数と解釈しない。
分析コードの `4-4-8 -> 4-4-10` はE値を結合するための別名対応で、フォルダの命名規則を定めるものではない。
8という旧識別名の採用意図や、E計算の10を決めた厳密な丸め規則は、この資料では未確認として扱う。

## 参照コード

- [1層の推論](../scripts/inference/mashup_dis_nosp_1layer_api.py): `main2()` 内の `layer1_model_path`。
- [2層の推論](../scripts/inference/mashup_dis_nosp_2layer_api.py): `TARGET` / `LAYER1_CATEGORY_NUM` と `layer1_model_path` / `layer2_model_path`。
- [3層の推論](../scripts/inference/mashup_dis_nosp_3layer_api.py): `TARGET` と `layer1_model_path`〜`layer3_model_path`。
- [4層の推論](../scripts/inference/mashup_dis_nosp_4layer_api.py): `TARGET` と `layer1_model_path`〜`layer4_model_path`。
- [一括ランナー](../scripts/inference/run_all_existing_hierarchy_patterns.py): `discover_tasks()`、`make_task()`、`run_task()`。
- [パス変換](../scripts/common/project_paths.py): `_legacy_weight_path()`、`_legacy_named_weight_path()`、`legacy_path()`。
- [E計算用の別名対応](../analysis/layer_ana/plot_e_f1_from_readable_summary_pil.py): `E_STRUCTURE_ALIASES`。
