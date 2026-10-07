# Early Commitment Score E の定義と計算経路監査

## 1. 結論

Table II / Fig. 3 / Discussion で使われている Early Commitment Score `E` は、既存実装から再現できる。
ただし、論文中の説明どおりに `N_l` を「layer `l` の実label数」、`N_L` を「final category layer の実label数」と読むと、Table II の値とは一致しない。

最重要例である `(12,48,152,909) -> E = 0.5625` は、表示上の `[12, 48, 152]` ではなく、実装上の `branch=[12,4,4,6]` から API推薦層 `6` を除外し、累積値 `[12,48,192]` を使うことで再現される。

したがって最終判定は **Case D: `N_l` の意味が論文と実装で異なる**。

## 2. E計算コード

| ファイル | 関数 | 行番号 | 使用される計算式 | 用途 |
|---|---|---:|---|---|
| `analysis/layer_ana/layer_ana.py` | `compute_E_from_branching` | 5 | `C_l = prod(b_1..b_l)`, `E = sum_l ((L-l+1)/L) * (C_l/C_L)` | `E_F1_scatter_for_paper.csv` 生成候補 |
| `analysis/layer_ana/layer_ana_pare.py` | `compute_E_from_branching` | 5 | 同上 | Pareto付き旧Fig生成候補 |
| `analysis/layer_ana/tansaku.py` | `compute_E_from_cumulative` | 3 | 入力済み累積値 `C_l` に対して同式 | E範囲に入る階層候補探索 |
| `analysis/layer_ana/plot_pareto_by_test_data.py` | `load_points` | 283 | E再計算なし。`E_F1_layer_structure_for_paper.csv` を読込 | Fig. 3系CSV/図生成 |
| `analysis/layer_ana/plot_e_f1_from_readable_summary_pil.py` | `load_e_values` / `build_points` | 219 / 231 | E再計算なし。CSV読込後、`summary_structure` を `E_structure` にalias | 現行 `results/.../e_f1_pareto_latest_auto_y` 生成 |

実装式を数式化すると以下になる。

```text
Input: branching_factors = [b_1, b_2, ..., b_L, b_API]
category_branch = branching_factors[:-1]
C_l = product_{i=1}^{l} b_i
C_L = product_{i=1}^{L} b_i
w_l = (L - l + 1) / L

E = sum_{l=1}^{L} w_l * (C_l / C_L)
```

ここで `C_l` は実label数ではなく、分岐数から作る累積容量である。

## 3. `N_l` / `L` の実装上の定義

| 項目 | 論文上の説明 | 実装上の定義 | 一致 |
|---|---|---|---|
| `L` | final API recommendation layer を含まない | `branching_factors[:-1]` の長さ | ほぼ一致 |
| `N_l` | layer `l` のlabel数 | 分岐数の累積積 `C_l` | 不一致 |
| `N_L` | final category layer のlabel数、例: 152 | 累積積の最後、例: 192 | 不一致 |
| API layer | 含まない | 入力には含むが、計算では除外 | 一致 |
| root layer | 含まない | 含まない | 一致 |
| branch factor | 明示上は使わない | E計算の入力として使用 | 不一致 |

例: `layer3_12_4_3` / `(12,48,152,909)` の場合、実データ上のカテゴリ総数は `results/hierarchy/output/by_pattern/12_4_3/category_hierarchy_12_4_3.json` の `num_categories = 152`。
一方、E計算では `12 * 4 * 4 = 192` を `N_L` 相当として使っている。

また、同じ `12_4_3` の `simple_hierarchy_12_4_3.json` では以下を確認した。

| 実階層 | 値 |
|---|---:|
| layer1 count | 12 |
| layer2 count | 48 |
| category count | 152 |
| layer2 per layer1 | 4 x 12 |
| categories per layer2 | 3カテゴリが41個、4カテゴリが6個、5カテゴリが1個 |

したがって、`12_4_3` は「第1層12、第2層は各第1層あたり4、カテゴリ層は実数152」だが、Eはカテゴリ層を一律4分岐とみなし、累積192で計算している。

## 4. configuration別再計算

| Configuration | Table/Fig E | 実装から再計算 | 論文式から再計算 | 一致 |
|---|---:|---:|---:|---|
| `(4,152,909)` | 0.5263 | 0.5263 | 0.5263 | 実装・論文とも一致 |
| `(12,152,909)` | 0.5769 | 0.5769 | 0.5789 | 実装のみ一致 |
| `(51,152,909)` | 0.8333 | 0.8333 | 0.8355 | 実装のみ一致 |
| `(4,16,152,909)` | 0.4250 | 0.4250 | 0.4298 | 実装のみ一致 |
| `(12,24,152,909)` | 0.5000 | 0.5000 | 0.5175 | 実装のみ一致 |
| `(12,48,152,909)` | 0.5625 | 0.5625 | 0.6228 | 実装のみ一致 |
| `(12,72,152,909)` | 0.6111 | 0.6111 | 0.7281 | 実装のみ一致 |
| `(12,24,48,152,909)` | 0.5312 | 0.5312 | 0.6053 | 実装のみ一致 |

論文式からの再計算は、表示上のカテゴリ総数系列をそのまま `[N_1, ..., N_L]` として代入した値である。

## 5. `(12,48,152,909)` の詳細計算

### 実装に基づく計算

入力:

```text
branching_factors = [12, 4, 4, 6]
```

API推薦層を除外:

```text
category_branch = [12, 4, 4]
L = 3
```

累積値:

```text
C_1 = 12
C_2 = 12 * 4 = 48
C_3 = 12 * 4 * 4 = 192
C_L = 192
```

各項:

```text
l=1:
  w_1 = (3-1+1)/3 = 1
  term_1 = 1 * (12/192) = 0.0625

l=2:
  w_2 = (3-2+1)/3 = 2/3
  term_2 = (2/3) * (48/192) = 0.1666666667

l=3:
  w_3 = (3-3+1)/3 = 1/3
  term_3 = (1/3) * (192/192) = 0.3333333333
```

合計:

```text
E = 0.0625 + 0.1666666667 + 0.3333333333
  = 0.5625
```

### 論文文面どおりに読んだ場合

入力:

```text
[N_1, N_2, N_3] = [12, 48, 152]
N_L = 152
L = 3
```

計算:

```text
E = 1 * (12/152)
  + (2/3) * (48/152)
  + (1/3) * (152/152)
  = 0.6228070175
```

したがって、`0.5625` は論文文面どおりの `N_l = 実label数` からは再現できない。

## 6. Table II / Fig. 3 / Discussion の生成経路

確認できた主要経路は以下。

```text
hierarchy experiment result
 例: layer3_12_4_3
    |
    v
results/hierarchy/.../readable_model_summary.csv
  F1などの性能値
    |
    v
analysis/layer_ana/E_F1_layer_structure_for_paper.csv
  E値の表
    |
    v
analysis/layer_ana/plot_e_f1_from_readable_summary_pil.py
  summary_structure -> E_structure のalias適用
  例: 12-4-3 -> 12-4-4
    |
    v
results/hierarchy/.../e_f1_pareto_latest_auto_y/all_test_data_points_with_E.csv
results/hierarchy/.../e_f1_pareto_latest_auto_y/overall_336_E_F1_points.csv
results/hierarchy/.../e_f1_pareto_latest_auto_y/overall_336_E_F1_pareto_front.csv
    |
    v
Fig. 3 / Discussionで使用
```

以下は分析コードの固定値・alias対応を記録したもので、モデルフォルダの命名規則や改名の根拠ではない。モデル参照は[推論コードに基づく命名ルール](hierarchy_model_naming.md)に従い、既存識別名を保持する。

重要な対応:

| 実験名 | summary_structure | E_structure | display_structure_plot | E |
|---|---|---|---|---:|
| `layer3_12_4_3` | `12-4-3` | `12-4-4` | `12-48-152-909` | 0.5625 |
| `layer3_4_4_8` | `4-4-8` | `4-4-10` | `4-16-152-909` | 0.4250 |
| `layer4_12_2_2_3` | `12-2-2-3` | `12-2-2-4` | `12-24-48-152-909` | 0.53125 |
| `layer2_51_3` | `51-3` | `51-3` | `51-152-909` | 0.833333 |

`plot_e_f1_from_readable_summary_pil.py` および `plot_pareto_by_test_data.py` の `E_STRUCTURE_ALIASES` が、実験名とE計算用構造のずれを吸収している。

## 7. Table II値の保存元

最低限確認した値は以下のファイルに保存されている。

| ファイル | 内容 |
|---|---|
| `analysis/layer_ana/E_F1_layer_structure_for_paper.csv` | `category_structure` と `E` の対応表。`12-4-4 -> 0.5625` を含む |
| `analysis/layer_ana/E_F1_scatter_for_paper.csv` | 表示構造とE/F1の対応表。`12-48-152-909 -> 0.5625` を含む |
| `results/hierarchy/hierarchy_existing_models_api_name_masked/e_f1_pareto_latest_auto_y/all_test_data_points_with_E.csv` | 旧mask条件のE/F1結合済み成果物 |
| `results/hierarchy/hierarchy_existing_models_official_api_name_masked/e_f1_pareto_latest_auto_y/all_test_data_points_with_E.csv` | official mask条件のE/F1結合済み成果物 |
| `results/hierarchy/hierarchy_existing_models_official_api_name_masked/e_f1_pareto_latest_auto_y/layer_mapping_check.csv` | `summary_structure` と `E_structure` の対応確認 |

CSVと `results/` は `.gitignore` 対象であり、Git管理されていない。
そのため、CSVが手入力か、スクリプト実行で生成されたかをGit履歴だけで完全には断定できない。
ただし、現行CSV上のE値は `analysis/layer_ana/layer_ana.py` / `analysis/layer_ana/layer_ana_pare.py` の実装式から再現できる。

## 8. 過去の定義変更

確認できた範囲では、古い `analysis/layer_ana/E_F1_layer_structure.csv` に `17.5` など1を超えるE値が残っている。
これは現行Table/Fig.3のE値とは異なる旧定義または旧スケールの成果物である。

一方、現行Fig.3生成経路で使われる `E_F1_layer_structure_for_paper.csv` は、正規化後のE値を持つ。

Git履歴:

| コミット | 日時 | 内容 |
|---|---|---|
| `012b5ac` | 2026-08-06 17:24 +0900 | `layer_ana.py`, `layer_ana_pare.py`, `tansaku.py`, `plot_*` などE関連スクリプト追加 |
| `d1f47ca` | 2026-08-06 18:41 +0900 | `plot_pareto_by_test_data.py` のパス修正。E定義変更ではない |

CSV成果物はGit管理外のため、CSV単体の導入時・変更時の完全な履歴は確認できない。

## 9. Fig. 3との整合性

Fig. 3系の生成スクリプトはEを再計算せず、`analysis/layer_ana/E_F1_layer_structure_for_paper.csv` のE列を読む。
その後、`summary_structure` を `E_structure` に変換してEを結合する。

したがって、Table II相当のE値とFig. 3の横軸Eは同じE値CSVに由来している。
確認した `all_test_data_points_with_E.csv` でも、`layer3_12_4_3` は `E=0.5625`、`display_structure_plot=12-48-152-909` で一致している。

## 10. Discussionとの整合性

Discussionで使われる主要値は、Table II / Fig. 3 / 計算実装と以下のように一致する。

| Configuration | Discussion/Table値 | 実装再計算 | Fig.3系CSV |
|---|---:|---:|---:|
| `(51,152,909)` | 0.8333 | 0.8333 | 0.833333 |
| `(4,16,152,909)` | 0.4250 | 0.4250 | 0.425000 |
| `(12,48,152,909)` | 0.5625 | 0.5625 | 0.562500 |

ただし、これらは論文文面どおりの「実label数」を使った式とは一致しない。

## 11. 最終判定

**Case D: `N_l` の意味が論文と実装で異なる。**

根拠:

- `compute_E_from_branching` は `branching_factors` から累積積を作ってEを計算している。
- `branching_factors` の最後、つまりAPI推薦層への分岐数は除外している。
- `N_l` 相当の値は実label数ではなく、分岐数から作る累積容量である。
- `(12,48,152,909)` の実カテゴリ総数は152だが、E計算では192を使う。
- Table II / Fig. 3 / DiscussionのE値はこの実装定義から再現できる。
- 論文文面どおりに `[12,48,152]` を使うと `0.622807` となり、Table IIの `0.5625` とは一致しない。

## 12. 論文で必要になる修正

現時点ではコードやCSVを変更しない前提で、必要な修正は以下。

| 対象 | 要否 | 理由 |
|---|---|---|
| 式(30)の修正 | 必要 | `N_l` が実label数ではなく累積分岐容量であることを反映する必要がある |
| `L`の説明修正 | 軽微に必要 | API層除外は一致。ただし「カテゴリ層数」と「表示stage数」の区別を明記する必要がある |
| `N_l`の説明修正 | 必要 | 最大の不一致点 |
| `N_L`の説明修正 | 必要 | final category実数152ではなく、累積分岐容量192等を使っているため |
| Table II修正 | E値を維持するなら不要 | ただしEの定義注記は必要 |
| Fig. 3修正 | E値を維持するなら不要 | 横軸定義の説明修正は必要 |
| Discussion修正 | 値を維持するなら不要 | `E` の意味説明は必要 |
| 修正不要 | 不可 | 現状の論文式説明では読者が0.5625を再現できない |

推奨される説明方針:

```text
For a hierarchy with branching factors b_1, ..., b_L before the final API recommendation layer,
we define C_l = product_{i=1}^{l} b_i as the cumulative branching capacity up to layer l.
The Early Commitment Score is
E = sum_{l=1}^{L} ((L-l+1)/L) * (C_l/C_L).
```

そのうえで、表示構造 `(12,48,152,909)` とE計算用構造 `12-4-4` の関係を明記する必要がある。
