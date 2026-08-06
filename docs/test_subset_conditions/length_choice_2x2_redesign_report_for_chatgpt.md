# Length/Choice 2x2 Redesign Report for ChatGPT

作成日: 2026-06-08
対象ディレクトリ: `/home/nakagawa/lab-MLMLC/codex/analysis_test_subset_conditions`

## 目的

現在はテストデータ 336 件に対して、説明文の長さを `short / medium / long`、選択肢数を `small / medium / large` の 3 分割で分析している。
今後は 3 分割ではなく、長さは `short / long`、選択肢数は `small / large` の 2 分割で進めたい。

このレポートの目的は、ChatGPT に渡して「2 分割の基準をどう定義すべきか」を検討してもらうために、現状のデータ、既存基準、問題点、推奨案を整理することである。

## 現状の入力・生成物

- テストデータ総数: 336
- 元テストデータ: `/home/nakagawa/lab-MLMLC/codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`
- 分析CSV: `/home/nakagawa/lab-MLMLC/codex/analysis_test_subset_conditions/sample_condition_flags.csv`
- 既存レポート:
  - `extended_subset_condition_report.md`
  - `length_choice_subset_report.md`
  - `test_subsets/reports/test_subset_creation_report.md`
- 既存サブセット:
  - `test_subsets/by_length/`
  - `test_subsets/by_choice/`
  - `test_subsets/by_length_choice_3x3/`
  - `test_subsets/by_length_choice_2x2/`
- 既存生成時の検証:
  - 各グループの件数合計は 336 件で一致
  - 重複IDは 0
  - 元データに存在しないIDは 0
  - warnings は 0
- モデル推論、学習、F1 計算はまだ実施していない。

## 現在使っている指標

### 長さ

`word_count` を使う。

既存の 3 分割:

| label | definition | count | ratio |
| --- | --- | ---: | ---: |
| short | `1 <= word_count <= 20` | 128 | 0.381 |
| medium | `21 <= word_count <= 50` | 177 | 0.527 |
| long | `word_count >= 51` | 31 | 0.092 |

分布:

| metric | value |
| --- | ---: |
| min | 4 |
| max | 203 |
| mean | 28.875 |
| median | 24 |
| p25 | 17 |
| p75 | 36 |
| p90 | 48 |

### 選択肢数

`sum_oracle_category_candidate_size` を使う。
これは「正解カテゴリパスが分かっていると仮定した後に残る API 候補数」の合計であり、API 層での選択肢の多さを近似する指標である。

既存の 3 分割:

| label | definition | count | ratio |
| --- | --- | ---: | ---: |
| small | `1 <= sum_oracle_category_candidate_size <= 20` | 49 | 0.146 |
| medium | `21 <= sum_oracle_category_candidate_size <= 100` | 253 | 0.753 |
| large | `sum_oracle_category_candidate_size >= 101` | 34 | 0.101 |

分布:

| metric | value |
| --- | ---: |
| min | 1 |
| max | 229 |
| mean | 57.917 |
| median | 52 |
| p25 | 40 |
| p75 | 73 |
| p90 | 100.5 |

注意点: 候補数 52 のサンプルが 77 件あり、中央値付近に大きな同値の山がある。そのため、中央値閾値を使う場合は `<= 52` を small 側に含めるか、`>= 52` を large 側に含めるかで件数が変わる。

## 既存 3x3 の件数

| length | choice small | choice medium | choice large | total |
| --- | ---: | ---: | ---: | ---: |
| short | 17 | 100 | 11 | 128 |
| medium | 26 | 132 | 19 | 177 |
| long | 6 | 21 | 4 | 31 |

既存 3x3 では、比較に使いやすいセルは `short_medium` 100 件と `medium_medium` 132 件のみである。
他のセルは 30 件未満が多く、F1 比較の主張には向かない。

## 既存 2x2 の状態

現在の `by_length_choice_2x2` は、実質的に以下の統合である。

- `InfoLimited`: 既存 `short`
- `InfoRich`: 既存 `medium + long`
- `SmallChoice`: 既存 `small`
- `LargeChoice`: 既存 `medium + large`

| subset | definition | count | ratio | feasibility |
| --- | --- | ---: | ---: | --- |
| InfoLimited_SmallChoice | `word_count_bin == short` and `choice_bin == small` | 17 | 0.051 | case study only |
| InfoLimited_LargeChoice | `word_count_bin == short` and `choice_bin in {medium, large}` | 111 | 0.330 | comparison ready |
| InfoRich_SmallChoice | `word_count_bin in {medium, long}` and `choice_bin == small` | 32 | 0.095 | diagnostic only |
| InfoRich_LargeChoice | `word_count_bin in {medium, long}` and `choice_bin in {medium, large}` | 176 | 0.524 | comparison ready |

この設計は全サンプルを使えるが、`long` が本来の長文ではなく `medium + long` を意味し、`large` も `medium + large` を意味する。
また small 側のセルが 17 件と 32 件で小さい。

## 2分割基準の候補

### 候補A: 既存3分割の両端だけを使う

定義:

- length short: `word_count <= 20`
- length long: `word_count >= 51`
- choice small: `candidate_size <= 20`
- choice large: `candidate_size >= 101`
- medium 領域は除外

件数:

| subset | count |
| --- | ---: |
| short_small | 17 |
| short_large | 11 |
| long_small | 6 |
| long_large | 4 |
| used total | 38 |
| excluded total | 298 |

判断:

- ラベルの意味は直感的で、短い・長い、小さい・大きいのコントラストは強い。
- ただし全体 336 件中 38 件しか使えず、最小セルが 4 件なので、F1 比較には不適。
- 定性的なケーススタディ用途に限定すべき。

### 候補B: 既存2x2を short/long, small/large にリネームする

定義:

- length short: `word_count <= 20`
- length long: `word_count >= 21`
- choice small: `candidate_size <= 20`
- choice large: `candidate_size >= 21`

件数:

| subset | count |
| --- | ---: |
| short_small | 17 |
| short_large | 111 |
| long_small | 32 |
| long_large | 176 |

判断:

- 全サンプルを使える。
- 既存 `by_length_choice_2x2` とほぼ同じで、生成済み資産を活用しやすい。
- ただし `long` は「21語以上」であり、本来の長文というより `not short` に近い。
- `large` は「21候補以上」であり、本来の大候補というより `not small` に近い。
- small 側のセルが小さく、主比較には弱い。

### 候補C: 全サンプルを中央値ベースで2分割する

推奨案。

定義:

- length short: `word_count <= 24`
- length long: `word_count >= 25`
- choice small: `sum_oracle_category_candidate_size <= 52`
- choice large: `sum_oracle_category_candidate_size >= 53`

この定義では、`short / long` と `small / large` は絶対的な短文・長文ではなく、このデータセット内での相対的な短め・長め、小さめ・大きめを意味する。
混乱を避けるなら、内部名は `relative_short`, `relative_long`, `relative_small_choice`, `relative_large_choice` または `short2`, `long2`, `small2`, `large2` とするのがよい。

件数:

| subset | count | ratio |
| --- | ---: | ---: |
| short_small | 107 | 0.318 |
| short_large | 64 | 0.190 |
| long_small | 88 | 0.262 |
| long_large | 77 | 0.229 |
| total | 336 | 1.000 |

セル別統計:

| subset | avg_word_count | median_word_count | word_range | avg_candidate_size | median_candidate_size | candidate_range | avg_gt_apis | avg_gt_categories | seen_only_ratio | multi_category_ratio | ambiguous_requirement_ratio |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| short_small | 15.822 | 16 | 4-24 | 38.037 | 48 | 1-52 | 1.262 | 1.112 | 0.925 | 0.112 | 0.271 |
| short_large | 17.016 | 18 | 7-24 | 84.797 | 75.5 | 54-229 | 2.734 | 2.328 | 0.844 | 0.844 | 0.250 |
| long_small | 44.091 | 36 | 25-203 | 35.307 | 44.5 | 1-52 | 1.636 | 1.409 | 0.784 | 0.307 | 0.295 |
| long_large | 39.481 | 36 | 25-105 | 89.039 | 84 | 53-225 | 3.416 | 2.675 | 0.779 | 0.922 | 0.169 |

判断:

- 全 336 件を漏れなく使える。
- 最小セルが 64 件なので、既存3x3や既存2x2より主比較に使いやすい。
- 閾値が F1 結果を見る前の共変量だけで決まるため、事後的な都合のよい分割を避けやすい。
- 候補数の中央値 52 に同値が多く、small 側が 195 件、large 側が 141 件になる。完全な 50/50 ではない点は注記が必要。
- `long` の下限が 25 語なので、通常の意味での長文ではない。論文・レポートでは「relative long」または「longer」などの表現が望ましい。

### 候補D: コントラスト重視のtertile両端抽出

定義例:

- length short: `word_count <= 20`
- length long: `word_count >= 31`
- choice small: `candidate_size <= 47`
- choice large: `candidate_size >= 62`
- 中央領域は除外

件数:

| subset | count |
| --- | ---: |
| short_small | 32 |
| short_large | 37 |
| long_small | 35 |
| long_large | 38 |
| used total | 142 |
| excluded total | 194 |

判断:

- 候補Aよりセル件数が大きく、4セルのバランスもよい。
- 長さ・候補数の差がはっきり出る。
- ただし全サンプル評価ではなく、中央の 194 件を捨てるため、主実験ではなく感度分析・診断向け。

## 推奨する基準

主分析で 2x2 を使いたいなら、候補Cを推奨する。

推奨基準:

| axis | label | definition |
| --- | --- | --- |
| length | short | `word_count <= 24` |
| length | long | `word_count >= 25` |
| choice | small | `sum_oracle_category_candidate_size <= 52` |
| choice | large | `sum_oracle_category_candidate_size >= 53` |

理由:

- 336 件すべてを使える。
- 4セルすべてが 50 件以上で、F1 比較の最低限の安定性がある。
- 閾値が中央値ベースで説明しやすい。
- 既存の固定3分割よりセル不足が大幅に改善される。
- medium をどちらに吸収したかという恣意性を減らせる。

ただし、名称は注意する。
`long` と書くと「51語以上」の旧定義と混同するため、レポート内では `relative_long` または `longer` と明記するのが望ましい。
同様に `small / large` も、既存の `<=20 / >=101` ではなく中央値ベースの相対的な小・大であると明記する。

## 実験上のガードレール

1. 閾値はモデル評価結果を見る前に固定する。
2. 閾値を F1 が都合よく分かれるように後から調整しない。
3. 各セルの件数、平均 `word_count`、平均 `candidate_size`、平均正解API数、平均正解カテゴリ数を必ず併記する。
4. F1 だけでなく、可能なら bootstrap confidence interval も出す。
5. 既存の代表サブセット `representative_with_length` は主実験の基準として残し、2x2 は補助分析・要因分析として扱う。
6. `sum_oracle_category_candidate_size` は「正解カテゴリが分かっている前提」の候補数なので、カテゴリ予測自体の難しさではなく API 選択段階の候補数を表している、と明記する。
7. `long_large` は候補数だけでなく、正解API数・正解カテゴリ数・multi-category比も高いので、単純に「候補数だけの効果」と解釈しない。

## ChatGPTに確認してほしい論点

1. 主分析として候補Cの中央値ベース2x2を採用してよいか。
2. `short / long`、`small / large` という名前でよいか、それとも `relative_short / relative_long`、`relative_small / relative_large` にすべきか。
3. 候補数 52 の同値が 77 件あるため、`<=52` を small に含める現在案でよいか。
4. 比較の主張を F1 の単純比較にするか、bootstrap CI 付きの差分比較にするか。
5. `long_large` が multi-category や正解API数の多さも同時に含んでいるため、追加で `num_gt_apis` や `num_gt_categories` を統制した分析が必要か。
6. 中央値ベースの全サンプル2x2に加えて、候補Dのtertile両端抽出を感度分析として併記すべきか。

## ChatGPTへの貼り付け用依頼文

以下の前提で、テストデータを長さ2分割・選択肢数2分割の4セルに分ける基準を検討してください。

- 総数は 336 件です。
- 長さ指標は `word_count` です。
- 選択肢数指標は `sum_oracle_category_candidate_size` です。これは正解カテゴリパスが分かっていると仮定した後に残る API 候補数です。
- 既存の3分割は、長さが `short <=20`, `medium 21-50`, `long >=51`、選択肢数が `small <=20`, `medium 21-100`, `large >=101` です。
- 既存3x3の件数は `short_small=17`, `short_medium=100`, `short_large=11`, `medium_small=26`, `medium_medium=132`, `medium_large=19`, `long_small=6`, `long_medium=21`, `long_large=4` です。
- 既存3分割の両端だけを使うと合計 38 件しか使えず、主比較には不安定です。
- 現在の推奨案は、全サンプルを使う中央値ベース2x2です。
  - `short: word_count <= 24`
  - `long: word_count >= 25`
  - `small: sum_oracle_category_candidate_size <= 52`
  - `large: sum_oracle_category_candidate_size >= 53`
- この場合の件数は `short_small=107`, `short_large=64`, `long_small=88`, `long_large=77` です。
- ただし候補数 52 が 77 件あり、small側がやや大きくなります。また `long` は絶対的な長文ではなく相対的に長いという意味になります。

この条件で、主分析に採用すべき2分割基準、命名、注意書き、追加すべき感度分析を提案してください。
