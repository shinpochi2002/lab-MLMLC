# Condition Analysis Test Subset Creation Report

作成日: 2026-06-08 15:52:45

## 1. 使用した入力ファイル

- 元テストデータ: `/home/nakagawa/lab-MLMLC/codex/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json`
- 分析CSV: `/home/nakagawa/lab-MLMLC/codex/analysis_test_subset_conditions/sample_condition_flags.csv`
- 元JSON形式: `dict`

## 2. 使用した指標

- 説明文長: `word_count`
- API候補数: `sum_oracle_category_candidate_size`

## 3. 使用した閾値

- `shorter_desc`: `word_count <= 24`
- `longer_desc`: `word_count >= 25`
- `fewer_candidates`: `sum_oracle_category_candidate_size <= 52`
- `more_candidates`: `sum_oracle_category_candidate_size >= 53`

## 4. 作成したサブセット一覧

| subset | condition | count | expected | match | file |
| --- | --- | --- | --- | --- | --- |
| overall | all 336 existing test samples | 336 | 336 | yes | overall.json |
| shorter_desc | word_count <= 24 | 171 | 171 | yes | by_length/shorter_desc.json |
| longer_desc | word_count >= 25 | 165 | 165 | yes | by_length/longer_desc.json |
| fewer_candidates | sum_oracle_category_candidate_size <= 52 | 195 | 195 | yes | by_candidate_size/fewer_candidates.json |
| more_candidates | sum_oracle_category_candidate_size >= 53 | 141 | 141 | yes | by_candidate_size/more_candidates.json |
| shorter_desc_fewer_candidates | word_count <= 24 and sum_oracle_category_candidate_size <= 52 | 107 | 107 | yes | by_length_candidate_2x2/shorter_desc_fewer_candidates.json |
| shorter_desc_more_candidates | word_count <= 24 and sum_oracle_category_candidate_size >= 53 | 64 | 64 | yes | by_length_candidate_2x2/shorter_desc_more_candidates.json |
| longer_desc_fewer_candidates | word_count >= 25 and sum_oracle_category_candidate_size <= 52 | 88 | 88 | yes | by_length_candidate_2x2/longer_desc_fewer_candidates.json |
| longer_desc_more_candidates | word_count >= 25 and sum_oracle_category_candidate_size >= 53 | 77 | 77 | yes | by_length_candidate_2x2/longer_desc_more_candidates.json |

## 5. 件数検証

- 期待件数との一致: yes
- `overall`: 336
- `shorter_desc + longer_desc = 336`
- `fewer_candidates + more_candidates = 336`
- 2x2 の4セル合計 = 336

## 6. 分類軸ごとの重複・欠落・元データ外ID確認

| group | count_sum | expected_total | unique_count | duplicate_count | missing_count | outside_original_count | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| by_length | 336 | 336 | 336 | 0 | 0 | 0 | yes |
| by_candidate_size | 336 | 336 | 336 | 0 | 0 | 0 | yes |
| by_length_candidate_2x2 | 336 | 336 | 336 | 0 | 0 | 0 | yes |

## 7. 合算関係の確認

| relationship | left_count | right_count | ok |
| --- | --- | --- | --- |
| shorter_desc = shorter_desc_fewer_candidates + shorter_desc_more_candidates | 171 | 171 | yes |
| longer_desc = longer_desc_fewer_candidates + longer_desc_more_candidates | 165 | 165 | yes |
| fewer_candidates = shorter_desc_fewer_candidates + longer_desc_fewer_candidates | 195 | 195 | yes |
| more_candidates = shorter_desc_more_candidates + longer_desc_more_candidates | 141 | 141 | yes |
| overall = shorter_desc + longer_desc | 336 | 336 | yes |
| overall = fewer_candidates + more_candidates | 336 | 336 | yes |

## 8. 注意事項

- 今回のサブセットは、新しいtrain/test splitではなく、既存テストデータ336件を条件別に再集計するためのものである。
- `shorter_desc` / `longer_desc` は、テストデータ内での相対的な説明文長を表す。
- `fewer_candidates` / `more_candidates` は、テストデータ内での相対的なAPI候補数を表す。
- `sum_oracle_category_candidate_size` は、正解カテゴリパスが分かっていると仮定した後に残るAPI候補数であり、カテゴリ予測自体の難しさを直接表すものではない。
- 単体サブセットと2x2サブセットは独立したデータではなく、同じ336件を異なる観点で集計したものである。

## 9. 出力ディレクトリ

- `/home/nakagawa/lab-MLMLC/codex/test_subsets/condition_analysis`

## 10. 実行コマンドの記録

```bash
python3 scripts/create_condition_analysis_subsets.py
```

- スクリプト保存場所: `/home/nakagawa/lab-MLMLC/codex/scripts/create_condition_analysis_subsets.py`
- バックアップ作成数: 0
