# Hierarchy Pattern Condition-Analysis Subset Creation Report

作成日: 2026-06-08 16:01:06

## Overview

- 既存の `condition_analysis` 9分割を、階層構造パターン別テストデータへ同じ `sample_id` で投影した。
- 新しいtrain/test splitではなく、既存テストデータ336件を条件別に再集計するための派生ファイルである。
- サンプル内容は変更せず、各パターンJSON内の値をそのまま保持した。

## Inputs

- condition split dir: `/home/nakagawa/lab-MLMLC/codex/test_subsets/condition_analysis`
- hierarchy pattern dir: `/home/nakagawa/lab-MLMLC/codex/output/by_pattern`
- source glob: `*/*_drop_unknown_samples.json`

## Output

- output dir: `/home/nakagawa/lab-MLMLC/codex/test_subsets/condition_analysis_by_pattern`
- source files processed: 22
- JSON files created: 198

## Subset Counts

| subset | count |
| --- | --- |
| overall | 336 |
| shorter_desc | 171 |
| longer_desc | 165 |
| fewer_candidates | 195 |
| more_candidates | 141 |
| shorter_desc_fewer_candidates | 107 |
| shorter_desc_more_candidates | 64 |
| longer_desc_fewer_candidates | 88 |
| longer_desc_more_candidates | 77 |

## Processed Sources

| pattern | source_file | count | output_root |
| --- | --- | --- | --- |
| 12_13 | test_data_mashup_apis_with_hierarchy_layer2_12_drop_unknown_samples.json | 336 | 12_13/test_data_mashup_apis_with_hierarchy_layer2_12_drop_unknown_samples |
| 12_2_3_2 | test_data_mashup_apis_layer1_12_drop_unknown_samples.json | 336 | 12_2_3_2/test_data_mashup_apis_layer1_12_drop_unknown_samples |
| 12_2_3_2 | test_data_mashup_apis_with_hierarchy_layer1_12_drop_unknown_samples.json | 336 | 12_2_3_2/test_data_mashup_apis_with_hierarchy_layer1_12_drop_unknown_samples |
| 12_2_3_2 | test_data_mashup_apis_with_hierarchy_layer4_12_2_2_3_drop_unknown_samples.json | 336 | 12_2_3_2/test_data_mashup_apis_with_hierarchy_layer4_12_2_2_3_drop_unknown_samples |
| 12_2_6 | test_data_mashup_apis_with_hierarchy_layer3_12_2_6_drop_unknown_samples.json | 336 | 12_2_6/test_data_mashup_apis_with_hierarchy_layer3_12_2_6_drop_unknown_samples |
| 12_4_3 | test_data_mashup_apis_with_hierarchy_layer3_12_4_3_drop_unknown_samples.json | 336 | 12_4_3/test_data_mashup_apis_with_hierarchy_layer3_12_4_3_drop_unknown_samples |
| 12_6_2 | test_data_mashup_apis_with_hierarchy_layer3_12_6_2_drop_unknown_samples.json | 336 | 12_6_2/test_data_mashup_apis_with_hierarchy_layer3_12_6_2_drop_unknown_samples |
| 15-10 | test_data_mashup_apis_with_hierarchy_layer2_15_drop_unknown_samples.json | 336 | 15-10/test_data_mashup_apis_with_hierarchy_layer2_15_drop_unknown_samples |
| 16_2_4 | test_data_mashup_apis_with_hierarchy_16_2_4_drop_unknown_samples.json | 336 | 16_2_4/test_data_mashup_apis_with_hierarchy_16_2_4_drop_unknown_samples |
| 16cluster | test_data_mashup_apis_with_hierarchy_layer3_16_4_3_drop_unknown_samples.json | 336 | 16cluster/test_data_mashup_apis_with_hierarchy_layer3_16_4_3_drop_unknown_samples |
| 2-76 | test_data_mashup_apis_with_hierarchy_2-76_drop_unknown_samples.json | 336 | 2-76/test_data_mashup_apis_with_hierarchy_2-76_drop_unknown_samples |
| 24-7 | test_data_mashup_apis_with_hierarchy_layer2_24_drop_unknown_samples.json | 336 | 24-7/test_data_mashup_apis_with_hierarchy_layer2_24_drop_unknown_samples |
| 33-4 | test_data_mashup_apis_with_hierarchy_layer2_33_drop_unknown_samples.json | 336 | 33-4/test_data_mashup_apis_with_hierarchy_layer2_33_drop_unknown_samples |
| 4-43 | test_data_mashup_apis_with_hierarchy_layer2_4_drop_unknown_samples.json | 336 | 4-43/test_data_mashup_apis_with_hierarchy_layer2_4_drop_unknown_samples |
| 4cluster | test_data_mashup_apis_with_hierarchy_layer3_4_4_8_drop_unknown_samples.json | 336 | 4cluster/test_data_mashup_apis_with_hierarchy_layer3_4_4_8_drop_unknown_samples |
| 51-3 | test_data_mashup_apis_with_hierarchy_layer2_51_drop_unknown_samples.json | 336 | 51-3/test_data_mashup_apis_with_hierarchy_layer2_51_drop_unknown_samples |
| 6-25 | test_data_mashup_apis_with_hierarchy_layer2_6_drop_unknown_samples.json | 336 | 6-25/test_data_mashup_apis_with_hierarchy_layer2_6_drop_unknown_samples |
| 8-19 | test_data_mashup_apis_with_hierarchy_layer2_8_drop_unknown_samples.json | 336 | 8-19/test_data_mashup_apis_with_hierarchy_layer2_8_drop_unknown_samples |
| 8_3_2_3 | test_data_mashup_apis_with_hierarchy_layer1_8_drop_unknown_samples.json | 336 | 8_3_2_3/test_data_mashup_apis_with_hierarchy_layer1_8_drop_unknown_samples |
| 8_3_2_3 | test_data_mashup_apis_with_hierarchy_layer4_8_3_2_3_drop_unknown_samples.json | 336 | 8_3_2_3/test_data_mashup_apis_with_hierarchy_layer4_8_3_2_3_drop_unknown_samples |
| 8_4_4 | test_data_mashup_apis_with_hierarchy_layer3_8_4_4_drop_unknown_samples.json | 336 | 8_4_4/test_data_mashup_apis_with_hierarchy_layer3_8_4_4_drop_unknown_samples |
| 8_5_4 | test_data_mashup_apis_with_hierarchy_8_5_4_drop_unknown_samples.json | 336 | 8_5_4/test_data_mashup_apis_with_hierarchy_8_5_4_drop_unknown_samples |

## Validation

- 各source JSONの `sample_id` 集合は `condition_analysis/overall.json` と完全一致した。
- 各source x subset の件数は元の9分割件数と一致した。
- 各分類軸の重複・欠落は、入力分割側で検証済みのpartitionを使用した。

## Notes

- `shorter_desc` / `longer_desc` は、テストデータ内での相対的な説明文長を表す。
- `fewer_candidates` / `more_candidates` は、テストデータ内での相対的なAPI候補数を表す。
- `sum_oracle_category_candidate_size` は、正解カテゴリパスが分かっていると仮定した後に残るAPI候補数であり、カテゴリ予測自体の難しさを直接表すものではない。

## Execution Command

```bash
python3 scripts/create_condition_analysis_hierarchy_pattern_subsets.py
```

- script: `/home/nakagawa/lab-MLMLC/codex/scripts/create_condition_analysis_hierarchy_pattern_subsets.py`
- backups created: 0
