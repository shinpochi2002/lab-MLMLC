# Test Subset Creation Report

## Overview
- total samples: 336
- Created concrete test-data files for document length, choice size, 3x3 length x choice, and 2x2 integrated length x choice splits.
- No model training, inference, or F1 calculation was performed.
- original data format: dict

## Bin Definitions
- Document length: short = 1-20 words, medium = 21-50 words, long = 51+ words.
- Choice size: small = 1-20 candidate APIs, medium = 21-100 candidate APIs, large = 101+ candidate APIs.
- Choice size is based on sum_oracle_category_candidate_size, meaning the number of APIs remaining after assuming the correct category path.
- 2x2 length groups: InfoLimited = short; InfoRich = medium or long.
- 2x2 choice groups: SmallChoice = small; LargeChoice = medium or large.

## By Document Length
| subset_name | count | ratio | output_data_file | feasibility_label |
| --- | --- | --- | --- | --- |
| short | 128 | 0.381 | by_length/short_test.json | main_or_comparison_ready |
| medium | 177 | 0.527 | by_length/medium_test.json | main_or_comparison_ready |
| long | 31 | 0.092 | by_length/long_test.json | diagnostic_only |

## By Choice Size
| subset_name | count | ratio | output_data_file | feasibility_label |
| --- | --- | --- | --- | --- |
| small_choice | 49 | 0.146 | by_choice/small_choice_test.json | diagnostic_only |
| medium_choice | 253 | 0.753 | by_choice/medium_choice_test.json | main_or_comparison_ready |
| large_choice | 34 | 0.101 | by_choice/large_choice_test.json | diagnostic_only |

## By Length x Choice, 3x3
| subset_name | count | ratio | output_data_file | feasibility_label |
| --- | --- | --- | --- | --- |
| short_small | 17 | 0.051 | by_length_choice_3x3/short_small_test.json | case_study_only |
| short_medium | 100 | 0.298 | by_length_choice_3x3/short_medium_test.json | main_or_comparison_ready |
| short_large | 11 | 0.033 | by_length_choice_3x3/short_large_test.json | case_study_only |
| medium_small | 26 | 0.077 | by_length_choice_3x3/medium_small_test.json | case_study_only |
| medium_medium | 132 | 0.393 | by_length_choice_3x3/medium_medium_test.json | main_or_comparison_ready |
| medium_large | 19 | 0.057 | by_length_choice_3x3/medium_large_test.json | case_study_only |
| long_small | 6 | 0.018 | by_length_choice_3x3/long_small_test.json | case_study_only |
| long_medium | 21 | 0.062 | by_length_choice_3x3/long_medium_test.json | case_study_only |
| long_large | 4 | 0.012 | by_length_choice_3x3/long_large_test.json | case_study_only |

## By Length x Choice, 2x2
| subset_name | count | ratio | output_data_file | feasibility_label |
| --- | --- | --- | --- | --- |
| info_limited_small_choice | 17 | 0.051 | by_length_choice_2x2/info_limited_small_choice_test.json | case_study_only |
| info_limited_large_choice | 111 | 0.330 | by_length_choice_2x2/info_limited_large_choice_test.json | main_or_comparison_ready |
| info_rich_small_choice | 32 | 0.095 | by_length_choice_2x2/info_rich_small_choice_test.json | diagnostic_only |
| info_rich_large_choice | 176 | 0.524 | by_length_choice_2x2/info_rich_large_choice_test.json | main_or_comparison_ready |

## Validation
| group | count_sum | expected_total | sum_ok | expected_counts_ok |
| --- | --- | --- | --- | --- |
| by_length | 336 | 336 | yes | yes |
| by_choice | 336 | 336 | yes | yes |
| by_length_choice_3x3 | 336 | 336 | yes | yes |
| by_length_choice_2x2 | 336 | 336 | yes | yes |

### Duplicate Checks
| group | unique_count | duplicate_count | duplicates_ok |
| --- | --- | --- | --- |
| by_length | 336 | 0 | yes |
| by_choice | 336 | 0 | yes |
| by_length_choice_3x3 | 336 | 0 | yes |
| by_length_choice_2x2 | 336 | 0 | yes |
- missing IDs in original data: 0

## Recommended Usage
- Main experiment: use globally representative subsets such as representative_with_length; these condition splits are mainly for auxiliary and diagnostic analysis.
- By document length: short_test.json and medium_test.json are comparison-ready; long_test.json is diagnostic/case-study oriented.
- By choice size: medium_choice_test.json is comparison-ready; small_choice_test.json and large_choice_test.json are diagnostic-oriented.
- 3x3 split: short_medium_test.json and medium_medium_test.json are the most comparison-ready; sparse cells should be used diagnostically or qualitatively.
- 2x2 split: info_limited_large_choice_test.json and info_rich_large_choice_test.json are comparison-ready; info_limited_small_choice_test.json is case-study only and info_rich_small_choice_test.json is diagnostic-only.

## Feasibility Summary
- main_or_comparison_ready: short, medium, medium_choice, short_medium, medium_medium, info_limited_large_choice, info_rich_large_choice
- auxiliary_ready: none
- diagnostic_only: long, small_choice, large_choice, info_rich_small_choice
- case_study_only: short_small, short_large, medium_small, medium_large, long_small, long_medium, long_large, info_limited_small_choice

## Notes
- warnings: 0
- backups created: 40
