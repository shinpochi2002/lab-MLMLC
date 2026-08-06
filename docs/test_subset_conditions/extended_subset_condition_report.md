# Extended Test Dataset Subset Condition Analysis

## Overview
- total samples: 336
- previous conditions are preserved in summary_counts.csv.
- newly added conditions cover description length stratification, oracle candidate size, combination novelty, TF-IDF requirement clarity, and duplicate descriptions.

## Newly Added Conditions Summary
| condition_name | condition_definition | count | ratio | notes |
| --- | --- | --- | --- | --- |
| Description short | 1 <= word_count <= 20 | 128 | 0.381 |  |
| Description medium | 21 <= word_count <= 50 | 177 | 0.527 |  |
| Description long | word_count >= 51 | 31 | 0.092 |  |
| Small-oracle-candidate | sum_oracle_category_candidate_size between 1 and 20 | 49 | 0.146 |  |
| Medium-oracle-candidate | sum_oracle_category_candidate_size between 21 and 100 | 253 | 0.753 |  |
| Large-oracle-candidate | sum_oracle_category_candidate_size >= 101 | 34 | 0.101 |  |
| Large-candidate-category-including | max_oracle_category_candidate_size >= 50 | 182 | 0.542 |  |
| Seen-category-combination | test category set exactly appears in train | 275 | 0.818 |  |
| Novel-category-combination | test category set does not exactly appear in train | 61 | 0.182 |  |
| Seen-API-combination | test API set exactly appears in train | 206 | 0.613 |  |
| Novel-API-combination | test API set does not exactly appear in train | 130 | 0.387 |  |
| Clear-requirement | requirement_clarity_status == Clear-requirement | 84 | 0.250 |  |
| Middle-requirement | requirement_clarity_status == Middle-requirement | 168 | 0.500 |  |
| Ambiguous-requirement | requirement_clarity_status == Ambiguous-requirement | 84 | 0.250 |  |
| Exact-duplicate-description | duplicate_description_status == Exact-duplicate-description | 2 | 0.006 |  |
| Near-duplicate-description | duplicate_description_status == Near-duplicate-description | 0 | 0.000 |  |
| Duplicate-free-description | duplicate_description_status == Duplicate-free-description | 334 | 0.994 |  |

## Description Length
- word_count_bin is now included in representative_with_length stratification.
| method | size | feasible | candidate_file | max_delta_word_count_bin |
| --- | --- | --- | --- | --- |
| representative_with_length | 50 | yes | subset_candidates/representative_with_length_50_ids.txt | 0.013 |
| representative_with_length | 100 | yes | subset_candidates/representative_with_length_100_ids.txt | 0.018 |
| representative_with_length | 150 | yes | subset_candidates/representative_with_length_150_ids.txt | 0.020 |
| representative_with_length | 200 | yes | subset_candidates/representative_with_length_200_ids.txt | 0.022 |
| representative_with_length | 250 | yes | subset_candidates/representative_with_length_250_ids.txt | 0.008 |
| representative_with_length | 336 | yes | subset_candidates/representative_with_length_336_ids.txt | 0.000 |

## Oracle Candidate Size
- category candidate-size quantiles: q25=40.000, q75=73.000
- large oracle candidate samples: 34 (0.101)
- Large candidate sets are useful as an explanatory axis for E-F1 analysis because they approximate API-layer ambiguity after a correct category or branch decision.

## Combination Novelty
- train-based novelty analysis: enabled
- category_combination_novel samples: 7 (0.021)
- api_combination_novel samples: 44 (0.131)

## Requirement Clarity
- API-description TF-IDF analysis: enabled
- similarity_margin_max quantiles: q25=-0.104023, q75=0.022807
- ambiguous requirement samples: 84 (0.250)

## Duplicate Analysis
- near-duplicate threshold: 0.9
- duplicate or near-duplicate pairs: 1

## Updated Subset Feasibility
### representative_with_length
| method | size | feasible | candidate_file | max_delta_word_count_bin | max_delta_oracle_candidate_size_bin | max_delta_category_combination_status | max_delta_api_combination_status | max_delta_requirement_clarity_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| representative_with_length | 50 | yes | subset_candidates/representative_with_length_50_ids.txt | 0.013 | 0.054 | 0.082 | 0.107 | 0.110 |
| representative_with_length | 100 | yes | subset_candidates/representative_with_length_100_ids.txt | 0.018 | 0.074 | 0.041 | 0.027 | 0.030 |
| representative_with_length | 150 | yes | subset_candidates/representative_with_length_150_ids.txt | 0.020 | 0.087 | 0.048 | 0.004 | 0.030 |
| representative_with_length | 200 | yes | subset_candidates/representative_with_length_200_ids.txt | 0.022 | 0.048 | 0.002 | 0.039 | 0.015 |
| representative_with_length | 250 | yes | subset_candidates/representative_with_length_250_ids.txt | 0.008 | 0.037 | 0.007 | 0.021 | 0.024 |
| representative_with_length | 336 | yes | subset_candidates/representative_with_length_336_ids.txt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### representative_extended
| method | size | feasible | candidate_file | max_delta_word_count_bin | max_delta_oracle_candidate_size_bin | max_delta_category_combination_status | max_delta_api_combination_status | max_delta_requirement_clarity_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| representative_extended | 50 | yes | subset_candidates/representative_extended_50_ids.txt | 0.059 | 0.081 | 0.102 | 0.147 | 0.110 |
| representative_extended | 100 | yes | subset_candidates/representative_extended_100_ids.txt | 0.037 | 0.051 | 0.052 | 0.067 | 0.040 |
| representative_extended | 150 | yes | subset_candidates/representative_extended_150_ids.txt | 0.014 | 0.048 | 0.035 | 0.018 | 0.017 |
| representative_extended | 200 | yes | subset_candidates/representative_extended_200_ids.txt | 0.013 | 0.036 | 0.037 | 0.034 | 0.040 |
| representative_extended | 250 | yes | subset_candidates/representative_extended_250_ids.txt | 0.012 | 0.009 | 0.014 | 0.020 | 0.040 |
| representative_extended | 336 | yes | subset_candidates/representative_extended_336_ids.txt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Recommended Subset Strategy
- Main experiment: representative_with_length is the safest default because it preserves the original stratification and adds description length.
- Richer diagnostic experiment: representative_extended adds oracle candidate size and category-combination novelty when the target size remains stable.
- Auxiliary diagnostics: use large_oracle_candidate_ids.txt, novel_category_combination_ids.txt, novel_api_combination_ids.txt, ambiguous_requirement_ids.txt, and near_duplicate_description_ids.txt for focused checks.
- Very small condition groups should be treated as diagnostic slices rather than balanced main subsets.

## Notes
- branch candidate-size analysis: enabled
- output directory: /home/nakagawa/lab-MLMLC/codex/analysis_test_subset_conditions
