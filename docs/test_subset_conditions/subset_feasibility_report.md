# Test Dataset Subset Condition Analysis

## Dataset Overview
- total samples: 336
- average number of ground-truth APIs: 2.134
- median/min/max ground-truth APIs: 2.000 / 1 / 12
- average number of ground-truth categories: 1.780
- median/min/max ground-truth categories: 1.000 / 1 / 7
- average word count: 28.875
- median/min/max word count: 24.000 / 4 / 203

## Condition Counts
| condition_name | condition_definition | count | ratio | notes |
| --- | --- | --- | --- | --- |
| Total samples | all test samples | 336 | 1.000 |  |
| Missing description | description is empty after string conversion | 0 | 0.000 |  |
| Missing ground-truth APIs | num_gt_apis == 0 | 0 | 0.000 |  |
| Missing API category labels | at least one gt API has no API-to-category mapping | 0 | 0.000 |  |
| 1 API | num_gt_apis bin == 1 API | 152 | 0.452 |  |
| 2 APIs | num_gt_apis bin == 2 APIs | 98 | 0.292 |  |
| 3 APIs | num_gt_apis bin == 3 APIs | 38 | 0.113 |  |
| 4 APIs | num_gt_apis bin == 4 APIs | 23 | 0.068 |  |
| 5+ APIs | num_gt_apis bin == 5+ APIs | 25 | 0.074 |  |
| 1 category | num_gt_categories bin == 1 category | 172 | 0.512 |  |
| 2 categories | num_gt_categories bin == 2 categories | 108 | 0.321 |  |
| 3 categories | num_gt_categories bin == 3 categories | 32 | 0.095 |  |
| 4 categories | num_gt_categories bin == 4 categories | 13 | 0.039 |  |
| 5+ categories | num_gt_categories bin == 5+ categories | 11 | 0.033 |  |
| Description short | 1 <= word_count <= 20 | 128 | 0.381 |  |
| Description medium | 21 <= word_count <= 50 | 177 | 0.527 |  |
| Description long | word_count >= 51 | 31 | 0.092 |  |
| Seen-only | seen_status == Seen-only | 282 | 0.839 |  |
| Partially-unseen | seen_status == Partially-unseen | 40 | 0.119 |  |
| All-unseen | seen_status == All-unseen | 14 | 0.042 |  |
| Rare-including | at least one gt API has train frequency 1-2 | 54 | 0.161 |  |
| Medium-including | at least one gt API has train frequency 3-9 | 63 | 0.188 |  |
| Frequent-only | all gt APIs have train frequency >= 10 | 201 | 0.598 |  |
| Low-frequency-api-including (quantile) | api_frequency_quantile_status == Low-frequency-api-including | 79 | 0.235 |  |
| Middle-frequency-api-including (quantile) | api_frequency_quantile_status == Middle-frequency-api-including | 17 | 0.051 |  |
| High-frequency-api-only (quantile) | api_frequency_quantile_status == High-frequency-api-only | 240 | 0.714 |  |
| Low-frequency-category-including | category_frequency_status == Low-frequency-category-including | 9 | 0.027 |  |
| Middle-frequency-category-including | category_frequency_status == Middle-frequency-category-including | 31 | 0.092 |  |
| High-frequency-category-only | category_frequency_status == High-frequency-category-only | 296 | 0.881 |  |
| Single-category | num_gt_categories == 1 | 172 | 0.512 |  |
| Multi-category | num_gt_categories >= 2 | 164 | 0.488 |  |
| Multi-category: 2 categories | Multi-category: 2 categories | 108 | 0.321 |  |
| Multi-category: 3 categories | Multi-category: 3 categories | 32 | 0.095 |  |
| Multi-category: 4+ categories | Multi-category: 4+ categories | 24 | 0.071 |  |
| low_e_4_4_8: Single-branch at layer1 | single_or_multi_branch_layer1 == Single-branch | 205 | 0.610 |  |
| low_e_4_4_8: Multi-branch at layer1 | single_or_multi_branch_layer1 == Multi-branch | 131 | 0.390 |  |
| low_e_4_4_8: Single-branch at layer2 | single_or_multi_branch_layer2 == Single-branch | 183 | 0.545 |  |
| low_e_4_4_8: Multi-branch at layer2 | single_or_multi_branch_layer2 == Multi-branch | 153 | 0.455 |  |
| middle_e_12_4_3: Single-branch at layer1 | single_or_multi_branch_layer1 == Single-branch | 179 | 0.533 |  |
| middle_e_12_4_3: Multi-branch at layer1 | single_or_multi_branch_layer1 == Multi-branch | 157 | 0.467 |  |
| middle_e_12_4_3: Single-branch at layer2 | single_or_multi_branch_layer2 == Single-branch | 174 | 0.518 |  |
| middle_e_12_4_3: Multi-branch at layer2 | single_or_multi_branch_layer2 == Multi-branch | 162 | 0.482 |  |
| high_e_51-3: Single-branch at layer1 | single_or_multi_branch_layer1 == Single-branch | 175 | 0.521 |  |
| high_e_51-3: Multi-branch at layer1 | single_or_multi_branch_layer1 == Multi-branch | 161 | 0.479 |  |
| high_e_51-3: Single-branch at layer2 | single_or_multi_branch_layer2 == Single-branch | 0 | 0.000 |  |
| high_e_51-3: Multi-branch at layer2 | single_or_multi_branch_layer2 == Multi-branch | 0 | 0.000 |  |
| Condition A: Representative-friendly | 1 <= num_gt_apis <= 3 and 1 <= num_gt_categories <= 2 | 267 | 0.795 |  |
| Condition B: Multi-category | num_gt_categories >= 2 | 164 | 0.488 |  |
| Condition C: Difficult rare multi-category | num_gt_categories >= 2 and Rare-including | 40 | 0.119 |  |
| Condition D: Hierarchy-sensitive | num_gt_categories >= 2 and multi-branch in middle_e_12_4_3 | 162 | 0.482 |  |
| Condition E: Stable evaluation | Seen-only and 1 <= num_gt_apis <= 3 and 1 <= num_gt_categories <= 2 | 240 | 0.714 |  |

## Subset Size Feasibility
| size | feasible | available | candidate_file | max_delta_num_gt_apis | max_delta_num_gt_categories | max_delta_api_frequency_status | max_delta_single_or_multi_category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | yes | 336 | subset_candidates/representative_50_ids.txt | 0.034 | 0.019 | 0.022 | 0.008 |
| 100 | yes | 336 | subset_candidates/representative_100_ids.txt | 0.024 | 0.008 | 0.012 | 0.008 |
| 150 | yes | 336 | subset_candidates/representative_150_ids.txt | 0.028 | 0.012 | 0.005 | 0.008 |
| 200 | yes | 336 | subset_candidates/representative_200_ids.txt | 0.007 | 0.013 | 0.008 | 0.003 |
| 250 | yes | 336 | subset_candidates/representative_250_ids.txt | 0.008 | 0.004 | 0.006 | 0.004 |
| 336 | yes | 336 | subset_candidates/representative_336_ids.txt | 0.000 | 0.000 | 0.000 | 0.000 |

## Important Findings
- Conditions with at least 50 samples: Condition A: Representative-friendly, Condition B: Multi-category, Condition D: Hierarchy-sensitive, Condition E: Stable evaluation
- Conditions below 50 samples: Condition C: Difficult rare multi-category
- Candidate ID lists were generated deterministically with stratification over API-count, category-count, API-frequency status, and single/multi-category labels.

## Recommended Subset Conditions
- Representative subset: Condition A when the target size fits; otherwise use the generated representative ID lists.
- Rare / long-tail subset: Condition C when train frequency is available.
- Multi-category subset: Condition B.
- Stable evaluation subset: Condition E when train frequency is available; otherwise Condition A.

## Notes
- train data: /home/nakagawa/lab-MLMLC/study_data_mashup_apis.json
- API frequency quantile thresholds: q25=1.000, q75=2.000
- category frequency source: train
- category frequency quantile thresholds: q25=1.000, q75=16.250
- hierarchy configurations: low_e_4_4_8, middle_e_12_4_3, high_e_51-3
- primary hierarchy for sample_condition_flags and Condition D: middle_e_12_4_3
- output directory: /home/nakagawa/lab-MLMLC/codex/analysis_test_subset_conditions
