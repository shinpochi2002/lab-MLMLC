# Length x Choice Test Subset Analysis

## Overview
- total samples: 336
- document length bins: short=1-20 words, medium=21-50 words, long=51+ words.
- oracle candidate size bins: small=1-20 APIs, medium=21-100 APIs, large=101+ APIs.
- These axes separate input information volume from API-layer choice-set size after a correct category decision.

## 3x3 Cross Table
| word_count_bin | oracle_candidate_size_bin | count | ratio |
| --- | --- | --- | --- |
| short | small | 17 | 0.051 |
| short | medium | 100 | 0.298 |
| short | large | 11 | 0.033 |
| medium | small | 26 | 0.077 |
| medium | medium | 132 | 0.393 |
| medium | large | 19 | 0.057 |
| long | small | 6 | 0.018 |
| long | medium | 21 | 0.062 |
| long | large | 4 | 0.012 |

### Pivot Counts
| word_count_bin | small | medium | large | total |
| --- | --- | --- | --- | --- |
| short | 17 | 100 | 11 | 128 |
| medium | 26 | 132 | 19 | 177 |
| long | 6 | 21 | 4 | 31 |

## 2x2 Integrated Subsets
| subset_name | condition_definition | count | ratio | feasibility_label |
| --- | --- | --- | --- | --- |
| InfoLimited_SmallChoice | word_count_bin == short and oracle_candidate_size_bin == small | 17 | 0.051 | case_study_only |
| InfoLimited_LargeChoice | word_count_bin == short and oracle_candidate_size_bin in {medium, large} | 111 | 0.330 | main_or_comparison_ready |
| InfoRich_SmallChoice | word_count_bin in {medium, long} and oracle_candidate_size_bin == small | 32 | 0.095 | diagnostic_only |
| InfoRich_LargeChoice | word_count_bin in {medium, long} and oracle_candidate_size_bin in {medium, large} | 176 | 0.524 | main_or_comparison_ready |

## Subset Statistics
| subset_name | count | ratio | avg_word_count | avg_sum_oracle_candidate_size | avg_num_gt_apis | avg_num_gt_categories | seen_only_ratio | multi_category_ratio | ambiguous_requirement_ratio | feasibility_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_small | 17 | 0.051 | 14.588 | 12.059 | 1.118 | 1.059 | 1.000 | 0.059 | 0.176 | case_study_only |
| short_medium | 100 | 0.298 | 13.910 | 55.610 | 1.590 | 1.430 | 0.910 | 0.380 | 0.270 | main_or_comparison_ready |
| short_large | 11 | 0.033 | 15.636 | 130.909 | 4.273 | 3.636 | 0.818 | 1.000 | 0.545 | case_study_only |
| medium_small | 26 | 0.077 | 31.115 | 9.154 | 1.269 | 1.154 | 0.846 | 0.154 | 0.231 | case_study_only |
| medium_medium | 132 | 0.393 | 30.856 | 58.432 | 2.114 | 1.689 | 0.856 | 0.500 | 0.242 | main_or_comparison_ready |
| medium_large | 19 | 0.057 | 32.737 | 134.368 | 5.158 | 3.947 | 0.526 | 1.000 | 0.158 | case_study_only |
| long_small | 6 | 0.018 | 101.667 | 9.500 | 1.833 | 1.667 | 0.333 | 0.500 | 0.000 | case_study_only |
| long_medium | 21 | 0.062 | 73.429 | 56.333 | 2.571 | 2.048 | 0.667 | 0.857 | 0.333 | case_study_only |
| long_large | 4 | 0.012 | 58.750 | 127.500 | 4.250 | 4.000 | 1.000 | 1.000 | 0.000 | case_study_only |
| InfoLimited_SmallChoice | 17 | 0.051 | 14.588 | 12.059 | 1.118 | 1.059 | 1.000 | 0.059 | 0.176 | case_study_only |
| InfoLimited_LargeChoice | 111 | 0.330 | 14.081 | 63.072 | 1.856 | 1.649 | 0.901 | 0.441 | 0.297 | main_or_comparison_ready |
| InfoRich_SmallChoice | 32 | 0.095 | 44.344 | 9.219 | 1.375 | 1.250 | 0.750 | 0.219 | 0.188 | diagnostic_only |
| InfoRich_LargeChoice | 176 | 0.524 | 36.773 | 67.949 | 2.545 | 2.028 | 0.801 | 0.608 | 0.239 | main_or_comparison_ready |

## Feasibility
- main_or_comparison_ready: short_medium, medium_medium, InfoLimited_LargeChoice, InfoRich_LargeChoice
- auxiliary_ready: none
- diagnostic_only: InfoRich_SmallChoice
- case_study_only: short_small, short_large, medium_small, medium_large, long_small, long_medium, long_large, InfoLimited_SmallChoice

## Recommended Usage
- main experiment: keep using representative_with_length for balanced global evaluation.
- auxiliary experiment: use 2x2 integrated subsets whose feasibility label is main_or_comparison_ready or auxiliary_ready.
- diagnostic analysis: use 3x3 cells with at least 30 samples.
- case study: use cells below 30 samples qualitatively instead of making stable F1 claims.

## Hypotheses
- Hypothesis 1: short documents with large choice sets may have lower F1 because little input information is paired with many possible APIs.
- Hypothesis 2: rich documents with small choice sets may have higher F1 because more input information is paired with fewer possible APIs.
- Hypothesis 3: length and choice-set size may change the relationship between E and F1; InfoLimited_LargeChoice may expose early commitment effects in the hierarchy.
- These are hypotheses only; they should not be treated as conclusions until model evaluation is run.

## Final Judgement Checks
- InfoLimited_LargeChoice count: 111
- InfoRich_SmallChoice count: 32
- The 2x2 subsets are the practical comparison units when 3x3 cells are sparse.

## Notes
- warnings: 0
- no missing required columns were encountered.
