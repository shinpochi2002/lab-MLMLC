# Phase 1/2 implementation report

Category services: 21907; API positives: 4855; groups: 21884.
API branch positive rows: 6514; single-service API classifiers: 10; singleton API labels: 87.

No training, inference, checkpoint replacement, hierarchy rebuilding or negative-record generation was executed.
CSV classifier rows repeat the same 152 API classifiers for 16 patterns for traceability; do not add them as distinct services.
Training/ES unavailability and zero-positive labels are recorded; no fallback algorithm is implemented.
Labels positive in every Train record may have zero pos_weight in legacy trainers. This is a Phase 3 warning, not repaired here.
Reference checks are assertions only: reconstructed data are never adjusted to expected counts.
Pre-existing files: declared input/code SHA-256 plus repository-wide file metadata checks; model contents are not hashed.

## Fold service counts

| Fold | Outer Train | Gradient Train | Inner ES | Outer Validation |
| --- | --- | --- | --- | --- |
| fold_01 | 17522 | 14020 | 3502 | 4385 |
| fold_02 | 17528 | 14022 | 3506 | 4379 |
| fold_03 | 17529 | 14024 | 3505 | 4378 |
| fold_04 | 17528 | 14026 | 3502 | 4379 |
| fold_05 | 17521 | 14017 | 3504 | 4386 |

## Rare classifiers

```json
[
  {
    "fold": "fold_01",
    "full_development_single_positive_api_labels": 87,
    "train_zero_positive_api_labels": 18,
    "outer_train_zero_positive_api_labels": 18,
    "validation_zero_positive_api_labels": 417,
    "early_stopping_zero_positive_api_labels": 620,
    "train_single_positive_api_labels": 250,
    "shared_api_classifiers_train_empty": 2,
    "shared_api_classifiers_early_stopping_unavailable": 78,
    "category_classifiers_train_empty_all_patterns": 0,
    "category_classifiers_early_stopping_unavailable_all_patterns": 0
  },
  {
    "fold": "fold_02",
    "full_development_single_positive_api_labels": 87,
    "train_zero_positive_api_labels": 18,
    "outer_train_zero_positive_api_labels": 18,
    "validation_zero_positive_api_labels": 401,
    "early_stopping_zero_positive_api_labels": 629,
    "train_single_positive_api_labels": 256,
    "shared_api_classifiers_train_empty": 2,
    "shared_api_classifiers_early_stopping_unavailable": 79,
    "category_classifiers_train_empty_all_patterns": 0,
    "category_classifiers_early_stopping_unavailable_all_patterns": 0
  },
  {
    "fold": "fold_03",
    "full_development_single_positive_api_labels": 87,
    "train_zero_positive_api_labels": 17,
    "outer_train_zero_positive_api_labels": 17,
    "validation_zero_positive_api_labels": 395,
    "early_stopping_zero_positive_api_labels": 631,
    "train_single_positive_api_labels": 262,
    "shared_api_classifiers_train_empty": 3,
    "shared_api_classifiers_early_stopping_unavailable": 82,
    "category_classifiers_train_empty_all_patterns": 0,
    "category_classifiers_early_stopping_unavailable_all_patterns": 0
  },
  {
    "fold": "fold_04",
    "full_development_single_positive_api_labels": 87,
    "train_zero_positive_api_labels": 17,
    "outer_train_zero_positive_api_labels": 17,
    "validation_zero_positive_api_labels": 417,
    "early_stopping_zero_positive_api_labels": 620,
    "train_single_positive_api_labels": 250,
    "shared_api_classifiers_train_empty": 2,
    "shared_api_classifiers_early_stopping_unavailable": 79,
    "category_classifiers_train_empty_all_patterns": 0,
    "category_classifiers_early_stopping_unavailable_all_patterns": 0
  },
  {
    "fold": "fold_05",
    "full_development_single_positive_api_labels": 87,
    "train_zero_positive_api_labels": 17,
    "outer_train_zero_positive_api_labels": 17,
    "validation_zero_positive_api_labels": 417,
    "early_stopping_zero_positive_api_labels": 628,
    "train_single_positive_api_labels": 249,
    "shared_api_classifiers_train_empty": 1,
    "shared_api_classifiers_early_stopping_unavailable": 80,
    "category_classifiers_train_empty_all_patterns": 0,
    "category_classifiers_early_stopping_unavailable_all_patterns": 0
  }
]
```

## Completion checks

See checks/validation_results.json. Runtime generation timestamps are intentionally variable; records, groups, fold assignments and CSV counts are deterministic.
No original paper manuscript was located during input inventory; terminology and structure IDs follow the existing README, audit reports and inference code.
