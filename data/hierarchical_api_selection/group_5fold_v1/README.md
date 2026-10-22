# Hierarchical API Selection — Group 5-Fold v1 (Phase 1/2)

Mashup requirements -> 909 APIs; 152 fixed categories, 16 existing Balanced K-means structures. No hierarchy rebuilding, training or inference.

## Data usage

Category services: 21907; API positive subset: 4855; groups: 21884; API branch-positive rows: 6514.
API positives are reused category services, not additional independent examples. Full counts: reports/data_usage_summary.csv.
61 overwritten non-evaluation Mashups are restored from immutable sources. 336 evaluation Mashup entities, text aliases and empty text are excluded before branch expansion; four distinct same-name APIs are retained. Fixed selection Validation 100 and final Test 236 are untouched, outside every fold.

## Files and schema

- manifests/source_manifest.json: relative input paths, SHA-256, source counts, code hashes, Git commit, UTC time and conditions.
- manifests/existing_file_snapshot.json: all pre-existing file metadata, including models/results. SHA-256 covers declared inputs/code; model contents are not hashed.
- manifests/dataset_manifest.json: counts and reusable dataset paths; exclusion_manifest.json: entity-specific reasons, retained API homonyms, evaluation identity/text fingerprints.
- manifests/label_schema.json: global 909 API/152 category order, API-category mapping, fixed category_paths, existing model/inference correspondence for 16 patterns.
- manifests/classifier_manifest.json: per-parent outputs and membership rules. All API classifiers are shared and use fixed catalog subsets, never fold-observed output lists.
- source_records/unified_services.json and category_classification.json: identical lists of accepted services.
- source_records/api_classification_positive.json: exact eligible subset; restored_mashups.json: 61 recovered services with original provenance; groups.json: group members and input hashes.
- splits/group_assignments.csv: one row/group, validation_fold, entity list, source counts. fold_assignments.csv: one row/entity, common group and outer validation fold.
- splits/fold_01..fold_05/: outer_train_groups.json is the outer development Train; train_groups.json excludes inner ES; early_stopping_groups.json and validation_groups.json are disjoint. Lists are group IDs, not copied training records.
- reports/fold_distribution.csv: source/task/service/group counts for each role; category_label_distribution.csv and api_label_distribution.csv: full label support including zeros.
- reports/classifier_sample_counts.csv: 16 patterns × layers/parent classifiers × folds/roles; API positive/known-Mashup negative candidate counts. No actual negatives are generated.
- reports/unavailable_classifiers.csv and rare_classifier_summary.csv: Train/ES unavailability, singleton/zero-support labels. Fallback learning is not implemented.
- reports/data_leakage_audit.md and implementation_report.md: evidence and limits. checks/validation_results.json: automated checks and generated-artifact hashes.

Every service has entity_id/source_type/source_id/source_key/source_path/teacher_source_path/source_record_sha256, unchanged description, category_labels, original_source_categories, used_apis, api_labels, api_classification_eligible, api_category_pairs, normalized_text_sha256, restored_mashup and group_id.
Raw Mashup/API integer IDs are checked unique, namespaced as mashup:<id>/api:<id>. Source hashes bind those IDs to the source revision. used_apis preserves the source Mashup list; api_labels is its sorted set or the candidate API's own name. Noncandidate APIs have api_labels=null and eligible=false, never an all-zero negative target.
Categories preserve the original rule: Mashup -> first category of each used API; API -> its first category. Duplicate category labels are represented as a set; descriptions are never normalized in the model input.

## Grouping and split algorithm

Comparison text: html.unescape -> Unicode NFKC -> str.split/join; case preserved; v1. Identical nonempty comparison input links records across types, even with different labels. Confirmed source identities cannot be duplicated; records sharing text keep separate labels. group_id = group:SHA256(canonical sorted entity IDs). Group members are not deleted or merged.
Deterministic rarest-remaining-multilabel deficit allocation uses category/API/source supports, hard group-count capacities with largest-remainder rounding, and seed 42. It approximates label balance, does not guarantee impossible rare-label stratification. Input group IDs are sorted before seed-42 shuffling. Inner splits independently reuse seed 42 on each outer Train's group subset, fraction 0.8/0.2. No after-the-fact sample moves.
All 16 structures and API subset inherit the same outer and inner groups. No individual classifier can repartition its examples. A record may train multiple branches in the same role; it may never cross roles in one fold. Fixed category paths are reused verbatim.

## Reproduce

From repository root, using Python >=3.10 and the standard library only (no dependency install):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/prepare_training_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/create_group_folds.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/summarize_data_usage.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/validate_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1 --write-report
```

These are creation commands, not overwrite commands. To regenerate, choose a different nonexistent --output-root (for example a separate temporary directory); never delete/replace existing artifacts. Each phase stops on existing outputs. No --force flag exists. Timestamps/output locations differ; record content/group/fold assignment is deterministic. Default validation is read-only and can be repeated:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/validate_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/tests -v
```

## Phase 3 handoff

Use group lists to select records; project category_labels through the fixed pattern's paths. For API category c, use only eligible records and target api_labels intersect that category's fixed catalog. Preserve entity/group/fold IDs. Negative candidates are known development Mashups not using that category, counted only here; generate actual negatives only in a future approved phase.
Train means gradient subset, ES means inner validation, Validation means outer CV evaluation. Fixed Validation100 selects hierarchy after CV; Test236 evaluates recommendation. Do not use either fixed set for ES, fitting or fold selection. Minority failures, train-positive-zero labels, and legacy zero-pos_weight cases require Phase 3 decisions; there is no automatic fix, moved data or learning fallback here.
