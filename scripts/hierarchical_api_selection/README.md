# Hierarchical API Selection: Phase 1 and Phase 2

This independent standard-library Python workflow prepares requirements-driven API recommendation data without importing or editing existing training/inference modules. The fixed spaces are 909 APIs, 152 categories and 16 existing hierarchy patterns. It restores non-evaluation overwritten Mashups, excludes fixed evaluation entities and matching normalized text, and produces shared group-aware development splits. It never trains, runs inference, rebuilds clusters, generates negative records, or replaces checkpoints.

## Commands

Run from repository root with Python 3.10 or later. No dependency installation is needed.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/prepare_training_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/create_group_folds.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/summarize_data_usage.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/validate_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1 --write-report
```

Every creation step uses exclusive output creation and stops if its own outputs exist. There is no force/overwrite/recovery option. To regenerate, use a different nonexistent output root; do not remove or replace the original outputs. `--repository-root` optionally points to another checkout containing the same relative input paths. All generated input/model references are relative to that repository root.

| Entry point | Reads | Creates |
| --- | --- | --- |
| prepare_training_dataset.py | pipeline.SOURCES, fixed 100/236 files and 16 fixed hierarchy JSON definitions | manifests/source_manifest.json, dataset_manifest.json, exclusion_manifest.json, label_schema.json, existing_file_snapshot.json; source_records/*.json |
| create_group_folds.py | Phase 1 records/groups/schema and original input integrity | splits/group_assignments.csv, fold_assignments.csv, fold_01..fold_05/*_groups.json; manifests/split_manifest.json |
| summarize_data_usage.py | Shared records/schema/groups/folds | README.md, manifests/classifier_manifest.json, reports/*.csv and *.md |
| validate_dataset.py | All artifacts and declared sources | Default: no writes; --write-report: checks/validation_results.json, refusing an existing checks directory |
| pipeline.py | Shared helper implementation | No standalone execution or import-time outputs |

Repeat validation without writing:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/validate_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/tests -v
```

The generated dataset README documents the full schema, source roles, exclusions, restoration evidence, grouping, algorithm, minority conditions, counts, verification and Phase 3 handoff. Reports distinguish outer Train from gradient Train and inner Early Stopping. API-positive records are reused services, not extra data; API reports repeat a common classifier for each of the 16 configurations only for traceability.

## Algorithm and safety

Raw Mashup/API IDs must be unique within their original source and become `mashup:<id>` / `api:<id>`. Source revisions are bound by SHA-256. Descriptions and original used API lists are retained exactly. Category labels preserve the existing first-category rule. Noncandidate APIs have null API targets and are never represented as negative examples.

Comparison text is HTML-entity decoded, Unicode NFKC normalized and whitespace-collapsed, preserving case. Identical nonempty text connects entities across source types; labels are not merged. Group IDs hash sorted entity membership. Seed-42 rarest-remaining-label allocation uses category/API/source deficits and hard group capacities. Outer proportions are 1/5 each; each outer Train independently uses group proportions 0.8/0.2 with the same seed. Group constraints take precedence over rare-label balance; no samples are moved after allocation.

The code checks baseline expectations as assertions against measured source data, never as trimming/padding targets. Any provenance, source, fixed-evaluation or count mismatch stops. Minority cases produce count/flag reports, not repaired examples or fallback learners. Existing file preservation is checked with source/code SHA-256 and repository-wide file metadata; hundreds of GB of existing checkpoint content are not rehashed.

Phase 3 must decide negative sampling, zero-positive/zero-weight losses and unsupported classifiers. Those learning algorithms are deliberately absent here. The fixed hierarchy paths, full label schemas, original identifiers and positive API/category mapping are saved for that future work.
