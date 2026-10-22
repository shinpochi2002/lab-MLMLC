# Data leakage audit

Evaluation membership and all normalized evaluation descriptions were excluded before grouping/expansion.

- Excluded fixed evaluation Mashup entities: 336 (includes four collision Mashups not restored).
- Additional text-only alias: 1; empty descriptions: 1.
- Retained distinct API entities sharing evaluation Mashup names: 4.
- Restored non-evaluation collision Mashups: 61.
- Normalization: html.unescape -> Unicode NFKC -> str.split/join; case preserved; v1. Labels were not combined for identical text.
- Fixed evaluation category GT differences: [{"fixed_categories":["shipping"],"mashup_derived_categories":["shipping","social"],"name":"trackthis"},{"fixed_categories":["blockchain"],"mashup_derived_categories":["social"],"name":"rosetta"}]. Evaluation files were retained.
- Fold roles are shared across all sources, all layers and all 16 structures; no sample moves or negative records.

Full automated integrity, source preservation and deterministic recomputation results: `../checks/validation_results.json`.
The fixed evaluation was used in historical experiments; this work does not erase that history.
