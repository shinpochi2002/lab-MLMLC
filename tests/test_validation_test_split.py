"""Checks for split reproducibility, isolation, and artifact integrity."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import evaluation_split
from data_prep.create_validation_test_split import create_split, split_keys


class ValidationTestSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "source"
        self.source = self.sources / "layer1_direct_api" / "test_drop_unknown_samples.json"
        self.pattern = self.sources / "by_pattern" / "2_3" / "test_drop_unknown_samples.json"
        self.train = self.root / "train.json"
        self.output = self.root / "split"
        self.data = {f"mashup-{i}": [["category"], f"text {i}", ["api"], i] for i in range(8)}
        self.write(self.source, self.data)
        self.write(self.pattern, {
            key: {"categories": entry[0], "description": entry[1], "apis": entry[2],
                  "id": entry[3], "gt_layer1": ["1"], "gt_layer2": ["category"]}
            for key, entry in reversed(list(self.data.items()))
        })
        self.write(self.train, {"training-only": [["category"], "training text", ["api"], 100]})

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def create(self, **kwargs):
        return create_split(self.sources, self.source, self.train, self.output, 3, **kwargs)

    def test_partition_matches_every_variant_and_preserves_records_and_order(self):
        original = self.source.read_bytes()
        manifest = self.create()
        validation = set(manifest["splits"]["validation"]["keys"])
        test = set(manifest["splits"]["test"]["keys"])
        self.assertEqual((len(validation), len(test)), (3, 5))
        self.assertFalse(validation & test)
        self.assertEqual(validation | test, set(self.data))
        for source in (self.source, self.pattern):
            full = json.loads(source.read_text())
            for split, keys in (("validation", validation), ("test", test)):
                with patch.object(evaluation_split, "SOURCE_ROOT", self.sources):
                    path = evaluation_split.resolve_split_path(source, split, self.output)
                subset = json.loads(path.read_text())
                self.assertEqual(list(subset), [key for key in full if key in keys])
                self.assertEqual(subset, {key: value for key, value in full.items() if key in keys})
        self.assertEqual(self.source.read_bytes(), original)

    def test_sampling_is_stable_when_source_key_order_changes(self):
        reversed_data = dict(reversed(list(self.data.items())))
        self.assertEqual(split_keys(self.data, 3, 42), split_keys(reversed_data, 3, 42))

    def test_repeat_is_idempotent_and_repartition_is_rejected(self):
        manifest = self.create()
        self.assertEqual(manifest, self.create())
        before = {path: path.read_bytes() for path in self.output.rglob("*.json")}
        with self.assertRaises(FileExistsError):
            self.create(seed=7)
        self.assertEqual(before, {path: path.read_bytes() for path in self.output.rglob("*.json")})

    def test_training_overlap_fails_before_writing(self):
        self.write(self.train, {"mashup-0": self.data["mashup-0"]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_inconsistent_variant_fails_before_writing(self):
        data = json.loads(self.pattern.read_text())
        data["mashup-0"]["id"] = 999
        self.write(self.pattern, data)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_changed_or_missing_subset_never_falls_back_to_full(self):
        self.create()
        with patch.object(evaluation_split, "SOURCE_ROOT", self.sources):
            subset = evaluation_split.resolve_split_path(self.source, "test", self.output)
            self.write(subset, self.data)
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluation_split.resolve_split_path(self.source, "test", self.output)
            subset.unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation_split.resolve_split_path(self.source, "test", self.output)

    def test_changed_source_is_rejected(self):
        self.create()
        self.source.write_text(self.source.read_text() + "\n")
        with patch.object(evaluation_split, "SOURCE_ROOT", self.sources):
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluation_split.resolve_split_path(self.source, "validation", self.output)

    def test_invalid_size_leaves_output_unwritten(self):
        for size in (0, -1, len(self.data), len(self.data) + 1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                create_split(self.sources, self.source, self.train, self.output, size)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
