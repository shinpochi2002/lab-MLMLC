"""Selection must use complete, matching validation predictions only."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "select_hierarchy_from_validation.py"
SPEC = importlib.util.spec_from_file_location("validation_selector", SCRIPT_PATH)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.source_root = self.root / "sources"
        self.split_dir = self.root / "split"
        self.results_dir = self.root / "results"
        self.results_dir.mkdir()
        source_patch = patch.object(selector.evaluation_split, "SOURCE_ROOT", self.source_root)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        self.data = {
            "mashup-a": {"description": "requires \\ API\nwith quoted \"name\""},
            "mashup-b": {"description": "second request"},
        }
        manifest = {
            "schema_version": 1,
            "seed": 42,
            "source_count": 3,
            "splits": {
                "validation": {"count": 2, "keys": list(self.data)},
                "test": {"count": 1, "keys": ["test-only"]},
            },
            "datasets": {},
        }
        # Layer 1 has no usable data or CSV: the selector must ignore it.
        self.tasks = [{"name": "layer1_direct_api", "layer": 1}]
        for name, scores in (("layer2_a", [0.2, 0.4]), ("layer3_b", [0.5, 0.9])):
            relative_path = Path("by_pattern") / name / "dataset.json"
            source_path = self.source_root / relative_path
            data_path = self.split_dir / "validation" / relative_path
            for path, data in (
                (source_path, {**self.data, "test-only": {"description": "unused"}}),
                (data_path, self.data),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding="utf-8")
            manifest["datasets"][relative_path.as_posix()] = {
                "source_sha256": selector.evaluation_split.sha256_file(source_path),
                "sha256": {"validation": selector.evaluation_split.sha256_file(data_path)},
            }
            task = {
                "name": name,
                "layer": int(name[5]),
                "evaluation_data_path": str(data_path),
                "out_csv_path": str(self.results_dir / f"{name}_results.csv"),
            }
            self.tasks.append(task)
            self.write_rows(task, self.rows(scores))
        (self.split_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.write_task_manifest()

    def rows(self, scores):
        return [
            {"idx": idx, "text": entry["description"], "f1": score}
            for idx, (entry, score) in enumerate(zip(self.data.values(), scores))
        ]

    def write_rows(self, task, rows):
        with Path(task["out_csv_path"]).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["idx", "text", "f1"], escapechar="\\", quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

    def write_task_manifest(self):
        (self.results_dir / "test_data_manifest.json").write_text(json.dumps(self.tasks), encoding="utf-8")

    def test_best_validation_mean_selected_without_test_files(self):
        selection = selector.select_hierarchy(self.results_dir, self.split_dir)
        self.assertEqual(selection["selected_pattern"], "layer3_b")
        self.assertAlmostEqual(selection["sample_mean_f1"], 0.7)
        self.assertEqual(selection["candidate_count"], 2)
        self.assertEqual(selection["validation_count"], 2)
        self.assertEqual(selection["seed"], 42)
        self.assertEqual(selection["metric"], "sample_mean_f1")
        self.assertEqual(len(selection["split_manifest_sha256"]), 64)
        self.assertEqual([row["rank"] for row in selection["rankings"]], [1, 2])
        self.assertFalse((self.split_dir / "test").exists())

    def test_tie_break_uses_pattern_name_not_manifest_order(self):
        self.write_rows(self.tasks[2], self.rows([0.2, 0.4]))
        self.tasks.reverse()
        self.write_task_manifest()
        selection = selector.select_hierarchy(self.results_dir, self.split_dir)
        self.assertEqual(selection["selected_pattern"], "layer2_a")

    def test_incomplete_or_full_dataset_predictions_rejected(self):
        for rows in (self.rows([0.2]), self.rows([0.2, 0.4]) + [{"idx": 2, "text": "test-only", "f1": 1}]):
            with self.subTest(count=len(rows)):
                self.write_rows(self.tasks[1], rows)
                with self.assertRaisesRegex(ValueError, "CSV count"):
                    selector.select_hierarchy(self.results_dir, self.split_dir)

    def test_mismatched_indices_or_text_rejected(self):
        for column, value in (("idx", 1), ("text", "wrong request")):
            with self.subTest(column=column):
                rows = self.rows([0.2, 0.4])
                rows[0][column] = value
                self.write_rows(self.tasks[1], rows)
                with self.assertRaisesRegex(ValueError, "record order/text"):
                    selector.select_hierarchy(self.results_dir, self.split_dir)

    def test_invalid_f1_rejected(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(score=score):
                self.write_rows(self.tasks[1], self.rows([score, 0.4]))
                with self.assertRaisesRegex(ValueError, "finite"):
                    selector.select_hierarchy(self.results_dir, self.split_dir)

    def test_test_evaluation_path_rejected_before_reading(self):
        self.tasks[1]["evaluation_data_path"] = str(self.split_dir / "test" / "missing.json")
        self.write_task_manifest()
        with self.assertRaisesRegex(ValueError, "evaluation data must be inside"):
            selector.select_hierarchy(self.results_dir, self.split_dir)

    def test_missing_candidate_csv_rejected(self):
        Path(self.tasks[2]["out_csv_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            selector.select_hierarchy(self.results_dir, self.split_dir)

    def test_cli_writes_selection_record(self):
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--results-dir", str(self.results_dir), "--split-dir", str(self.split_dir)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(selector.main(), 0)
        selection = json.loads((self.results_dir / "selection.json").read_text())
        self.assertEqual(selection["selected_pattern"], "layer3_b")
        self.assertEqual(selection["selection_split"], "validation")


if __name__ == "__main__":
    unittest.main()
