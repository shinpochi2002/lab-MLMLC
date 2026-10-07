"""Check that hierarchy evaluation cannot silently return to the full dataset."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "run_all_existing_hierarchy_patterns.py"
SPEC = importlib.util.spec_from_file_location("hierarchy_split_runner", SCRIPT_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class HierarchyEvaluationSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.split_dir = self.root / "splits"
        self.results_dir = self.root / "results"
        self.sources = []
        for name in ("layer3_12_4_3", "layer3_16_4_3"):
            source = self.root / "original" / f"{name}_api_name_masked.json"
            self.sources.append(source)
            for split, count in (("full", 4), ("validation", 1), ("test", 3)):
                path = source if split == "full" else self.split_dir / split / source.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({f"sample-{i}": {"description": "uses <API_NAME>"} for i in range(count)}))

    def discover(self, results_dir: Path, dataset: str):
        return [
            runner.make_task(
                layer=3,
                target=source.name.split("_api_name_masked")[0].removeprefix("layer3_"),
                layer1_category_num="12" if index == 0 else "16",
                script_name="mashup_dis_nosp_3layer_api.py",
                eval_path=source,
                results_dir=results_dir,
            )
            for index, source in enumerate(self.sources)
        ]

    def resolve(self, source: Path, split: str, split_dir: Path) -> Path:
        path = source if split == "full" else split_dir / split / source.name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def run_cli(self, *args: str):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", [str(SCRIPT_PATH), "--dry-run", "--split-dir", str(self.split_dir), *args]),
            patch.object(runner, "discover_tasks", side_effect=self.discover) as discovery,
            patch.object(runner, "resolve_split_path", side_effect=self.resolve) as resolver,
            patch.object(runner, "DEFAULT_MASKED_RESULTS_DIR", self.results_dir),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = runner.main()
        return status, stdout.getvalue(), stderr.getvalue(), discovery, resolver

    def test_default_test_split_uses_remaining_records_and_separate_output(self):
        status, output, errors, discovery, resolver = self.run_cli()
        self.assertEqual(status, 0, errors)
        self.assertIn("Dataset split: test", output)
        self.assertIn("samples=3", output)
        self.assertNotIn("samples=4", output)
        self.assertEqual(discovery.call_args.args[0], self.results_dir / "test")
        self.assertEqual([call.args[1] for call in resolver.call_args_list], ["test", "test"])

    def test_validation_split_uses_held_out_records_and_separate_output(self):
        status, output, errors, discovery, resolver = self.run_cli("--split", "validation")
        self.assertEqual(status, 0, errors)
        self.assertIn("samples=1", output)
        self.assertEqual(discovery.call_args.args[0], self.results_dir / "validation")
        self.assertEqual([call.args[1] for call in resolver.call_args_list], ["validation", "validation"])

    def test_full_requires_explicit_choice_and_preserves_legacy_output_path(self):
        status, output, errors, discovery, resolver = self.run_cli("--split", "full")
        self.assertEqual(status, 0, errors)
        self.assertIn("samples=4", output)
        self.assertEqual(discovery.call_args.args[0], self.results_dir)
        self.assertEqual([call.args[1] for call in resolver.call_args_list], ["full", "full"])

    def test_patterns_select_exact_task_before_resolving_data(self):
        status, output, errors, _, resolver = self.run_cli("--patterns", "layer3_12_4_3")
        self.assertEqual(status, 0, errors)
        self.assertIn("Found 1 runnable task(s).", output)
        self.assertNotIn("layer3_16_4_3", output)
        resolver.assert_called_once_with(self.sources[0], "test", self.split_dir)

    def test_unknown_patterns_are_rejected(self):
        with self.assertRaises(SystemExit) as exc:
            self.run_cli("--patterns", "layer3_12")
        self.assertEqual(exc.exception.code, 2)

    def test_missing_split_does_not_fall_back_to_full_dataset(self):
        (self.split_dir / "test" / self.sources[0].name).unlink()
        status, output, errors, _, resolver = self.run_cli()
        self.assertEqual(status, 1)
        self.assertIn(str(self.split_dir / "test"), errors)
        self.assertNotIn("Found", output)
        resolver.assert_called_once_with(self.sources[0], "test", self.split_dir)

    def test_condition_subsets_require_explicit_full_split(self):
        with self.assertRaises(SystemExit) as exc:
            self.run_cli("--condition-2x2", "--dataset", "unmasked")
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
