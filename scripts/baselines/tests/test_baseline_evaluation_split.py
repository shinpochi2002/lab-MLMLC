import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


BASELINES_DIR = Path(__file__).resolve().parents[1]
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

import baseline_utils
import run_tfidf_ovr_baseline


class BaselineEvaluationSplitTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_default_loads_236_held_out_records(self):
        parser = argparse.ArgumentParser()
        baseline_utils.add_common_io_args(parser)
        path = baseline_utils.resolved_eval_path(parser.parse_args([]))
        self.assertEqual(path, baseline_utils.DEFAULT_EVAL_PATH)
        self.assertEqual(len(baseline_utils.load_mashup_records(path)), 236)

    def test_explicit_argument_overrides_environment_and_default(self):
        args = argparse.Namespace(eval_path="custom/validation.json")
        with mock.patch.dict(os.environ, {"EVALUATION_DATA_PATH": "custom/environment.json"}):
            with mock.patch.object(baseline_utils, "resolve_split_path") as resolve:
                self.assertEqual(
                    baseline_utils.resolved_eval_path(args),
                    Path("custom/validation.json"),
                )
                resolve.assert_not_called()

    def test_environment_overrides_default(self):
        args = argparse.Namespace(eval_path=None)
        with mock.patch.dict(os.environ, {"EVALUATION_DATA_PATH": "custom/environment.json"}):
            with mock.patch.object(baseline_utils, "resolve_split_path") as resolve:
                self.assertEqual(
                    baseline_utils.resolved_eval_path(args),
                    Path("custom/environment.json"),
                )
                resolve.assert_not_called()

    def test_missing_or_invalid_split_does_not_fall_back_to_full_data(self):
        args = argparse.Namespace(eval_path=None, target="16_4_3")
        for error_type in (FileNotFoundError, ValueError):
            with self.subTest(error_type=error_type):
                with mock.patch.object(
                    baseline_utils, "resolve_split_path", side_effect=error_type("invalid split")
                ) as resolve:
                    with mock.patch.object(baseline_utils, "resolve_default_eval_path") as legacy:
                        with self.assertRaisesRegex(error_type, "invalid split"):
                            baseline_utils.resolved_eval_path(args)
                        resolve.assert_called_once_with(baseline_utils.BASELINE_SOURCE, "test")
                        legacy.assert_not_called()

    def test_ovr_namespace_without_eval_path_uses_fixed_test_split(self):
        with mock.patch.object(sys, "argv", ["run_tfidf_ovr_baseline.py", "--model_type", "logistic"]):
            args = run_tfidf_ovr_baseline.parse_args()
        self.assertFalse(hasattr(args, "eval_path"))
        self.assertEqual(baseline_utils.resolved_eval_path(args), baseline_utils.DEFAULT_EVAL_PATH)


if __name__ == "__main__":
    unittest.main()
