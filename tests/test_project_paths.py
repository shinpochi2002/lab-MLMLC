"""Regression checks for hierarchy checkpoint directory selection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common.project_paths import hierarchy_checkpoint_dir, legacy_path


class HierarchyCheckpointPathTests(unittest.TestCase):
    def test_failed_validation_and_test_branches_keep_selected_configuration(self):
        for branch in ("4_3_3_0", "4_0_0_3"):
            name = f"mashup_softfill_api_all_layer3_{branch}_model_weights.pth"
            expected = hierarchy_checkpoint_dir(3, "layer3_4_4_8") / name
            for prefix in ("/home/nakagawa/lab-MLMLC/", "/home/nakagawa/", ""):
                with self.subTest(branch=branch, prefix=prefix):
                    self.assertEqual(
                        Path(legacy_path(f"{prefix}layer3_4_4_8/{name}")), expected
                    )

    def test_same_branch_filename_stays_in_each_selected_configuration(self):
        name = "mashup_softfill_api_all_layer3_12_0_0_1_model_weights.pth"
        for pattern in ("layer3_12_2_6", "layer3_12_4_3", "layer3_12_6_2"):
            with self.subTest(pattern=pattern):
                self.assertEqual(
                    Path(legacy_path(f"/home/nakagawa/lab-MLMLC/{pattern}/{name}")),
                    hierarchy_checkpoint_dir(3, pattern) / name,
                )

    def test_four_layer_checkpoints_keep_configuration_at_every_depth(self):
        for branch in ("8", "8_0", "8_0_1", "8_0_1_2"):
            name = f"mashup_softfill_api_all_layer4_{branch}_model_weights.pth"
            with self.subTest(branch=branch):
                self.assertEqual(
                    Path(legacy_path(f"layer4_8_3_2_3/{name}")),
                    hierarchy_checkpoint_dir(4, "layer4_8_3_2_3") / name,
                )

    def test_converted_paths_remain_unchanged_when_resolved_again(self):
        for layer, pattern, branch in (
            (3, "layer3_4_4_8", "4_3_3_0"),
            (4, "layer4_8_3_2_3", "8_0_1_2"),
        ):
            name = f"mashup_softfill_api_all_layer{layer}_{branch}_model_weights.pth"
            expected = hierarchy_checkpoint_dir(layer, pattern) / name
            with self.subTest(pattern=pattern):
                self.assertEqual(Path(legacy_path(expected)), expected)

    def test_legacy_filenames_without_configuration_keep_existing_mapping(self):
        cases = (
            ("mashup_softfill_api_all_layer1_model_weights.pth", 1, None,
             "mashup_softfill_api_all_layer1_model_weights.pth"),
            ("mashup_softfill_api_all_layer2_4_3_model_weights.pth", 2, "layer2_4",
             "mashup_softfill_api_all_layer2_4_3_model_weights.pth"),
            ("mashup_softfill_api_all_layer3_12_4_3_model_weights.pth", 3, "layer3_12_4_3",
             "mashup_softfill_api_all_layer3_12_4_3_model_weights.pth"),
            ("mashup_softfill_api_4category_small3-0clusters_new_model_weights_test.pth",
             3, "layer3_4_4_8", "mashup_softfill_api_all_layer3_4_3_3_0_model_weights.pth"),
        )
        for old_name, layer, pattern, new_name in cases:
            with self.subTest(name=old_name):
                self.assertEqual(
                    Path(legacy_path(f"/home/nakagawa/{old_name}")),
                    hierarchy_checkpoint_dir(layer, pattern) / new_name,
                )


if __name__ == "__main__":
    unittest.main()
