#!/usr/bin/env python3
import os
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

TARGET_DIR = Path(legacy_path("/home/nakagawa/studydata/layer3/layer3_12_6_2"))
DRY_RUN = False

OLD_STRUCT = "3layer_12a6a2"
NEW_STRUCT = "3layer_12-6-2"
HEAD_INDEX = "12"


def main():
    for filename in os.listdir(TARGET_DIR):
        old_path = TARGET_DIR / filename
        if not old_path.is_file():
            continue

        # まず 12a6a2 → 12-6-2
        new_name = filename.replace(OLD_STRUCT, NEW_STRUCT)

        if new_name == filename:
            continue

        # パターン1: 末尾が "3layer_12-6-2.json" のもの（ベースファイル）
        if new_name.endswith(NEW_STRUCT + ".json"):
            new_name = new_name.replace(
                NEW_STRUCT + ".json",
                f"{NEW_STRUCT}_{HEAD_INDEX}.json",
                1
            )
        else:
            # パターン2: "3layer_12-6-2_..." と続く普通のやつ
            pattern = NEW_STRUCT + "_"
            replacement = NEW_STRUCT + f"_{HEAD_INDEX}_"
            new_name = new_name.replace(pattern, replacement, 1)

        new_path = TARGET_DIR / new_name

        print(f"[DRY-RUN] {filename} → {new_name}")
        if not DRY_RUN:
            os.rename(old_path, new_path)


if __name__ == "__main__":
    main()
