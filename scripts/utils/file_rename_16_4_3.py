import os
import re

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

DRY_RUN = False  # 実際にリネームするときは False にする

BASE_DIR = legacy_path("/home/nakagawa/studydata/layer3/layer3_16a4a3")

OLD_PREFIX = "softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_16a4a3"
NEW_PREFIX = "softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_16-4-3_16"

os.chdir(BASE_DIR)

for fname in os.listdir("."):
    if not fname.startswith(OLD_PREFIX) or not fname.endswith(".json"):
        continue

    # 例:
    # ""                     （ルート）
    # "_study_data"
    # "_test_data"
    # "_0"
    # "_0_study_data"
    # "_0-0"
    # "_0-0_test_data"
    tail = fname[len(OLD_PREFIX):-len(".json")]

    m = re.match(r'(_\d+(?:-\d+)*)((?:_study_data|_test_data)?)$', tail)
    if not m:
        # ルートの study/test など: "_study_data" / "_test_data" / "" を扱う
        if tail in ("", "_study_data", "_test_data"):
            idx_part = ""   # インデックスなし
            suffix = tail   # そのまま suffix として扱う
        else:
            print(f"[SKIP] {fname} (パターンにマッチしない)")
            continue
    else:
        idx_part, suffix = m.groups()
        # 例: idx_part = "_0-0" → "_0_0" に
        idx_part = idx_part.replace("-", "_")

    new_tail = idx_part + suffix
    new_name = NEW_PREFIX + new_tail + ".json"

    if fname == new_name:
        continue

    if DRY_RUN:
        print(f"[DRY-RUN] {fname} -> {new_name}")
    else:
        print(f"[RENAME]  {fname} -> {new_name}")
        os.rename(fname, new_name)
