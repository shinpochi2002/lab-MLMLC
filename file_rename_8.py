import os
import re
from pathlib import Path

# ===== 設定 =====
TARGET_DIR = Path("/home/nakagawa/studydata/layer3/layer3_8_4_4")

DRY_RUN = False  # 実際にリネームするときは False に

BASE_PREFIX = "softfill_mashup_data_new_api_category_inclu_nomash"
LAYER_TAG = "3layer"
STRUCT = "8-4-4"
L1_CONST = 8  # 12-4-3 のときの 12 / 4-4-8 のときの 4 に相当


def build_new_name(mid=None, small=None, suffix=None):
    """
    例:
      build_new_name(0, 0)                   -> ..._3layer_8-4-4_8_0_0.json
      build_new_name(0, 0, "study_data")     -> ..._3layer_8-4-4_8_0_0_study_data.json
      build_new_name(3, 2, "test_data")      -> ..._3layer_8-4-4_8_3_2_test_data.json
      build_new_name(1, 0, "split_new")      -> ..._3layer_8-4-4_8_1_0_split_new.json
    """
    parts = [BASE_PREFIX, "study_data", LAYER_TAG, STRUCT, str(L1_CONST)]

    if mid is not None:
        parts.append(str(mid))
    if small is not None:
        parts.append(str(small))

    if suffix:
        parts.append(suffix)

    return "_".join(parts) + ".json"


# ===== 元ファイル名のパターン =====
# 8category_midX_Y-Z_new.json
re_base_3idx = re.compile(
    r"^softfill_mashup_data_new_api_category_8category_"
    r"mid(?P<mid>\d+)_(?P<i2>\d+)-(?P<small>\d+)_new\.json$"
)

# 8category_midX_Y-Z_new_(study_data|test_data).json
re_split_3idx = re.compile(
    r"^softfill_mashup_data_new_api_category_8category_"
    r"mid(?P<mid>\d+)_(?P<i2>\d+)-(?P<small>\d+)_new_"
    r"(?P<suffix>study_data|test_data)\.json$"
)

# 8category_midX_Y-Z_split_new.json
re_split_only = re.compile(
    r"^softfill_mashup_data_new_api_category_8category_"
    r"mid(?P<mid>\d+)_(?P<i2>\d+)-(?P<small>\d+)_split_new\.json$"
)


def main():
    if not TARGET_DIR.exists():
        print(f"[ERROR] TARGET_DIR not found: {TARGET_DIR}")
        return

    for entry in TARGET_DIR.iterdir():
        if not entry.is_file():
            continue
        if not entry.name.endswith(".json"):
            continue

        name = entry.name

        # すでに inclu_nomash から始まるファイルはスキップ（完成形とみなす）
        if name.startswith("softfill_mashup_data_new_api_category_inclu_nomash"):
            continue

        new_name = None

        # --- base: 8category_midX_Y-Z_new.json ---
        m = re_base_3idx.match(name)
        if m:
            mid = int(m.group("mid"))
            i2 = int(m.group("i2"))
            small = int(m.group("small"))

            # mid と i2 が違っていたら一応警告だけ出す（リネームは mid 優先）
            if mid != i2:
                print(f"[WARN] mid({mid}) != i2({i2}) in {name}")

            new_name = build_new_name(mid, small, suffix=None)

        # --- split: 8category_midX_Y-Z_new_(study_data|test_data).json ---
        if new_name is None:
            m = re_split_3idx.match(name)
            if m:
                mid = int(m.group("mid"))
                i2 = int(m.group("i2"))
                small = int(m.group("small"))
                suffix = m.group("suffix")

                if mid != i2:
                    print(f"[WARN] mid({mid}) != i2({i2}) in {name}")

                new_name = build_new_name(mid, small, suffix=suffix)

        # --- split_only: 8category_midX_Y-Z_split_new.json ---
        if new_name is None:
            m = re_split_only.match(name)
            if m:
                mid = int(m.group("mid"))
                i2 = int(m.group("i2"))
                small = int(m.group("small"))
                if mid != i2:
                    print(f"[WARN] mid({mid}) != i2({i2}) in {name}")

                # suffix を "split_new" として残す
                new_name = build_new_name(mid, small, suffix="split_new")

        # マッチしなかったものは触らない
        if new_name is None:
            continue

        if name == new_name:
            continue

        src = entry
        dst = TARGET_DIR / new_name

        if DRY_RUN:
            print(f"[DRY-RUN] {name} -> {new_name}")
        else:
            if dst.exists():
                print(f"[WARN] Skip (dest exists): {dst}")
                continue
            print(f"[RENAME] {name} -> {new_name}")
            os.rename(src, dst)


if __name__ == "__main__":
    main()
