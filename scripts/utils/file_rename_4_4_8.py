import os
import re
from pathlib import Path

# ===== 設定 =====
TARGET_DIR = Path("/home/nakagawa/studydata/layer3/big4_mid4/")

DRY_RUN = False  # 実行前に必ず True で挙動確認 → 問題なければ False に

# 新ファイル名の共通プレフィックス（★必要に応じて調整）
BASE_PREFIX = "softfill_mashup_data_new_api_category_inclu_nomash"
LAYER_TAG = "3layer"
STRUCT = "4-4-8"          # ★構造名（4-4-3 で良ければこのまま）
BIG_IDX_CONST = 4         # ★先頭に付ける layer1 インデックス（4グループ想定）


def build_new_name(idx1=None, idx2=None, idx3=None, suffix=None):
    """
    例：
      build_new_name()                 -> ..._3layer_4-4-3_4.json
      build_new_name(0)               -> ..._3layer_4-4-3_4_0.json
      build_new_name(0, 1, 2)         -> ..._3layer_4-4-3_4_0_1_2.json
      build_new_name(0, 1, 2, "test_data")
                                      -> ..._3layer_4-4-3_4_0_1_2_test_data.json
    """
    parts = [BASE_PREFIX, "study_data", LAYER_TAG, STRUCT, str(BIG_IDX_CONST)]

    if idx1 is not None:
        parts.append(str(idx1))
    if idx2 is not None:
        parts.append(str(idx2))
    if idx3 is not None:
        parts.append(str(idx3))

    if suffix:
        parts.append(suffix)

    return "_".join(parts) + ".json"


# ===== 元ファイル名のパターン定義 =====
# 4category_x_y-z_new(.json)
re_base_3idx = re.compile(
    r"^softfill_mashup_data_new_api_category_4category_"
    r"(?P<i1>\d+)_(?P<i2>\d+)-(?P<i3>\d+)_new\.json$"
)

# 4category_x_y-z_new_(study_data|test_data).json
re_split_3idx = re.compile(
    r"^softfill_mashup_data_new_api_category_4category_"
    r"(?P<i1>\d+)_(?P<i2>\d+)-(?P<i3>\d+)_new_"
    r"(?P<suffix>study_data|test_data)\.json$"
)

# 4category_x_new.json
re_base_1idx = re.compile(
    r"^softfill_mashup_data_new_api_category_4category_"
    r"(?P<i1>\d+)_new\.json$"
)

# 4category_x_new_(study_data|test_data).json
re_split_1idx = re.compile(
    r"^softfill_mashup_data_new_api_category_4category_"
    r"(?P<i1>\d+)_new_(?P<suffix>study_data|test_data)\.json$"
)

# 4category_new(.json) （全体まとめ）
re_base_all = re.compile(
    r"^softfill_mashup_data_new_api_category_4category_new\.json$"
)

# 4category_new_(study_data|test_data).json
re_split_all = re.compile(
    r"^softfill_mashup_data_new_api_category_4category_new_"
    r"(?P<suffix>study_data|test_data)\.json$"
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
        new_name = None

        # --- 3インデックス base ---
        m = re_base_3idx.match(name)
        if m:
            i1 = int(m.group("i1"))
            i2 = int(m.group("i2"))
            i3 = int(m.group("i3"))
            new_name = build_new_name(i1, i2, i3, suffix=None)

        # --- 3インデックス split ---
        if new_name is None:
            m = re_split_3idx.match(name)
            if m:
                i1 = int(m.group("i1"))
                i2 = int(m.group("i2"))
                i3 = int(m.group("i3"))
                suffix = m.group("suffix")
                new_name = build_new_name(i1, i2, i3, suffix=suffix)

        # --- 1インデックス base ---
        if new_name is None:
            m = re_base_1idx.match(name)
            if m:
                i1 = int(m.group("i1"))
                new_name = build_new_name(i1, suffix=None)

        # --- 1インデックス split ---
        if new_name is None:
            m = re_split_1idx.match(name)
            if m:
                i1 = int(m.group("i1"))
                suffix = m.group("suffix")
                new_name = build_new_name(i1, suffix=suffix)

        # --- 全体ファイル base ---
        if new_name is None:
            if re_base_all.match(name):
                new_name = build_new_name()

        # --- 全体ファイル split ---
        if new_name is None:
            m = re_split_all.match(name)
            if m:
                suffix = m.group("suffix")
                new_name = build_new_name(suffix=suffix)

        # 対象外はスキップ
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
            else:
                print(f"[RENAME] {name} -> {new_name}")
                os.rename(src, dst)


if __name__ == "__main__":
    main()
