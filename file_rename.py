import os
import re
from pathlib import Path

# ===== 設定 =====
# リネームしたいディレクトリを指定
TARGET_DIR = Path("/home/nakagawa/studydata/layer3/big12_mid4/")  # ★ここだけ自分のディレクトリに変えて

DRY_RUN = True  # 実際にリネームする時は False にする

# 新ファイル名の共通プレフィックス
BASE_PREFIX = "softfill_mashup_data_new_api_category_inclu_nomash"
LAYER_TAG = "3layer"
STRUCT = "12-4-3"
BIG_INDEX = 12

# ===== 新しいファイル名を組み立てる関数 =====
def build_new_name(idx1: int, idx2: int | None = None, split_suffix: str | None = None) -> str:
    """
    idx1, idx2, split_suffix から新しいファイル名を組み立てる。
    例:
      idx1=0, idx2=None, split_suffix=None
        -> softfill_..._study_data_3layer_12-4-3_0.json
      idx1=0, idx2=1, split_suffix="test_data"
        -> softfill_..._study_data_3layer_12-4-3_0_1_test_data.json
    """
    parts = [
        BASE_PREFIX,
        "study_data",
        LAYER_TAG,
        STRUCT,
        str(BIG_INDEX),
        str(idx1),
    ]

    if idx2 is not None:
        parts.append(str(idx2))

    if split_suffix:
        parts.append(split_suffix)

    return "_".join(parts) + ".json"


# ===== 元ファイル名のパターン =====
# ベース（split 前）
# 例: softfill_..._big12_middle4_0_study_data.json
#     softfill_..._big12_middle4_0-1_study_data.json
re_base = re.compile(
    r"^softfill_mashup_data_new_api_category_inclu_nomash_big12_middle4_"
    r"(?P<idx1>\d+)(?:-(?P<idx2>\d+))?_study_data\.json$"
)

# split 後
# 例: softfill_..._big12_middle4_0_study_test_data.json
#     softfill_..._big12_middle4_0-1_study_study_data.json
re_split = re.compile(
    r"^softfill_mashup_data_new_api_category_inclu_nomash_big12_middle4_"
    r"(?P<idx1>\d+)(?:-(?P<idx2>\d+))?_study_(?P<suffix>test_data|study_data)\.json$"
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

        # 1. ベース（split 前）のパターンにマッチするか
        m_base = re_base.match(name)
        if m_base:
            idx1 = int(m_base.group("idx1"))
            idx2_str = m_base.group("idx2")
            idx2 = int(idx2_str) if idx2_str is not None else None

            new_name = build_new_name(idx1, idx2, split_suffix=None)
        else:
            # 2. split 後のパターンにマッチするか
            m_split = re_split.match(name)
            if not m_split:
                # 対象外のファイルはスキップ
                continue

            idx1 = int(m_split.group("idx1"))
            idx2_str = m_split.group("idx2")
            idx2 = int(idx2_str) if idx2_str is not None else None
            suffix = m_split.group("suffix")  # "test_data" or "study_data"

            new_name = build_new_name(idx1, idx2, split_suffix=suffix)

        # すでに目的の名前なら何もしない
        if name == new_name:
            continue

        src = entry
        dst = TARGET_DIR / new_name

        if DRY_RUN:
            print(f"[DRY-RUN] {name} -> {new_name}")
        else:
            # 実リネーム
            if dst.exists():
                print(f"[WARN] Skip (dest exists): {dst}")
            else:
                print(f"[RENAME] {name} -> {new_name}")
                os.rename(src, dst)


if __name__ == "__main__":
    main()
