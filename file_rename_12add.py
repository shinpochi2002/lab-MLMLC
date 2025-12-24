import os
import re
from pathlib import Path

# ===== 設定 =====
TARGET_DIR = Path("/home/nakagawa/studydata/layer3/big12_mid4/")  # 必要なら変更
DRY_RUN = False  # 最初は True で確認。OKなら False にして実行

BASE_PREFIX = "softfill_mashup_data_new_api_category_inclu_nomash"
LAYER_TAG = "3layer"
STRUCT = "12-4-3"
BIG_INDEX = 12  # 先頭に固定で入れるインデックス

# いまのファイル名パターン:
#   softfill_..._study_data_3layer_12-4-3_11_2_test_data.json
#   softfill_..._study_data_3layer_12-4-3_11_2_study_data.json
#   softfill_..._study_data_3layer_12-4-3_11_2.json
#   softfill_..._study_data_3layer_12-4-3_11.json
#
# を対象にする。
pattern = re.compile(
    rf"^{re.escape(BASE_PREFIX)}_study_data_{LAYER_TAG}_{STRUCT}_"
    r"(?P<idx1>\d+)"                       # 11 のところ
    r"(?:_(?P<idx2>\d+))?"                 # 2 のところ（あってもなくてもOK）
    r"(?:_(?P<suffix>study_data|test_data))?"  # suffix（study_data/test_data）があれば
    r"\.json$"
)

def build_new_name(idx1: int, idx2: int | None, suffix: str | None) -> str:
    """
    いまの idx1, idx2, suffix から
    softfill_..._study_data_3layer_12-4-3_12_idx1_idx2_suffix.json
    を作る。
    """
    parts = [
        BASE_PREFIX,
        "study_data",
        LAYER_TAG,
        STRUCT,
        str(BIG_INDEX),  # ここで 12 を先頭に固定で入れる
        str(idx1),
    ]

    if idx2 is not None:
        parts.append(str(idx2))

    if suffix is not None:
        parts.append(suffix)

    return "_".join(parts) + ".json"


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

        m = pattern.match(name)
        if not m:
            # 例えば全体用の
            #   softfill_..._study_data_3layer_12-4-3.json
            # はマッチしないのでそのままスキップ
            continue

        idx1 = int(m.group("idx1"))
        idx2 = m.group("idx2")
        idx2 = int(idx2) if idx2 is not None else None
        suffix = m.group("suffix")  # "study_data" / "test_data" / None

        new_name = build_new_name(idx1, idx2, suffix)

        # すでに目的の名前ならスキップ
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
