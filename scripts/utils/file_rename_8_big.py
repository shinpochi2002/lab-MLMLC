#!/usr/bin/env python
import os
import re

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# 対象ディレクトリ
TARGET_DIR = legacy_path("/home/nakagawa/studydata/layer3/layer3_8_4_4")

# True の間は DRY-RUN（表示だけで実際にはリネームしない）
DRY_RUN = False

def main():
    for fname in os.listdir(TARGET_DIR):
        old_path = os.path.join(TARGET_DIR, fname)

        # ---------- パターン1: all_inc_8big_Xmid(_study_data/_test_data).json ----------
        # 例:
        #   softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_0mid.json
        #   softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_0mid_study_data.json
        #   softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_0mid_test_data.json
        m_all_mid = re.match(
            r"^softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_(\d)mid(_study_data|_test_data)?\.json$",
            fname
        )
        if m_all_mid:
            mid_idx = m_all_mid.group(1)        # 0〜7
            suffix = m_all_mid.group(2) or ""   # '', '_study_data', '_test_data'

            # 新しいファイル名:
            # softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_8-4-4_8_<mid_idx><suffix>.json
            new_name = (
                f"softfill_mashup_data_new_api_category_inclu_nomash_study_data_"
                f"3layer_8-4-4_8_{mid_idx}{suffix}.json"
            )
            new_path = os.path.join(TARGET_DIR, new_name)

            if DRY_RUN:
                print(f"[DRY-RUN] {fname} -> {new_name}")
            else:
                os.rename(old_path, new_path)
            continue

        # ---------- パターン2: all_inc_8big(_study_data/_test_data).json（mid なし） ----------
        # 例:
        #   softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_study_data.json
        #   softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_test_data.json
        #   （あれば）softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big.json
        m_all_big = re.match(
            r"^softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big(_study_data|_test_data)?\.json$",
            fname
        )
        if m_all_big:
            suffix = m_all_big.group(1) or ""   # '', '_study_data', '_test_data'

            # 新しいファイル名:
            # softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_8-4-4_8<suffix>.json
            new_name = (
                "softfill_mashup_data_new_api_category_inclu_nomash_study_data_"
                f"3layer_8-4-4_8{suffix}.json"
            )
            new_path = os.path.join(TARGET_DIR, new_name)

            if DRY_RUN:
                print(f"[DRY-RUN] {fname} -> {new_name}")
            else:
                os.rename(old_path, new_path)
            continue

        # 上のどちらにもマッチしないファイルは何もしない
        # print(f"[SKIP] {fname}")

if __name__ == "__main__":
    main()
