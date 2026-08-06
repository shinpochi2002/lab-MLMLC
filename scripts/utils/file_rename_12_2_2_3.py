import os
from pathlib import Path

# 対象ディレクトリ
BASE_DIR = Path("/home/nakagawa/studydata/layer4/layer4_12_2_2_3")

# 変更前・変更後の共通プレフィックス
OLD_CORE = "softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_12a2a2a3"
NEW_CORE = "softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_12-2-2-3_12"

DRY_RUN = False  # ← 最初は True で確認、本番で False にする

def main():
    for path in BASE_DIR.iterdir():
        if not path.is_file():
            continue

        name = path.name
        if not name.startswith(OLD_CORE):
            continue

        new_name = name.replace(OLD_CORE, NEW_CORE, 1)
        new_path = path.with_name(new_name)

        if DRY_RUN:
            print(f"[DRY-RUN] {name} -> {new_name}")
        else:
            print(f"RENAMING {name} -> {new_name}")
            os.rename(path, new_path)

if __name__ == "__main__":
    main()
