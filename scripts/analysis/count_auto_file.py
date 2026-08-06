import json
import os
import glob

# ベースパス
base_dir = "/home/nakagawa/studydata/layer3/layer3_16a4a3/"

# ファイルパターン（*はワイルドカード）
pattern = os.path.join(base_dir, "softfill_mashup_data_new_api_category_inclu_nomash_study_data_3layer_16a4a3_*-*_study_data.json")

# パターンに一致するファイルを全部取得
file_list = sorted(glob.glob(pattern))

# 件数表示
for file_path in file_list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"{os.path.basename(file_path)}: {len(data)} 件")
    except Exception as e:
        print(f"{os.path.basename(file_path)}: 読み込み失敗 ({e})")
