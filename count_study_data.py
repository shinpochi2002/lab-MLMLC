import json
import os

# ベースディレクトリとファイル名の共通部分
base_path = "/home/nakagawa/studydata/layer3/big4_mid4/"
base_filename = "softfill_mashup_data_new_api_category_inclu_nomash_study_data_2layer_51_{}_study_data.json"

# いくつのファイルを試すか（ここでは例として0〜9）
max_index = 51

# 各ファイルの件数を表示
for i in range(max_index + 1):
    file_path = os.path.join(base_path, base_filename.format(i))
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"{os.path.basename(file_path)}: {len(data)} 件")
    else:
        print(f"{os.path.basename(file_path)}: ファイルが存在しません")
