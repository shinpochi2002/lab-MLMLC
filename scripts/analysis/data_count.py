import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# 読み込むJSONファイルの名前
input_file = legacy_path('/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_study_data.json')

# JSONファイルを読み込む
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# # 使用されているAPIのリストを作成
# api_list = []
# for key, value in data.items():
#     apis = value[2]  
#     api_list.append(apis)  # リストに追加

# print(len(api_list))
print(len(data))