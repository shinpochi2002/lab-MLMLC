import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# 読み込むJSONファイルの名前
input_file = legacy_path('/home/nakagawa/softfill_mashup_data_new_api_category_8category_7_3_study_data.json')
output_file = legacy_path('/home/nakagawa/api_category_list/softfill_inc_8category_categorylist_category7_3.json')

# JSONファイルを読み込む
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 使用されているAPIのリストを作成
api_list = set()
for value in data:
    apis = value["categories"]  
    api_list.update(apis)  # リストに追加

# 重複するAPIを除外
api_list = list(api_list)

# APIリストをJSON形式で保存
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(api_list, file, ensure_ascii=False, indent=4)

print(f"APIリストが{output_file}に保存されました。")
