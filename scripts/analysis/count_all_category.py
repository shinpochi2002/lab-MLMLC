import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

with open(legacy_path('/home/nakagawa/new_api_data/api_data_new.json'), 'r', encoding='utf-8') as file:
    apidata = json.load(file)

with open(legacy_path('/home/nakagawa/mashup_data/mashup_data_new.json'), 'r', encoding='utf-8') as file:
    mashdata = json.load(file)

apicategory = {}
mashcategory = {}
allcategory = {}

# api_dataの処理
for api_name, api_info in apidata.items():
    apicategories = api_info[0]  # カテゴリ情報を取得
    for category in apicategories:
        if category not in allcategory:
            allcategory[category] = 1  
        else:
            allcategory[category] += 1

for api_name, api_info in apidata.items():
    apicategories = api_info[0]  # カテゴリ情報を取得
    for category in apicategories:
        if category not in apicategory:
            apicategory[category] = 1  
        else:
            apicategory[category] += 1

for mash_name, mash_info in mashdata.items():
    mashcategories = mash_info[0]  # カテゴリ情報を取得
    for category in mashcategories:
        if category not in allcategory:
            allcategory[category] = 1
        else:
            allcategory[category] += 1


for mash_name, mash_info in mashdata.items():
    mashcategories = mash_info[0]  # カテゴリ情報を取得
    for category in mashcategories:
        if category not in mashcategory:
            mashcategory[category] = 1
        else:
            mashcategory[category] += 1

#print(allcategory)
print("API",len(apicategory))
print("Mash",len(mashcategory))
print("ALL",len(allcategory))

sorted_categories = sorted(allcategory.items(), key=lambda x: x[1], reverse=True)

# ソート結果を表示（例）
for category, count in sorted_categories:
    print(f"{category}: {count}")