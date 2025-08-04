import json

# JSONファイルを読み込む
with open("/home/nakagawa/api_category_list/softapilist.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    
a = len(data)
print(a)