import json

# JSONファイルを読み込む
with open("/home/nakagawa/study_data_mashup_apis.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    
a = len(data)
print(a)