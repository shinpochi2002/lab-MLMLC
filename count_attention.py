import json

# JSONファイルを読み込む
with open('mashup_data_new_updated.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# APIに「atention」が含まれているエントリーをカウントする関数
def count_atention_apis(data):
    count = 0
    apilist=[]
    for key,entry in data.items():
        apis = entry[1]  
        if "Attention!!!!" in apis or len(apis)==1:
            if key not in apilist:  # APIリストに「Attention!!!!」が含まれているか確認
                count += 1  # 含まれていればカウント
                apilist.append(key)
    return count,apilist

# カウントの実行
count,apilist = count_atention_apis(data)
print("Number of entries with 'Attention!!!!' in any API:", count)
print(apilist)
print(len(apilist))


# apilistの内容をJSONファイルに保存
with open('hardapilist.json', 'w', encoding='utf-8') as f:
    json.dump(apilist, f, ensure_ascii=False, indent=4)