import json

# クラスター情報を category_cluster.json から読み込む
with open('kmean_5_category_cluster.json', 'r') as f:
    cluster_mapping = json.load(f)

# api_new_data.json のデータを読み込む
with open('softfill_mashup_data_new_api_to_category.json', 'r') as f:
    mashupapi_data = json.load(f)


# すべてのAPIデータに対して処理を行う
for api_name, api_info in mashupapi_data.items():
    # 各APIの配列1のすべての要素をクラスター番号に置き換える
    new_elements = []
    for element in api_info[1]:  # api_info[1]はリストなので、そのすべての要素に対して処理を行う
        if element in cluster_mapping:
            # クラスター番号に置き換え
            new_elements.append(cluster_mapping[element])
        else:
            # クラスターに該当しない要素はそのまま残す場合
            new_elements.append(element)
    
    # 置き換え後のリストで更新
    api_info[1] = new_elements



# 処理後のデータを新しいファイルに保存する
with open('softfill_mashup_data_new_api_to_4category.json', 'w') as f:
    json.dump(mashupapi_data, f, indent=4)

print("すべてのAPIデータの変更が完了しました。jsonに保存されました。")
