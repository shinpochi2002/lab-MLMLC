import json

# クラスター情報を category_cluster.json から読み込む
with open('kmean_4_category_cluster.json', 'r') as f:
    cluster_mapping = json.load(f)

# mashupapi_data と api_data のデータを読み込む
with open('softfill_mashup_data_new_api_to_category.json', 'r') as f:
    mashupapi_data = json.load(f)

with open('api_data_new.json', 'r') as f:
    api_data = json.load(f)

# 判別結果と説明文のみを含むデータを作成する関数
def create_cluster_and_description_data(data, element_index, dis_index, single_element=False):
    result_data = {}
    for api_name, api_info in data.items():
        # 説明文（ここでは api_info[0] を説明文と仮定）
        description = api_info[dis_index]
        
        # クラスター番号に置き換えたリスト
        new_elements = []
        target_elements = [api_info[element_index][0]] if single_element else api_info[element_index]
        
        for element in target_elements:
            if element in cluster_mapping:
                new_elements.append(cluster_mapping[element])  # クラスター番号に置き換え
        
        # 該当しない要素はスキップ
        if new_elements:
            # 判別したクラスター番号と説明文を新しいデータに格納
            result_data[api_name] = [
                new_elements,
                description
            ]
    return result_data

def create_description_data(data, element_index, dis_index, single_element=False):
    result_data = {}
    for api_name, api_info in data.items():
        # 説明文（ここでは api_info[dis_index] を説明文と仮定）
        description = api_info[dis_index]

        
        # クラスター番号に置き換えたリスト
        new_elements = []
        target_elements = [api_info[element_index][0]] if single_element else api_info[element_index]
        
        for element in target_elements:
            if element in cluster_mapping:  # クラスター番号に置き換えられるもののみ対象
                new_elements.append(cluster_mapping[element])
        
        # 置き換え済みの要素が存在する場合のみ格納
        if new_elements:
            result_data[api_name] = [
                target_elements,  # 置き換え済みの要素
                description    # 説明文
            ]
    return result_data


# mashupapi_data のクラスター番号と説明文だけを抽出
processed_mashupapi_data = create_cluster_and_description_data(mashupapi_data, 1, 0)

# api_data の配列0の1つ目の要素のみクラスター番号に変換し、説明文と一緒に保持
processed_api_data = create_cluster_and_description_data(api_data, 0, 1, single_element=True)

# # mashupapi_data のクラスター番号と説明文だけを抽出
# more_mashupapi_data = create_description_data(mashupapi_data, 1, 0)

# # api_data の配列0の1つ目の要素のみクラスター番号に変換し、説明文と一緒に保持
# more_api_data = create_description_data(api_data, 0, 1, single_element=True)



combined_data={}

nomashapi = {}

nocludata={}

# 両方のデータをまとめてIDを付与
id_counter = 1  # IDカウンターを初期化
nomash_cou = 1
noclucou = 1

for api_name, data in {**processed_mashupapi_data, **processed_api_data}.items():
    combined_data[api_name] = [
        data[0],      # クラスター番号
        data[1],
        id_counter    # 説明文
    ]
    id_counter += 1  # 次のエントリに向けてIDをインクリメント

# for api_name, data in {**processed_api_data}.items():
#     nomashapi[api_name] = [
#         data[0],      # クラスター番号
#         data[1],
#         nomash_cou    # 説明文
#     ]
#     nomash_cou += 1  # 次のエントリに向けてIDをインクリメント

# for api_name, data in {**more_mashupapi_data, **more_api_data}.items():
#     combined_data[api_name] = [
#         data[0],      # クラスター番号
#         data[1],
#         id_counter    # 説明文
#     ]
#     id_counter += 1  # 次のエントリに向けてIDをインクリメント


with open('softfill_mashup_data_new_api_to_4category_inclu_nomash.json', 'w') as f:
    json.dump(combined_data, f, indent=4)

# with open('new_api_data_to_4category.json', 'w') as f:
#     json.dump(nomashapi, f, indent=4)

# with open('softfill_mashup_data_new_api_category_inclu_nomash.json', 'w') as f:
#     json.dump(combined_data, f, indent=4)

print("すべてのAPIデータの変更が完了し、jsonファイルに保存されました。")
