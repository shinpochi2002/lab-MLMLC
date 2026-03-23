import json
import random
import os

# 分割対象のファイル数（0〜32）
max_index = 12

# 分割比率（20%がテストデータ）
test_ratio = 0.20

base_path = '/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_8_3_2_3/'

for i in range(max_index + 1):
    input_path = os.path.join(base_path, f'softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_8_3_2_3_{i}.json')
    test_output_path = os.path.join(base_path, f'softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_8_3_2_3_{i}_test_data.json')
    train_output_path = os.path.join(base_path, f'softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_8_3_2_3_{i}_study_data.json')

    if not os.path.exists(input_path):
        print(f"[{i}] 入力ファイルが見つかりません: {input_path}")
        continue

    with open(input_path, 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    all_ids = [api_info["name"] for api_info in all_data]
    total_count = len(all_ids)
    test_size = int(total_count * test_ratio)

    random_ids = random.sample(all_ids, test_size)
    test_data = [api_info for api_info in all_data if api_info["name"] in random_ids]
    train_data = [api_info for api_info in all_data if api_info["name"] not in random_ids]

    with open(test_output_path, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)

    with open(train_output_path, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, indent=4, ensure_ascii=False)

    print(f"[{i}] {test_size}件のテストデータと{len(train_data)}件の学習データを出力完了")
