import json
import random
import os

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# 処理する a, b, c の最大値（12_2_2_3）
max_a = 11
max_b = 1
max_c = 1

# テストデータの比率
test_ratio = 0.10

base_path = legacy_path('/home/nakagawa/lab-MLMLC/studydata/layer4/layer4_12_2_2_3/')

for a in range(max_a + 1):
    for b in range(max_b + 1):
        for c in range(max_c + 1):
            input_filename = f'softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_12_2_2_3_12_{a}_{b}_{c}_study_data.json'
            input_path = os.path.join(base_path, input_filename)

            test_output_filename = f'softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_12_2_2_3_12_{a}_{b}_{c}_test_data.json'
            train_output_filename = f'softfill_mashup_data_new_api_category_inclu_nomash_study_data_4layer_12_2_2_3_12_{a}_{b}_{c}_study_data.json'

            test_output_path = os.path.join(base_path, test_output_filename)
            train_output_path = os.path.join(base_path, train_output_filename)

            if not os.path.exists(input_path):
                print(f"[{a}_{b}] 入力ファイルが見つかりません: {input_path}")
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

            print(f"[{a}-{b}] {test_size}件のテストデータと{len(train_data)}件の学習データを出力完了")
