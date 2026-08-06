from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import API_CATEGORY_LIST_DIR, hierarchy_checkpoint_dir

def categorylist():
    for i in range(12):
        # 最上位カテゴリ (layer4)
        print(f"M{i}API_CATEGORY_LIST_PATH = str(API_CATEGORY_LIST_DIR / 'softfill_inc_categorylist_layer3_12a6a2_{i}.json')")

        # 中間カテゴリ (layer3)
        for j in range(6):
            print(f"M{i}_{j}API_CATEGORY_LIST_PATH = str(API_CATEGORY_LIST_DIR / 'softfill_inc_categorylist_layer3_12a6a2_{i}_{j}.json')")

            # 下位カテゴリ (layer2)
            for k in range(0):
                print(f"M{i}_{j}_{k}API_CATEGORY_LIST_PATH = str(API_CATEGORY_LIST_DIR / 'softfill_inc_categorylist_layer2_12a6_{i}_{j}_{k}.json')")
        
        print()  # 空行で見やすく
        
        
def categorylistopen():
    for i in range(12):
        print(f"with open(M{i}API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:")
        print(f"    mid{i}_all_categories_list = json.load(f)")

        for j in range(6):  # _0, _1
            print(f"with open(M{i}_{j}API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:")
            print(f"    mid{i}_{j}_all_categories_list = json.load(f)")
            for k in range(0):  # _0_0, _0_1 and _1_0, _1_1
                print(f"with open(M{i}_{j}_{k}API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:")
                print(f"    mid{i}_{j}_{k}_all_categories_list = json.load(f)")
        print()  # 空行で区切る


def modelpath():
    for i in range(12):
        # 最上位モデル（layer4）
        print(f"M{i}MODEL_WEIGHTS_PATH = str(hierarchy_checkpoint_dir(3, 'layer3_12_6_2') / 'mashup_softfill_api_all_layer3_12a6a2_{i}_model_weights.pth')")

        # 中間層モデル（layer3）
        for j in range(6):
            print(f"M{i}_{j}MODEL_WEIGHTS_PATH = str(hierarchy_checkpoint_dir(3, 'layer3_12_6_2') / 'mashup_softfill_api_all_layer3_12a6a2_{i}_{j}_model_weights.pth')")

            # 下位層モデル（layer2）
            for k in range(0):
                print(f"M{i}_{j}_{k}MODEL_WEIGHTS_PATH = str(hierarchy_checkpoint_dir(2, 'layer2_12') / 'mashup_softfill_api_all_layer2_12a6_{i}_{j}_{k}_model_weights.pth')")
        
        print()  # 空行でブロックを区切る


def categorytoindex():
    for i in range(12):
        print(f"m{i}_category_to_index = {{category: idx for idx, category in enumerate(mid{i}_all_categories_list)}}")
        print(f"m{i}_index_to_category = {{idx: category for category, idx in m{i}_category_to_index.items()}}")

        for j in range(6):  # _0, _1
            print(f"m{i}_{j}_category_to_index = {{category: idx for idx, category in enumerate(mid{i}_{j}_all_categories_list)}}")
            print(f"m{i}_{j}_index_to_category = {{idx: category for category, idx in m{i}_{j}_category_to_index.items()}}")
            
        print()  # 空行で区切り
        
def createmidcategory():
    result = [f"{i}-{j}" for i in range(12) for j in range(6)]
    print(result)
    
def numberoflabel():
    for i in range(12):
        print(f"m{i}_number_of_labels = len(mid{i}_all_categories_list)")
        
        for j in range(6):
            print(f"m{i}_{j}_number_of_labels = len(mid{i}_{j}_all_categories_list)")
            for k in range(0):
                print(f"m{i}_{j}_{k}_number_of_labels = len(mid{i}_{j}_{k}_all_categories_list)")
        print()  # 空行で区切り

def createprocess():
    indent = "    "

    for i in range(12):
        if i == 0:
            print(f"{indent}if nextrun[0] == {i}:")
        else:
            print(f"{indent}elif nextrun[0] == {i}:")

        for j in range(6):
            print(f"{indent*2}m{i}_{j}model = load_model(BertForMultilabelClassification, model_name, m{i}_{j}_number_of_labels, path=M{i}_{j}MODEL_WEIGHTS_PATH)")

        print(f"\n{indent*2}mid{i}results = run_model_and_get_results(")
        print(f"{indent*3}text=text,")
        print(f"{indent*3}model=m{i}model,")
        print(f"{indent*3}tokenizer=tokenizer,")
        print(f"{indent*3}index_to_category=m{i}_index_to_category,")
        print(f"{indent*2})\n")

        print(f"{indent*2}for nextrun in mid{i}results:")
        for j in range(6):
            if j == 0:
                print(f"{indent*3}if nextrun[0] == \"{i}-{j}\":")
            else:
                print(f"{indent*3}elif nextrun[0] == \"{i}-{j}\":")
            print(f"{indent*4}mid{i}_{j}results = run_model_and_get_results(")
            print(f"{indent*5}text=text,")
            print(f"{indent*5}model=m{i}_{j}model,")
            print(f"{indent*5}tokenizer=tokenizer,")
            print(f"{indent*5}index_to_category=m{i}_{j}_index_to_category,")
            print(f"{indent*4})")
            print(f"{indent*4}recommended_indices = [category[0] for category in mid{i}_{j}results]")
            print(f"{indent*4}allrecommended += recommended_indices")

        for j in range(6):
            print(f"{indent*2}del m{i}_{j}model")
        print(f"{indent*2}torch.cuda.empty_cache()\n")


createprocess()