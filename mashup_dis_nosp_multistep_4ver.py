import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score, average_precision_score,classification_report
import matplotlib.pyplot as plt
from matplotlib import cm

M0API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_4category_categorylist_category0_new.json'
M1API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_4category_categorylist_category1_new.json'
M2API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_4category_categorylist_category2_new.json'
M3API_CATEGORY_LIST_PATH = '/home/nakagawa/api_category_list/softfill_inc_4category_categorylist_category3_new.json'
EVALUATION_DATA_PATH = '/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_4big_test_data.json'
BMODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_4category_new_model_weights_test.pth"
M0MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_4category_0_new_model_weights_test.pth"
M1MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_4category_1_new_model_weights_test.pth"
M2MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_4category_2_new_model_weights_test.pth"
M3MODEL_WEIGHTS_PATH = "/home/nakagawa/mashup_softfill_api_4category_3_new_model_weights_test.pth"
DETAILED_PREDICTIONS_OUTPUT_PATH = "softfill_category_4bigcategory_allmiddle_new_detailed_predictions_test.json"
PRECISION_RECALL_CURVE_OUTPUT_PATH = "4bigcate_allmiddle_new_precision_recall_curve_with_colors_test.png"
F1_SCORE_CURVE_OUTPUT_PATH = "softfill_mashup_4bigcategory_allmiddle_new_f1_score_curve_test.png"


# Load class list from 'apilist.json' (API names list)
with open(M0API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid0_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M1API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid1_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M2API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid2_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

# Load class list from 'apilist.json' (API names list)
with open(M3API_CATEGORY_LIST_PATH, 'r', encoding='utf-8') as f:
    mid3_all_categories_list = json.load(f)  # The JSON file contains the list of 1538 API names

big_all_categories_list = [0,1,2,3]
allmid_all_categories_list = ["0-0","0-1","0-2","0-3","1-0","1-1","1-2","1-3","2-0","2-1","2-2","2-3","3-0","3-1","3-2","3-3"]

# Create mappings for the categories
big_category_to_index = {category: idx for idx, category in enumerate(big_all_categories_list)}
big_index_to_category = {idx: category for category, idx in big_category_to_index.items()}
m0_category_to_index = {category: idx for idx, category in enumerate(mid0_all_categories_list)}
m0_index_to_category = {idx: category for category, idx in m0_category_to_index.items()}
m1_category_to_index = {category: idx for idx, category in enumerate(mid1_all_categories_list)}
m1_index_to_category = {idx: category for category, idx in m1_category_to_index.items()}
m2_category_to_index = {category: idx for idx, category in enumerate(mid2_all_categories_list)}
m2_index_to_category = {idx: category for category, idx in m2_category_to_index.items()}
m3_category_to_index = {category: idx for idx, category in enumerate(mid3_all_categories_list)}
m3_index_to_category = {idx: category for category, idx in m3_category_to_index.items()}
allmid_category_to_index = {category: idx for idx, category in enumerate(allmid_all_categories_list)}
allmid_index_to_category = {idx: category for category, idx in allmid_category_to_index.items()}

print(allmid_category_to_index)

# Total number of labels
big_number_of_labels = len(big_all_categories_list)  
m0_number_of_labels = len(mid0_all_categories_list) 
m1_number_of_labels = len(mid1_all_categories_list) 
m2_number_of_labels = len(mid2_all_categories_list) 
m3_number_of_labels = len(mid3_all_categories_list)
allmid_number_of_labels = len(allmid_all_categories_list) 


# Load evaluation data from 'mashup_data_new_test_data.json'
with open(EVALUATION_DATA_PATH, 'r', encoding='utf-8') as f:
    mashup_data = json.load(f)

# Extract evaluation texts and labels from mashup_data_new_test_data.json
evaluation_texts = []
evaluation_labels = []

# Extract categories and descriptions for evaluation
for api_info in mashup_data:
    bigcategories = api_info["bigclusters"]  # Extract categories as labels
    middlecategories = api_info["middleclusters"]
    description = api_info["description"]  # Extract description as the text
    evaluation_texts.append(description)
    evaluation_labels.append(middlecategories)


# Define the BERT model for multilabel classification
class BertForMultilabelClassification(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Pooled output for classification
        logits = self.classifier(pooled_output)
        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# Function to load the saved model
def load_model(model_class, model_name, num_labels, path=None):
    model = model_class(model_name, num_labels)
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    model.eval()  # Set model to evaluation mode
    return model.to('cuda')

# Prediction function for a single text
def predict(text, model, tokenizer, index_to_category, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to('cuda')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    
    predicted_indices = np.where(probs > threshold)[0]
    recommended_categories = [(index_to_category[idx], probs[idx]) for idx in predicted_indices]
    
    return recommended_categories, probs


def evaluate_predictions(true_labels, predicted_labels):
    """
    モデルの予測結果を評価する関数。

    Args:
        true_labels (list of list of int): 正解ラベル（バイナリベクトル）
        predicted_labels (list of list of int): 予測ラベル（バイナリベクトル）

    Returns:
        dict: Precision, Recall, F1スコアなどの評価指標
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

    # 評価指標を計算
    precision = precision_score(true_labels, predicted_labels, average="micro")
    recall = recall_score(true_labels, predicted_labels, average="micro")
    f1 = f1_score(true_labels, predicted_labels, average="micro")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def plot_precision_recall_with_color(precision, recall, thresholds, num_colors=10, colormap_name="tab10"):
    """
    Precision-Recall曲線を閾値ごとに色を変えて描画する。
    
    Args:
        precision (array-like): Precision値のリスト。
        recall (array-like): Recall値のリスト。
        thresholds (array-like): 閾値のリスト。
        num_colors (int): 使用する色の範囲数。デフォルトは10。
    """
    # Normalize thresholds to map them to color indices
    thresholds = np.append(thresholds, 1)  # 最後の点を補間
    norm_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

    # カラーマップを生成 (viridisやplasmaなど好きなものに変更可能)
    colormap = plt.colormaps.get_cmap(colormap_name).resampled(num_colors)

    plt.figure(figsize=(10, 6))
    
    # 各区間を描画
    for i in range(len(precision) - 1):
        plt.plot(
            recall[i:i+2], 
            precision[i:i+2], 
            color=colormap(norm_thresholds[i]),
            linewidth=2
        )

    # カラーバーを追加
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=thresholds.min(), vmax=thresholds.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Thresholds", rotation=270, labelpad=20)

    plt.title('Precision-Recall Curve with Color by Threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.savefig(PRECISION_RECALL_CURVE_OUTPUT_PATH)
    plt.show()

def run_model_and_get_results(text, model, tokenizer, index_to_category, threshold=0.5):
    """
    モデルによる予測を行い、結果を返す関数。

    Args:
        texts (list of str): 入力テキストのリスト
        model (torch.nn.Module): 学習済みモデル
        tokenizer (BertTokenizer): トークナイザー
        index_to_category (dict): カテゴリインデックスからカテゴリ名へのマッピング

    Returns:
        list of list of float: 各テキストに対応する予測確率のリスト
    """
    _, probs = predict(text, model, tokenizer, index_to_category)  # predict関数を利用
    category_predictions = {
        i: float(prob) for i, prob in enumerate(probs)
    }
    # 閾値以上のカテゴリを抽出
    recommended_categories = [
        (category, prob) for category, prob in category_predictions.items() if prob >= threshold
    ]

    return recommended_categories

def convert_labels_to_binary_vectors(labels, category_to_index, num_categories):
    """
    カテゴリ名リストをバイナリベクトルに変換する関数。

    Args:
        labels (list of list of str): 各データポイントのカテゴリ名リスト
        category_to_index (dict): カテゴリ名 -> インデックスのマッピング
        num_categories (int): カテゴリの総数（ベクトルの次元数）

    Returns:
        list of list of int: 各データポイントに対応するバイナリベクトルのリスト
    """
    vector = [0] * num_categories  # 全て0で初期化
    for label in labels:
        if label in category_to_index:  # マッピングが存在する場合のみ処理
            vector[category_to_index[label]] = 1
    return vector



model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

bigmodel = load_model(BertForMultilabelClassification, model_name, big_number_of_labels, path=BMODEL_WEIGHTS_PATH)
m0model = load_model(BertForMultilabelClassification, model_name, m0_number_of_labels, path=M0MODEL_WEIGHTS_PATH)
m1model = load_model(BertForMultilabelClassification, model_name, m1_number_of_labels, path=M1MODEL_WEIGHTS_PATH)
m2model = load_model(BertForMultilabelClassification, model_name, m2_number_of_labels, path=M2MODEL_WEIGHTS_PATH)
m3model = load_model(BertForMultilabelClassification, model_name, m3_number_of_labels, path=M3MODEL_WEIGHTS_PATH)


# Main evaluation function (modified version of main2)
def main2():
    # 実行
    precisions=[]
    recalls=[]
    f1scores=[]
    resultbinary = []
    truebinary = []
    for idx, text in enumerate(evaluation_texts):
        allrecommended = []
        print(f"Processing text {idx + 1}/{len(evaluation_texts)}")
        
        bigresults = run_model_and_get_results(
            text=text,
            model=bigmodel,
            tokenizer=tokenizer,
            index_to_category=big_index_to_category,
        )

        #print(bigresults)
        for nextrun in bigresults:
            if nextrun[0] == 0:
                mid0results = run_model_and_get_results(
                text=text,
                model=m0model,
                tokenizer=tokenizer,
                index_to_category=m0_index_to_category,
                )
                # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                recommended_indices = [category[0] for category in mid0results]  # インデックスを抽出
                # インデックスからカテゴリ名に変換
                recommended_categories = [m0_index_to_category[index] for index in recommended_indices]
                #print("Recommended Categories:", recommended_categories)
                allrecommended+=recommended_categories

            elif nextrun[0] == 1:
                mid1results = run_model_and_get_results(
                text=text,
                model=m1model,
                tokenizer=tokenizer,
                index_to_category=m1_index_to_category,
                )
                # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                recommended_indices = [category[0] for category in mid1results]  # インデックスを抽出
                # インデックスからカテゴリ名に変換
                recommended_categories = [m1_index_to_category[index] for index in recommended_indices]
                #print("Recommended Categories:", recommended_categories)
                allrecommended+=recommended_categories

            elif nextrun[0] == 2:
                mid2results = run_model_and_get_results(
                text=text,
                model=m2model,
                tokenizer=tokenizer,
                index_to_category=m2_index_to_category,
                )
                # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                recommended_indices = [category[0] for category in mid2results]  # インデックスを抽出
                # インデックスからカテゴリ名に変換
                recommended_categories = [m2_index_to_category[index] for index in recommended_indices]
                #print("Recommended Categories:", recommended_categories)
                allrecommended+=recommended_categories

            elif nextrun[0] == 3:
                mid3results = run_model_and_get_results(
                text=text,
                model=m3model,
                tokenizer=tokenizer,
                index_to_category=m3_index_to_category,
                )
                # 推薦カテゴリを取得（仮に `mid0results` が推奨カテゴリのリスト形式だと仮定）
                recommended_indices = [category[0] for category in mid3results]  # インデックスを抽出
                # インデックスからカテゴリ名に変換
                recommended_categories = [m3_index_to_category[index] for index in recommended_indices]
                #print("Recommended Categories:", recommended_categories)
                allrecommended+=recommended_categories

        print(allrecommended)
        print(evaluation_labels[idx])
        allrecommended_binary = convert_labels_to_binary_vectors(allrecommended,allmid_category_to_index,allmid_number_of_labels)
        evaluate_label_binary = convert_labels_to_binary_vectors(evaluation_labels[idx],allmid_category_to_index,allmid_number_of_labels)
        resultbinary.append(allrecommended_binary)
        truebinary.append(evaluate_label_binary)
        #print(allrecommended_binary)
        #print(evaluate_label_binary)
    
    array_resultbinary=np.array(resultbinary)
    array_truebinary=np.array(truebinary)
    print(classification_report(array_truebinary,array_resultbinary,target_names=list(allmid_index_to_category.values())))


if __name__ == "__main__":
    main2()