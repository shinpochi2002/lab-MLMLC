import json

jsonl_path = "/home/nakagawa/logs/mashup_dis_nosp_2layer_4_api_20251209_122857/mashup_dis_nosp_2layer_4_api_20251209_122857_records.jsonl"

pred_counts = []
true_counts = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        pred_counts.append(len(data["pred_apis"]))
        true_counts.append(len(data["true_apis"]))


average_pred = sum(pred_counts) / len(pred_counts)
average_true = sum(true_counts) / len(true_counts)
print("平均 pred_apis 数:", average_pred)
print("平均 true_apis 数:", average_true)
