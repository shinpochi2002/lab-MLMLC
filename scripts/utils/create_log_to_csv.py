import csv
import re
import ast

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

def safe_parse_literal(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def is_valid_label_list(obj):
    # ['payments', ...] のような str のリストだけを通す
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)

def is_list_line(s):
    s = s.strip()
    return s.startswith('[') and s.endswith(']')

def process_log_filtering_tuples(log_path, csv_output_path):
    results = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Processing text"):
            entry = {}

            # テキスト番号
            match = re.search(r"Processing text (\d+)/\d+", line)
            entry["text_index"] = int(match.group(1)) if match else -1

            # 説明テキスト
            i += 1
            if i >= len(lines): break
            entry["raw_text"] = lines[i].strip()

            # ラベルとして妥当な [str, str, ...] リストを2つ見つける
            predicted, true = [], []
            label_lists_found = 0
            while i + 1 < len(lines) and label_lists_found < 2:
                i += 1
                line_candidate = lines[i].strip()
                if is_list_line(line_candidate):
                    parsed = safe_parse_literal(line_candidate)
                    if is_valid_label_list(parsed):
                        if label_lists_found == 0:
                            predicted = parsed
                        elif label_lists_found == 1:
                            true = parsed
                        label_lists_found += 1

            entry["predicted"] = predicted
            entry["true"] = true

            # スコア抽出
            precision = recall = f1 = -1.0
            while i + 1 < len(lines):
                i += 1
                score_line = lines[i].strip()
                if "Precision:" in score_line:
                    try:
                        precision, recall, f1 = re.findall(r"[-+]?\d*\.\d+|\d+", score_line)
                        precision = float(precision)
                        recall = float(recall)
                        f1 = float(f1)
                    except Exception:
                        pass
                    break

            entry["precision"] = precision
            entry["recall"] = recall
            entry["f1"] = f1

            results.append(entry)
        else:
            i += 1

    # CSV 出力
    if results:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["text_index", "raw_text", "predicted", "true", "precision", "recall", "f1"])
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"[✓] CSV出力完了: {csv_output_path}")
    else:
        print("[!] 有効なデータがありませんでした。")

# 使用例
if __name__ == "__main__":
    process_log_filtering_tuples(legacy_path("/home/nakagawa/small4_4_result_5_14.log"), "small4_4_result_5_14.csv")
