import re
import ast
import json
import csv
import os


def parse_log(log_path: str):
    """
    ログをパースして、以下を返す：
      - records: 各テキストごとの結果（index, text, 推薦過程, pred_apis, true_apis, precision, recall, f1）
      - summary_stats: テスト全体平均
      - per_api_stats: APIごとの指標
      - label_avg_stats: ラベルごとの平均
    """
    records = []
    summary_stats = {}
    per_api_stats = {}
    label_avg_stats = {}

    current = {}
    current_label = None
    in_summary = False
    in_label_avg = False

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue

            # ===== 「テストごと平均」以降（集計パート） =====
            if in_summary:
                # 「ラベルごと平均」の見出し
                if stripped == "ラベルごと平均":
                    in_label_avg = True
                    continue

                # 「ラベルごと平均」の dict 本体
                if in_label_avg:
                    try:
                        metrics = ast.literal_eval(stripped)
                    except Exception:
                        continue
                    if isinstance(metrics, dict):
                        label_avg_stats = metrics
                    in_label_avg = False
                    continue

                # テスト全体平均 (Precision/Recall/F-Score)
                if stripped.startswith("Precision:"):
                    m = re.search(
                        r"Precision: ([0-9.]+), Recall: ([0-9.]+), F-Score: ([0-9.]+)",
                        stripped,
                    )
                    if m:
                        summary_stats["precision"] = float(m.group(1))
                        summary_stats["recall"] = float(m.group(2))
                        summary_stats["f1"] = float(m.group(3))
                    continue

                # Average False Positive / False Negative
                if stripped.startswith("Average False Positive:"):
                    m = re.search(
                        r"Average False Positive: ([0-9.]+), False Negative: ([0-9.]+)",
                        stripped,
                    )
                    if m:
                        summary_stats["avg_false_positive"] = float(m.group(1))
                        summary_stats["avg_false_negative"] = float(m.group(2))
                    continue

                # APIごとの集計: "api_name { ... }"
                m = re.match(r"(\S+)\s+(\{.*\})", stripped)
                if m:
                    api_name = m.group(1)
                    metrics_str = m.group(2)
                    try:
                        metrics = ast.literal_eval(metrics_str)
                    except Exception:
                        continue
                    if isinstance(metrics, dict):
                        per_api_stats[api_name] = metrics
                    continue

                # それ以外の行は特に扱わない
                continue

            # ===== テキストごとのブロック部分 =====

            # 「テストごと平均」が始まったら、ここでモード切り替え
            if stripped == "テストごと平均":
                if "index" in current:
                    records.append(current)
                    current = {}
                    current_label = None
                in_summary = True
                continue

            # ブロック開始: "Processing text i/N"
            if stripped.startswith("Processing text"):
                # 1つ前のブロックを保存
                if "index" in current:
                    records.append(current)
                    current = {}
                    current_label = None

                m = re.match(r"Processing text (\d+)/(\d+)", stripped)
                if m:
                    current["index"] = int(m.group(1))
                    current["total"] = int(m.group(2))
                    current["推薦過程"] = []
                continue

            # まだどのブロックにも入っていない場合（先頭の nohup 行など）は無視
            if "index" not in current:
                continue

            # 入力テキスト（ブロック内で最初に来る素の行）
            if (
                "text" not in current
                and not stripped.startswith("[")
                and not stripped.startswith("Precision:")
                and not stripped.startswith("False Positive:")
            ):
                current["text"] = stripped
                continue

            # 「◯◯のAPI推薦」見出し
            if stripped.endswith("のAPI推薦"):
                label = stripped.replace("のAPI推薦", "")
                current_label = label
                continue

            # 角カッコで始まる行: リストとみなす
            if stripped.startswith("[") and "]" in stripped:
                try:
                    obj = ast.literal_eval(stripped)
                except Exception:
                    continue

                if not isinstance(obj, list):
                    continue

                # ラベル付きのAPI推薦結果
                if current_label is not None:
                    # 形式: {"label": "entertainment", "apis": [...]}
                    current["推薦過程"].append({"label": current_label, "apis": obj})
                    current_label = None
                    continue

                # ラベル文脈のない空リストはノイズとして無視（例: Polux.fr のところの []）
                if len(obj) == 0:
                    continue

                # タプルのリスト → クラスタ/カテゴリなどの中間結果
                if isinstance(obj[0], tuple):
                    current["推薦過程"].append(obj)
                    continue

                # 文字列のリスト → pred_apis / true_apis
                if all(isinstance(x, str) for x in obj):
                    if "pred_apis" not in current:
                        current["pred_apis"] = obj
                    elif "true_apis" not in current:
                        current["true_apis"] = obj
                    else:
                        # 3つ目以降の文字列リストは今回は不要なので無視
                        pass
                    continue

                # その他のリストは特に使わない
                continue

            # テキストごとの Precision / Recall / F-Score
            if stripped.startswith("Precision:"):
                m = re.search(
                    r"Precision: ([0-9.]+), Recall: ([0-9.]+), F-Score: ([0-9.]+)",
                    stripped,
                )
                if m:
                    current["precision"] = float(m.group(1))
                    current["recall"] = float(m.group(2))
                    current["f1"] = float(m.group(3))
                continue

            # False Positive / False Negative は今回の要件には不要なので無視

    # 「テストごと平均」が出てこないまま終わった場合の保険
    if not in_summary and "index" in current:
        records.append(current)

    return records, summary_stats, per_api_stats, label_avg_stats


if __name__ == "__main__":
    # 処理したいログファイル
    log_path = "logs/mashup_dis_nosp_2layer_4_api_20251209_122857.log"  # パスは環境に合わせて修正

    # ログファイル名から ID を自動生成（拡張子を除いた部分）
    base = os.path.basename(log_path)
    ind_id, _ = os.path.splitext(base)

    # 出力先ディレクトリ作成
    out_dir = os.path.join("logs", ind_id)
    os.makedirs(out_dir, exist_ok=True)

    records, summary_stats, per_api_stats, label_avg_stats = parse_log(log_path)

    # ★ 要件どおり、必要な8項目だけを JSONL に書き出す
    needed_keys = ["index", "text", "推薦過程", "pred_apis", "true_apis", "precision", "recall", "f1"]

    with open(os.path.join(out_dir, f"{ind_id}_records.jsonl"), "w", encoding="utf-8") as jf:
        for rec in records:
            out_rec = {k: rec.get(k, None) for k in needed_keys}
            jf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # もし全体平均や APIごとの集計も残しておきたければ、以下も有効化
    with open(os.path.join(out_dir, f"{ind_id}_summary_stats.json"), "w", encoding="utf-8") as sf:
        json.dump(summary_stats, sf, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, f"{ind_id}_label_avg_stats.json"), "w", encoding="utf-8") as lf:
        json.dump(label_avg_stats, lf, ensure_ascii=False, indent=2)

    api_fieldnames = ["api_name", "precision", "recall", "f1", "TP", "FP", "FN"]
    with open(os.path.join(out_dir, f"{ind_id}_per_api_stats.csv"), "w", newline="", encoding="utf-8") as af:
        writer = csv.DictWriter(af, fieldnames=api_fieldnames)
        writer.writeheader()
        for name, metrics in per_api_stats.items():
            row = {"api_name": name}
            for k in ["precision", "recall", "f1", "TP", "FP", "FN"]:
                row[k] = metrics.get(k, "")
            writer.writerow(row)
