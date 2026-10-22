# Phase 3: API負例と分類器別条件選択

実装/実行詳細は[implementation_report.md](reports/implementation_report.md)。本格学習・固定評価は未実行。実物BERT CPUスモーク4例を完了。

| 比率 | Mashup正例延べ | API自己正例延べ | 正例計 | 負例延べ | Train延べ計 | 不足分類器数 | group超過分類器数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 5605 | 909 | 6514 | 6514 | 13028 | 0 | 0 |
| 1:3 | 5605 | 909 | 6514 | 16660 | 23174 | 1 | 0 |
| 1:5 | 5605 | 909 | 6514 | 25935 | 32449 | 2 | 1 |


独立サービス数は4,855（Mashup3,946+自己API909）。表は分類器別の延べ件数。各条件を合計しない。CV50/Holdout47/共通固定設定55。正例1件10個・単一出力68個を保持。

[コードと再現手順](../../../scripts/hierarchical_api_selection/api_negative_sampling/README.md)、[比較計画](reports/ratio_comparison_plan.md)、[計算量](reports/training_cost_estimate.md)、[学習可能性CSV](reports/classifier_training_feasibility.csv)、[Fold件数CSV](reports/negative_sample_counts.csv)。

負例は観測上未使用のAPIであり、本質的に不要なAPIである証明ではない。固定Validation100/Test236を負例・CV・ES・条件選択に使用しない。既存Phase 1/2・コード・モデルは変更しない。既存出力へ上書きしない。
