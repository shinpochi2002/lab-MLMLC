# 分類器単位CV/Holdout：実装結果

既存Phase 1と全層共通Group 5-Foldを保持したまま、新しい独立分割を作成した。モデル学習・推論・負例生成は行っていない。

## 分類器数

| 項目 | 件数 |
| --- | ---: |
| classifier_count | 805 |
| category_classifier_count | 653 |
| api_classifier_count | 152 |
| kfold_partition_count | 97 |
| fivefold_partition_count | 32 |
| reduced_fold_partition_count | 65 |
| holdout_partition_count | 698 |
| independent_partition_unavailable_count | 10 |
| single_output_classifier_count | 68 |
| single_output_cv_partition_count | 56 |
| single_output_holdout_partition_count | 2 |
| single_output_unavailable_count | 10 |
| constant_target_classifier_count | 68 |
| kfold_needing_training_policy_decision | 95 |
| classifier_with_unseen_train_labels_count | 11 |
| classifier_with_train_negative_zero_labels_count | 61 |
| training_or_inference_runs | 0 |

上記CV件数は**分割が作れた分類器数**であり、通常の教師学習・評価に成功した分類器数ではない。単一出力の全正例、Train未観測ラベル、負例0、既存pos_weight=0は別に記録する。
API分類器は152個を全16構成で共用するため16倍しない。カテゴリ653個は構成別IDを維持した。

## データと共有候補

カテゴリ21,907サービス・API正例4,855サービスは既存ファイル参照で保持し、本文を再出力・変更していない。分類器教師は固定経路の親prefixと出力一覧から再現した。
カテゴリ共有候補の等価群は23組。元ID・教師・出力順・意味対応・規定学習条件を比較したが、今回カテゴリ分類器の統合は適用していない。選択後のlr・epoch等も一致した場合にのみ将来のモデル共有を検討する。

## 保存形式

1分類器1JSONに教師のentity/group/target、全FoldのTrain/Validation IDとラベル正負支持数をまとめた。本文はPhase 1を参照する。Foldごとに別ファイルを大量に作る方式を避け、805個の分割ファイルと共通manifestで管理する。

## 未解決事項・Phase 3

- 正例のみのAPI暫定版。負例追加後は件数・CV閾値・ラベル分布が変わるため新しい出力先で再分割する。
- 1groupでは独立分割不能。固定予測・代替モデル・例の追加は実装していない。
- 単一出力全正例では全0負例を捏造せず、pos_weight=0や識別情報なしを明示する。
- Train正例0・負例0の出力の損失・予測規則と、最良epoch/最終再学習条件を決定する必要がある。
- 既存実装同様のValidationによるESと条件選択は独立した汎化推定ではない。固定評価での最終推薦性能とは区別する。
- 固定評価カテゴリGTの既知の2件の不整合はPhase 1の記録を引き継ぎ、固定ファイルを変更しない。

## 検証

checks/validation_results.jsonに元ファイル保護、教師再現、分類器内group分離、CVのValidation一巡、seed再現、全件数・例外検出と成果物ハッシュを保存する。
モデル本体の内容ハッシュは取得せず、既存全ファイルのメタデータと宣言入力・コードのSHA-256を照合する。
