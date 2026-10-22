# 学習回数・保存容量の算出

**ユーザーの追加指示「時間見積もりは必要ない」に従い、GPU時間の見積もりは実施しない。** GPUが利用できないため、1epoch秒数・VRAMの新規実測値も存在しない。空欄を0秒・0GBとして集計したり、CPUテストや過去の小規模スモークをGPU時間へ換算したりしていない。

## 学習回数の再計算

| 工程 | 計算根拠 | 学習回数 |
| --- | --- | --- |
| API条件選択 | (166 CV Fold + 47 Holdout) × 3比率 × 2lr | 1278 |
| カテゴリ条件選択 | (2×5 CV Fold + 651 Holdout) × 2lr | 1322 |
| API最終再学習 | 97選択対象 + 55固定設定 | 152 |
| カテゴリ最終再学習 | 全653分類器 | 653 |
| 合計 | APIは16構成で共用し16倍しない | 3405 |

API CV50分類器のFold内訳は2-fold20、3-fold10、4-fold4、5-fold16。合計40+30+16+80=166 Fold。Holdout47を足して213組。固定設定55は保存診断Foldがあっても条件選択0回である。最終再学習は選ばれた比率1つだけで、152×3としていない。固定設定55は既存の比率1・lr2e-5・10epochを使う。

カテゴリは5-fold2個、Holdout651個、計661組。新実装はHoldoutでも2lrを比較する。lr/epochを選ぶための条件別学習2,600回と、新規初期化からの最終805回を分けた結果、3,405回という元報告の値を再現した。最大50epochを掛けてGPU所要時間を断定する方法は用いていない。

全条件のUID・比率・Fold・lr・batch数はselection_run_inventory.csv、集計はtraining_cost_counts.json。必要なら確認できるデータ量の係数も記録した。全選択runを1epochずつ処理する場合、APIはTrain16,896batch／Val176,322batch、カテゴリはTrain192,424batch／Val48,560batch（batch8）となる。APIは小さいTrainでもValidationが大きく、Train件数だけを計算負荷の指標にすることはできない。これは計算時間の推定ではない。

## 保存容量

過去Phase 3の実ファイル7個を今回読み取り、1モデルは438,010,669〜438,167,533 byteだった。これは過去checkpointの容量実測であり、新GPU実行のcheckpoint容量ではない。既存FP32 state_dictのBERT部分437,928,960 byte、出力1個あたりLinear重み・bias3,076 byte、既存torch serialization overhead最大78,633 byteから、各分類器の固定出力次元に応じて算出した。バージョン等による保存形式差はあり得る。

| 保存対象 | 算出容量 |
| --- | --- |
| API最終152モデル | 66.580 GB（62.007 GiB） |
| カテゴリ最終653モデル | 286.028 GB（266.385 GiB） |
| 最終モデル計805 | 352.608 GB（328.392 GiB） |
| API条件選択checkpoint 1,278個 | 559.796 GB（521.350 GiB） |
| カテゴリ条件選択checkpoint 1,322個 | 579.065 GB（539.296 GiB） |
| 現実装ですべて保持した計3,405個 | 1,491.469 GB（1,389.039 GiB） |

観測時の空きは約3,087.108 GB（2,875.094 GiB）。最終805モデルの約352.6 GBに対しては余裕がある。現実装で全選択checkpointを残す場合は約1.491 TBで、残りは概算1.60 TBとなる。ログ・既存の並行使用分・安全余裕は別枠で確保する必要があり、ストレージが足りてもGPUが利用可能になるわけではない。

run_manifestやラベルJSON、training_resultsは通常のテキストで、全epochの重みは含まない。ただしtraining_resultsはhistoryと全runのTrain/Val ID、支持数、hashを保持するため、合計容量は実行内容に依存する。今回生成した報告・CSV・設定は数MB規模で、最終実測合計をartifact_manifest.jsonへ記録する。将来の全ログ容量の上限は確認していない。

GPU用のactivation/gradient/Adam状態はVRAMまたはRAMであり、checkpointのディスク容量と同一ではない。既存runはAdam optimizer状態を保存せず、resume完全checkpointではない。GPU offload用ディスクキャッシュや専用一時領域をこのエンジンは定義していない。GPU一時VRAM容量は未実測。

## checkpointの保持方針

現コードは各epochを保存する方式ではなく、各条件の最良stateをメモリに保持して、run終了後に条件ごと1個保存する。最大epoch50をcheckpoint数へ掛けない。ただし本格実行では2,600条件分をすべて保存するため、そのままでは大量保持になる。今回はGPU不在のため新規学習checkpointを一切保存していない。

提案は、将来の新規本格出力でのみ、既存runへopt-inのcheckpoint保持引数を小さく追加すること。既定動作を維持しつつ、final-onlyまたはselected-onlyを選べるようにし、全条件の設定・seed・入力/コードhash・history・best_epoch・最良state hash・選択理由は保持する。最終モデルはFoldモデルを使い回さずBERTから再初期化する既存仕様を維持する。全epochの保存や新しい損失/選択方式は導入しない。

最終のみ保持する場合のモデル容量は約352.6 GB。選択checkpointを1分類器単位で逐次処理する案では、既存全条件を一時保持した場合の最大working set概算は約13.579 GB（12.646 GiB）だが、既存runには自動抑制機能がない。既存ファイルの削除は提案の実施対象にせず、今回は何も削除・圧縮・移動していない。保持方針の変更は未実装案として扱う。
