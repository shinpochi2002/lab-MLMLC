# 既存実装の再利用・互換性

**データ接続はPASS、GPU実行の互換性は未確認。既存コード変更は0件。**

対象実装：`scripts/hierarchical_api_selection/api_negative_sampling/training.py`、同pipeline.py、train_api_classifier.py、train_category_classifier.py。正式入力はPhase 1 group_5fold_v1、Phase 2 classifier_cv_v1、Phase 3 api_negative_sampling_v2。旧v1 API負例や旧共通Foldへ切り替えていない。既存コードを実際に読み、既存ローダーで6代表をロードし、既存verify_inputsで入力・コード・生成物の保存ハッシュを照合した。再生成関数は呼んでいない。

## 再利用できたもの

| 処理 | 実装と今回の確認 |
| --- | --- |
| 入力保護 | training.verify_inputs(正式v2)を読み取り実行し成功。manifestハッシュの更新なし |
| データローダー | training.load_classifierを6代表に実行。本文はPhase 1 unified_servicesをentity_idで参照、教師は分類器members.target_labels。すべて解決 |
| 既存分割 | APIはv2の比率別JSON、カテゴリはPhase 2保存folds。旧コードの内部再分割を呼ばない |
| 比率接続 | API152分類器×3条件を読み、各FoldのValidation同一・正例不変・負例集合の包含と件数を確認 |
| 重み | 既存safe_weightsへ保存Trainだけを渡し、全API条件で保存重み一致。正例/負例0の保護あり |
| 条件選択 | choose_configが全保存Fold×2lrを要求し、val_lossでlr、APIの比率はF1→Recall→Precision→少ない負例、epochは最良epoch中央値 |
| Early Stopping | train_oneの最大50epoch・patience3・最良重み復元を静的確認。既存CPUモックテストでbest_epoch=2、停止epoch=5、復元を実確認 |
| 最終再学習 | 選択条件でtrain_oneを再初期化、early_stop=False。カテゴリは全members、APIは選択比率の保存final_trainingを使用 |
| 保存・推論形式 | bert.* / classifier.* state_dict＋固定順ラベルJSON。既存legacy_inference_classは旧推論モジュール全体を起動せずクラスASTだけ抽出 |
| 既存checkpoint読込 | 既存7個のpthをtorch.load(weights_only=True,mmap=True,map_location=cpu)で読取。1/5/52出力のLinear形状とFP32容量を確認 |

40,099件の接続チェックはlogs/data_connection_checks.jsonに記録。これはGPU上でモデルを動かした検証数ではない。

## 旧学習コードをそのまま起動しない理由

旧 `mashup_clu_mlc_nosp_nostop_layer1〜4.py` はファイル順80/20を内部生成し、観測ラベルから出力を作り、全study由来pos_weightを使う箇所がある。旧API category_kfold.pyも内部CV/Holdout、ValidationからTrainへの救済移動、固定50epoch最終学習を含む。これらを起動すると、今回維持するPhase 2/3の分割・固定出力・選択方針をそのまま使えない。既存のPhase 3エンジンを優先する方が変更が少なく、現在の研究方針と整合する。

ユーザーがカテゴリの現CV/Holdout維持を今回明示したため、前回監査の全層K-fold方針に関する保留を今回の阻害要因へ持ち越していない。APIの固定設定55件も既存方針として維持した。固定Validation100は階層選定、Test236は最終評価という役割を維持し、今回いずれも学習・条件選択・性能評価に使用していない。

## 代表6件

| 区分 | 分類器UID | 方式 | Train | Val | 出力数 | GPU結果 |
| --- | --- | --- | --- | --- | --- | --- |
| api_cv_multi | api_classification/category_internet_things | kfold | 14 | 1316 | 5 | SKIP_NO_CUDA |
| api_holdout_multi | api_classification/category_mapping | holdout | 2772 | 791 | 52 | SKIP_NO_CUDA |
| api_single_output | api_classification/category_content | holdout | 90 | 788 | 1 | SKIP_NO_CUDA |
| api_one_positive | api_classification/category_booking | fixed_config | 2 | 792 | 1 | SKIP_NO_CUDA |
| category_holdout_multi | category_classification/layer2_4_43/layer_01_parent_root | holdout | 17526 | 4381 | 4 | SKIP_NO_CUDA |
| category_fivefold | category_classification/layer3_12_6_2/layer_03_parent_5_3 | kfold | 9 | 3 | 2 | SKIP_NO_CUDA |

Train/Valは保存fold_01の全件数。APIは負例比率1、カテゴリは負例抽出なし。configs/representatives.jsonには全ID、固定出力順、lr、batch8、max_length512、FP32、予定epoch1を保存した。**GPU初期化失敗によりこの設定ではまだ実行していない。** bookingのValidationは負例のみの診断用で、条件選択には接続しない。分類器数を条件選択可能な件数と混同しない。

## 既存テストの今回の実行

既存tests/test_phase3.pyの8テストを選択実行し、8 PASS・FAIL0・ERROR0・SKIP0。対象は重み、空Train拒否、lr/比率の選択と同点処理、不完全/非有限結果の拒否、Early Stopping復元、単一出力lossの勾配。splitter/generatorを実行するテストは選択していない。

Early StoppingテストはCPUの小型モデル・tokenizer・Validation値をモックする既存テストで、実BERTやGPUの検証ではない。単一出力の勾配テストもCPUのlossテストである。GPUのforward/backward・optimizer更新・lr実適用・最良epoch保存・GPU checkpoint round-trip・旧推論とのlogits一致は、すべて未実施と区別した。過去Phase 3のBERTスモーク報告を今回のGPU成功として引用していない。

## 必要最小限の追加・残る制約

今回は新規の読取集計・報告生成スクリプトだけを追加し、学習ループは新設していない。GPU復旧後の1epoch確認には既存training.train_oneへepochs=1、batch_size=8、max_length=512、device=cuda:0を実行引数として渡せる。CLIには汎用--epochs/--batch-size引数がなく、--smokeはmax_length32・batch2・Train/Val各最大4件・1Fold・2epochへ制限する。このスモーク結果を通常512トークンの1epoch測定と呼ばないこと。

GPUが使える状態で、既存train_oneを呼ぶ外側の薄い実行・メモリ計測ラッパーを加えればGPU動作確認できる見込みだが、今は作成・実行していない。1epochだけではpatience3の停止発火までは確認できないため、停止発火のGPU検証は別に必要。推論の出力形式は静的互換だが、CUDA実行・保存復元・実BERT logitsの一致は保証しない。

ストレージ節に記したように、本格runは条件ごとの最良checkpointをすべて永続保存する。checkpoint保持方針を抑制する実行引数はない。保持数を制御する最小変更は将来案として提示し、今回はコードを自動修正していない。
