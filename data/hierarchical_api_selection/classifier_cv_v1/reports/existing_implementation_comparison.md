# 既存実装と分類器単位の分割の比較

根拠は現在の既存ソースコード。論文本体と直接照合したという主張はしない。最新依頼の明示仕様に従った。

| 項目 | 既存コード | 今回の分割 |
| --- | --- | --- |
| APIの少数CV条件 | 有効n<20かつ出力ラベル数>1 | n<20でG>=2なら単一出力でも試行 |
| n>=20 | 80/20 Holdout | 同じ適用閾値、group単位で約80/20 |
| n=1 | 複数ラベルならk=2へ進んで失敗し得る | 独立group1として分割不能を明示 |
| Fold数 | max(2,min(5,n)) | min(5,独立group数)、G<2は分割しない |
| 層化 | MultilabelStratifiedKFold、group非対応 | 分類器固有の正解分布からgroup対応の希少ラベル優先割当。完全な同一割当ではない |
| 単一出力 | 少数でもHoldoutへ送る | 正例/負例のbinary支持数を考慮。全正例は分割成立と教師分布不成立を区別 |
| 未観測Trainラベル | ValidationからTrainへ移す補正あり | 補正移動せず、該当ラベルと分類器を記録 |
| カテゴリ各段 | ファイル内の順序で80/20 | 全653分類器にも同じn/G条件を適用 |
| seed | 分割seed42 | 各分類器に独立して42を適用。旧全層Foldは読み込まない |
| CV学習率 | 2e-5,3e-5 | 次の学習用設定として記録。今回は選択・学習しない |
| 選択規則 | 平均val_loss最小、同値なら平均micro-F1最大 | 同じ規則をmanifestへ記録 |
| Holdout学習率 | 2e-5固定 | 同じ値を記録 |
| Early Stopping | 分類器のValidation loss、patience3、最大50epoch | 同じ利用目的を記録。旧共通Foldの内部ES割当は使用しない |
| CV後 | 選択lrで全nを50epoch、Validationなしで再学習 | 次の学習の参照条件として記録。今回は再学習しない |
| 出力空間 | 当該データに観測されたAPIから構築 | 固定909候補のカテゴリ別部分集合・既存階層の出力一覧を固定 |
| 負例 | 空used_apisは現行autoローダーで除外 | Phase 1の正例のみ。新規負例は作らない |

根拠ファイル：`scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py` の `load_category` 124–155行、`split_with_all_labels` 159–209行、`train_holdout` 222–270行、`train_kfold_then_final` 272–386行、`train_one_category_auto` 399–430行。単一ラベル条件は416行。
カテゴリ段は同ディレクトリの `mashup_clu_mlc_nosp_nostop_layer{1,2,3,4}.py`。例：layer3の127–146行で教師化・全studyからの重み計算・80/20スライス、163–176行で停止と保存。

既存trainerは最良checkpointの復元を行わず停止時の重みを保存する箇所があり、CV後50epoch固定が最適とは確認していない。損失重み、epoch数、最良重みの扱いはPhase 3の判断事項。

今回はモデル全体のOOF推薦性能を計測する分割ではない。同一サービスが別分類器のTrainとValidationへ現れることは許容される。分類器内では実体・本文groupを厳密に分離する。固定Validation100/Test236はいずれも分類器の条件選択・停止・教師に使わない。
