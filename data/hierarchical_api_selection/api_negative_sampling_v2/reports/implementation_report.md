# Phase 3 詳細実装・実行報告

## 1. 結論と今回の範囲

既存Phase 1/2を保持し、152 API分類器の負例候補・3比率のTrain・再分割を新規作成した。学習率/epoch選択、最良重み保存復元、BERTからの最終再学習、安全なclass weight、カテゴリ分類への同エンジン適用を実装した。代表4種類の実物BERTによるCPUスモークを完了した。

**全805分類器の本格学習、全16階層の推薦性能評価、固定Validation100件/Test236件を使う学習条件選択・推論は実施していない。スモークの選択値は研究用の最適設定ではない。**

## 2. 不変条件と既存成果物の保護

カテゴリサービス21,907、Mashup3,946、API正例4,855、候補909、カテゴリ152、階層16を維持。APIカテゴリ別の正例割当は延べ6,514件。元データの本文・教師・groupを変更しない。source_manifestの宣言入力875件と新規コード/README12件をSHA-256保護し、生成前に存在した14,766ファイルのsize/mtime_ns/mode/symlinkを再照合した。既存モデル内容全体はハッシュしていないため、その保証範囲を過大に述べない。

Phase 1の除外manifestから固定評価の実体ID・正規化本文との交差がないことを確認。固定336件そのものを負例や条件選択に使用しない。近似重複や未発見の別名を完全に検出する保証、過去の評価データ利用履歴を取り消す保証はない。

既存の `data_usage_redesign_proposal.md` は61 Mashup復元前の提案値21,846/4,794を記載している。今回の基準は復元後のPhase 1保存値21,907/4,855。Phase 2の全正例CV97個はカテゴリ2個を含み、APIは95個だった。以下の比較はAPI152個だけを対象とする。

## 3. 教師・負例定義とサービス数

分類器cのoutput_labels順を保持し、Mashupの候補内使用APIとcのAPI集合の交差をmulti-hot化する。交差が空なら全0負例。負例候補は3,946学習Mashupからだけ作り、909自己説明文・17,052候補外API・固定評価を負例にしない。API自己説明文は自己API正例として保持する。未使用は観測関係に基づく0であり、本質的に不要なAPIである証明ではない。

各分類器の母集合は全3,946 Mashupとそのカテゴリの自己API説明文。サービスは複数カテゴリへ再利用される。**延べ正例6,514件＋延べ負例候補594,187件＝延べ600,701件**であり、独立したサービス数ではない。独立したAPI学習母集合は従来と同じ4,855サービス。正例と負例に同一entity_idを二重登録しない。

## 4. 分割・抽出policy

policy: `api_negative_group_anchor_positive_threshold_v1`。seed42。API自己説明文を持つgroupはTrain固定。残るMashup groupを正例を含むgroup/負例だけのgroupに事前層別し、各層内で既存の決定的なgroup allocatorを利用する。分割後のValidation→Train移動は行わない。自己説明文に由来する特異な性能を混ぜず、Mashupに対する指標を条件選択に使うための設計。

- 正例合計20未満、検証可能な独立Mashup正例groupが2以上：min(5,正例group数)-fold。
- 正例合計20以上、同groupが2以上：正負groupそれぞれ概ね80/20 Holdout。少数層でもValidationに正例groupを最低1つ確保する容量を事前決定する。
- それ以外：正例groupをすべてTrain固定し、負例のみの診断Holdoutを保存。通常CV/条件選択は無効。共通設定比率1、lr2e-5、10epochを使う。これは事前の保守的な実装用設定で、データから最適と検証された値ではない。

分割の後、Trainの正例数（Mashup＋自己API）×比率を要求負例数とする。正例と同groupの負例は必ず同時採用。それ以外は分類器UID/seed42のSHA-256順でgroupを追加し、要求数へ到達したら止める。最後のgroupは分割しない。比率1⊆3⊆5。不足を複製で埋めず、超過・実比率・選択entity_idを保存する。Validationと正例Trainは比率間で完全共通。

API補助groupを常時Trainに置くため、全レコードを一巡する通常のK-foldではない。検証へ回せるMashupだけがFoldを一巡する。分類器間のFoldは独立で、階層OOF性能測定用ではない。

## 5. API分類器のCV/Holdout比較

| 方式 | Phase 2 | Phase 3 |
| --- | --- | --- |
| K-fold | 95 | 50 |
| Holdout | 47 | 47 |
| 分割不能/共通固定設定 | 10（旧分割不能） | 55 |


| Phase 3方式 | Fold数 | 分類器数 |
| --- | --- | --- |
| fixed_config | 0 | 55 |
| holdout | 0 | 47 |
| kfold | 2 | 20 |
| kfold | 3 | 10 |
| kfold | 4 | 4 |
| kfold | 5 | 16 |


共通設定55個は正例1件10個と、正例2件以上でも独立したMashup検証正例group不足の45個。自己API正例しか持たない分類器を、負例を増やしただけで検証可能になったと数えない。例外も負例付き最終学習データを持ち、152個すべてを学習対象として保持する。

## 6. 3比率の実件数

以下は全開発母集団を使う**最終再学習用候補データ**の分類器別延べ件数。3条件は重複集合なので合計しない。Fold学習用件数は `negative_sample_counts.csv` の各fold行へすべて保存し、final_refit行と区別する。55例外は3条件を作成するが、現policyで実際に採用するのは比率1のみ。

| 比率 | Mashup正例延べ | API自己正例延べ | 正例計 | 負例延べ | Train延べ計 | 不足分類器数 | group超過分類器数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 5605 | 909 | 6514 | 6514 | 13028 | 0 | 0 |
| 1:3 | 5605 | 909 | 6514 | 16660 | 23174 | 1 | 0 |
| 1:5 | 5605 | 909 | 6514 | 25935 | 32449 | 2 | 1 |


各Foldには負例Train候補全体のID、抽出されたID、要求/実比率、不足、group丸め、Validationの正負内訳を記録。Foldの学習集合を足し合わせると同じサービスが繰り返し現れる。

## 7. 単一出力と損失補正

| 負例比率 | 単一出力分類器数 | 正例延べ | 負例延べ | 正負両方を持つ分類器 |
| --- | --- | --- | --- | --- |
| 1 | 68 | 234 | 234 | 68 |
| 3 | 68 | 234 | 702 | 68 |
| 5 | 68 | 234 | 1171 | 68 |


68個すべてを単一出力Sigmoidとして保持。親カテゴリが予測されたことだけでAPIを推薦しない。閾値未満なら非推薦。既存コードに>0.5と>=0.5が混在するため、新規実装は>=0.5で固定しmanifestに記録した。

全APIのTrain支持数行（各比率・Fold・finalを含む）では正例0行0、負例0行0、補正行0、補正対象分類器0。自己APIをTrainに保持したことにより全API出力の正例支持を維持し、負例追加で全正例分布を解消した。学習精度が向上したことを意味するものではない。

各Trainのみからpos_weight=neg/pos。正例0または負例0なら1.0へ補正、有限値を[1e-4,1e4]に制限。負例0のとき旧重み0で正例損失が消える問題を回避する。重み補正は、未観測正例の意味上の不足を解決しない。カテゴリ分類の既存Foldにも同じ安全計算を適用した場合、0分類器・補正0出力/Fold行（{}）が対象になる。これはカテゴリ教師の変更ではなく、新学習エンジンの重み規則の変更。

## 8. 学習条件選択・モデル保存

各比率でlr2e-5/3e-5を比較。最大50epoch、patience3、val_loss最小epochの重みをCPUへ保存し復元する。本格実行では最良checkpointを別名でディスクにも保存する。スモークは各caseで最初のlrの最良checkpointを保存し、他方はメモリ保存復元とハッシュ照合。最良重みでPrecision/Recall/micro-F1/val_lossを記録する。

lrは比率内のFold平均Loss最小、同点F1最大、lr小。負例比率は選択lrの平均F1最大、同点Recall/Precision最大、比率小。epochは選択比率/lrの最良epoch中央値を四捨五入し1..50。Fold平均は等重みで、サービス数を重みとするpool集計とは異なる。

最終学習はBERT事前学習初期値へ戻し、全正例と全開発母集団から選択比率で再抽出した負例で学習する。旧Validationのサービスも最終学習候補へ戻すが、全負例候補を無条件に使用するわけではない。最終学習にはESを行わない。裸のstate_dict（bert.*, classifier.*）と出力順ラベルJSONを新規保存する。

カテゴリ分類はPhase 2の653個の教師・Foldをそのまま参照し、負例抽出なし、2lr、最良epoch中央値と全教師再学習を同じエンジンで利用できる。API補助のTrain固定ルールをカテゴリデータへ遡及適用していない。カテゴリの本格学習・実物BERTによるカテゴリ経路のスモークは今回未実行。

## 9. 小規模動作検証

| case | 分類器 | lr×Fold結果数 | 選択lr | 選択epoch（スモーク） | 実final epoch | 秒 | 既存推論互換 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sufficient_multi | category_mapping | 2 | 3e-05 | 2 | 2 | 12.16 | True |
| few_positive_multi | category_internet_things | 2 | 3e-05 | 2 | 2 | 9.55 | True |
| single_output | category_content | 2 | 2e-05 | 1 | 1 | 8.44 | True |
| one_positive | category_booking | 0 | 2e-05 | 10 | 2 | 2.97 | True |


ローカル `bert-base-uncased` の12層/hidden768を使用。CPU2スレッド、max_length32、各Train/Validation最大4件、1Fold、2epoch、比率1のみ。BERTの凍結はせず全パラメータへ勾配を通した。正例1件caseは通常CVを実行せず共通設定を通し、最終動作確認は2epochへ短縮した。固定epoch10の妥当性を検証した実験ではない。

各caseでBERT初期化、有限Loss、最終保存/再ロード、既存推論クラスへのstrict loadとlogits完全一致を確認。選択対象caseでは2lrで同一初期重みハッシュ、最良epoch記録・最良重み復元も確認。閾値以下の非推薦判定は通常のSigmoid処理を使用する。既存推論コードのmodule-level評価ループは実行せず、クラスASTのみ抽出して互換性を確認した。既存推論のモデル探索パスや固定評価データは実行しない。

スモーク結果は `results/hierarchical_api_selection/phase3_smoke_v1/` のcase別JSON/モデル。全比率の実物BERT比較・全Fold比較・50epoch完走・GPU実行・カテゴリ学習経路の実物BERT検証は未確認。3比率選択規則は合成結果の単体テストで確認した。


### スモークの部分抽出で生じる補正の区別

上の補正0件は、保存した全件のFold/最終学習候補に対する集計である。スモークは最大4件へ縮小するため、以下の正例0出力が別途生じる。その場合にも重み1へ補正し、有限Lossで学習できた。ただし、この動作確認はそのラベルを十分に学習した証拠ではない。重複するlr試行の行を、独立した分類器数として数えない。

| case | 学習区分 | 正例0の出力数 | 負例0の出力数 | 補正出力数 |
| --- | --- | --- | --- | --- |
| sufficient_multi | lr=2e-05 | 49 | 0 | 49 |
| sufficient_multi | lr=3e-05 | 49 | 0 | 49 |
| sufficient_multi | final_refit | 49 | 0 | 49 |
| few_positive_multi | lr=2e-05 | 2 | 0 | 2 |
| few_positive_multi | lr=3e-05 | 2 | 0 | 2 |
| few_positive_multi | final_refit | 2 | 0 | 2 |
| single_output | lr=2e-05 | 0 | 0 | 0 |
| single_output | lr=3e-05 | 0 | 0 | 0 |
| single_output | final_refit | 0 | 0 | 0 |
| one_positive | final_refit | 0 | 0 | 0 |

## 10. 自動検証と既存ファイル保護

データ準備検証：1,881,673条件passed。実体/本文の固定評価非混入、正例教師の保持、全0負例の定義、API由来負例の不在、Train/Validation group非重複、3比率の同一Validation・包含関係、全group採用、Trainのみの重み計算、同seed再生成を確認。最終検証はREADME/本報告等の作成後、`checks/validation_results.json`へ別途保存する（ready検証を上書きしない）。

単体テスト19件passed。比率包含、不足・group超過、正負混在groupの同時採用、API group固定、少数Fold、正例1件、正例group不足、Holdout group分離、有限weight、片側支持0でも非ゼロ勾配、欠落/重複/NaNの選択拒否、同点規則、中央値epoch、既存出力拒否を確認。ES合成テストではLoss列1.0,0.5,0.7,0.8,0.9でepoch5停止・epoch2の重みへ復元することも確認した。

既存保護はsource_manifestのSHA-256と全既存ファイルの保存メタデータを照合。全既存モデル内容のSHA-256をとった保証ではない。作業開始時の13,309件等の過去Phase 2報告の件数と、今回開始時のファイル数は新成果物追加により異なる。

## 11. 本格学習の計算量

API条件選択1278回＋最終152回、カテゴリ条件選択1322回＋最終653回、合計3405回のモデル学習。最大step概算16,675,510。詳細は `training_cost_estimate.md/json`。GPU時間は未確認。スモーク秒数から512token本格実験時間を外挿しない。

## 12. 新規コードと成果物

新規コードは `scripts/hierarchical_api_selection/api_negative_sampling/` のみ。データ成果物は `data/hierarchical_api_selection/api_negative_sampling_v2/`、動作確認重み/記録は `results/hierarchical_api_selection/phase3_smoke_v1/`。既存出力が存在すると無条件上書きを拒否する。既存ファイルを移動・改名しない。各入力・コード・データ・checkpointのハッシュ/由来/条件をmanifestへ記録する。

- `manifests/`: 入力/実装/既存ファイル保護、負例抽出方針と各条件のパス・ハッシュ、学習policy、152分類器一覧。
- `splits/api_classification/`: 152分類器の教師、group、正負由来、抽出前分割。
- `negative_samples/negative_ratio_1|3|5/`: 各152個、Foldと最終再学習用ID・件数・重み。
- `reports/negative_sample_counts.csv`: 全Fold/全比率/finalの実件数。
- `reports/classifier_training_feasibility.csv`: 分割前後比較と例外理由。
- `reports/train_label_weights.csv`: 各出力の正負支持と補正。
- `reports/training_cost_estimate.*`: 回数・service-epoch・step上限。
- `checks/`: 単体テスト、ready検証、最終検証と成果物ハッシュ。

初回生成の `api_negative_sampling_v1/` は保存したまま、正式検証対象を `api_negative_sampling_v2/` とした。初回生成後にスモークの単一出力対象weather（実際は出力11）をcontent（出力1）へ修正したため、入力/コードhashを上書きせず新しいパスへ生成した。教師・負例・分割policyは変更していない。旧v1は学習に使わない。

## 13. 未解決事項と次段階

1. 共通設定55個のlr/epoch/比率は最適性未確認。固定評価を使わず、十分な開発データの他分類器から設定を移す研究設計に変更する選択肢がある。現状は事前固定設定を採用したことを明記する。
2. API補助group固定により各APIのTrain正例を保証するが、API説明文からの汎化性能そのものはこのValidationで測らない。API正例だけのカテゴリはMashup再現率を測れない。
3. 正負不均衡とpos_weightが連動する。全0負例の真の関連性は観測データから保証できない。負例数増加が最終階層推薦F1改善に直結するかは未確認。
4. 同じ開発ValidationをES・lr・比率選択に使うため選択バイアスがある。固定100/236は学習条件選択に使わず、後続の階層選定/最終評価でのみ使用する。過去に利用済みのTestの独立性はこの作業だけでは回復しない。
5. 本格前にGPUのVRAM/1epoch時間/保存容量を確認し、checkpoint保持方針を決める。モデル自動配置や既存推論の切替、Eプロット、階層選定、最終評価は今回の範囲外。
6. CPUスモークは2epochで、実験結果を推薦性能として報告できない。全3比率×2lr×全Foldで条件選択し、BERTから最終再学習する実験は次段階。

## 付録A. 全152分類器の件数と分割

負例数は各比率のfinal_refit候補。Fold別の詳細はCSV/JSONを参照。単一出力も削除しない。

| 分類器 | 出力数 | Mashup正例 | API正例 | 負例候補 | 旧方式 | 新方式 | Fold数 | 負例1 | 負例3 | 負例5 | 例外理由 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category_accounting | 2 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 3 | 9 | 15 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_addresses | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_advertising | 11 | 55 | 11 | 3891 | holdout | holdout | 0 | 66 | 198 | 330 | — |
| category_air_travel | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_analytics | 8 | 25 | 8 | 3921 | holdout | holdout | 0 | 33 | 99 | 165 | — |
| category_animals | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_api | 2 | 5 | 2 | 3941 | kfold | kfold | 5 | 7 | 21 | 35 | — |
| category_application_development | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_applications | 3 | 3 | 3 | 3943 | kfold | kfold | 3 | 6 | 18 | 30 | — |
| category_astronomy | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_authentication | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_auto | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_backend | 5 | 7 | 5 | 3939 | kfold | kfold | 5 | 12 | 36 | 60 | — |
| category_barcodes | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_big_data | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_biometrics | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_bitcoin | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_blockchain | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_blogging | 9 | 28 | 9 | 3918 | holdout | holdout | 0 | 37 | 111 | 185 | — |
| category_booking | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_bookmarks | 5 | 47 | 5 | 3899 | holdout | holdout | 0 | 52 | 156 | 260 | — |
| category_books | 1 | 7 | 1 | 3939 | kfold | kfold | 5 | 8 | 24 | 40 | — |
| category_bots | 2 | 3 | 2 | 3943 | kfold | kfold | 3 | 5 | 15 | 25 | — |
| category_business | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_calendars | 4 | 17 | 4 | 3929 | holdout | holdout | 0 | 21 | 63 | 105 | — |
| category_chat | 7 | 12 | 7 | 3934 | kfold | kfold | 5 | 19 | 57 | 95 | — |
| category_classification | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_cloud | 4 | 37 | 4 | 3909 | holdout | holdout | 0 | 41 | 123 | 205 | — |
| category_collaboration | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_content | 1 | 55 | 1 | 3891 | holdout | holdout | 0 | 56 | 168 | 280 | — |
| category_coupons | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_covid_19 | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_cryptocurrency | 3 | 2 | 3 | 3944 | kfold | kfold | 2 | 5 | 15 | 25 | — |
| category_cycling | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_data | 2 | 3 | 2 | 3943 | kfold | kfold | 3 | 5 | 15 | 25 | — |
| category_data_as_a_service | 4 | 4 | 4 | 3942 | kfold | kfold | 4 | 8 | 24 | 40 | — |
| category_data_mining | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_database | 12 | 42 | 12 | 3904 | holdout | holdout | 0 | 54 | 162 | 270 | — |
| category_design | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_developers | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_documents | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_domains | 4 | 1 | 4 | 3945 | kfold | fixed_config | 0 | 5 | 15 | 25 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_ecommerce | 54 | 290 | 54 | 3656 | holdout | holdout | 0 | 344 | 1032 | 1720 | — |
| category_education | 12 | 26 | 12 | 3920 | holdout | holdout | 0 | 38 | 114 | 190 | — |
| category_electronic_signature | 1 | 21 | 1 | 3925 | holdout | holdout | 0 | 22 | 66 | 110 | — |
| category_email | 14 | 44 | 14 | 3902 | holdout | holdout | 0 | 58 | 174 | 290 | — |
| category_enterprise | 17 | 14 | 17 | 3932 | holdout | holdout | 0 | 31 | 93 | 155 | — |
| category_entertainment | 5 | 14 | 5 | 3932 | kfold | kfold | 5 | 19 | 57 | 95 | — |
| category_environment | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_esports | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_events | 6 | 19 | 6 | 3927 | holdout | holdout | 0 | 25 | 75 | 125 | — |
| category_fax | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_feedback | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_feeds | 4 | 14 | 4 | 3932 | kfold | kfold | 5 | 18 | 54 | 90 | — |
| category_file_sharing | 6 | 5 | 6 | 3941 | kfold | kfold | 5 | 11 | 33 | 55 | — |
| category_financial | 20 | 35 | 20 | 3911 | holdout | holdout | 0 | 55 | 165 | 275 | — |
| category_fitness | 2 | 18 | 2 | 3928 | holdout | holdout | 0 | 20 | 60 | 100 | — |
| category_food | 4 | 6 | 4 | 3940 | kfold | kfold | 5 | 10 | 30 | 50 | — |
| category_forms | 1 | 4 | 1 | 3942 | kfold | kfold | 4 | 5 | 15 | 25 | — |
| category_framework | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_gambling | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_games | 6 | 13 | 6 | 3933 | kfold | kfold | 5 | 19 | 57 | 95 | — |
| category_geography | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_government | 27 | 24 | 27 | 3922 | holdout | holdout | 0 | 51 | 153 | 255 | — |
| category_hardware | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_health | 2 | 3 | 2 | 3943 | kfold | kfold | 3 | 5 | 15 | 25 | — |
| category_healthcare | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_home_automation | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_hosting | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_identity | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 11 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_images | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_intelligence | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_internet_things | 5 | 3 | 5 | 3943 | kfold | kfold | 3 | 8 | 24 | 40 | — |
| category_invoicing | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_jobs | 6 | 30 | 6 | 3916 | holdout | holdout | 0 | 36 | 108 | 180 | — |
| category_language | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_learning_management_systems | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_library | 1 | 3 | 1 | 3943 | kfold | kfold | 3 | 4 | 12 | 20 | — |
| category_location | 2 | 3 | 2 | 3943 | kfold | kfold | 3 | 5 | 15 | 25 | — |
| category_machine_learning | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_mail | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_mapping | 52 | 1668 | 52 | 2278 | holdout | holdout | 0 | 1720 | 2278 | 2278 | — |
| category_marketing | 2 | 4 | 2 | 3942 | kfold | kfold | 4 | 6 | 18 | 30 | — |
| category_media | 8 | 20 | 8 | 3926 | holdout | holdout | 0 | 28 | 84 | 140 | — |
| category_medical | 5 | 3 | 5 | 3943 | kfold | kfold | 3 | 8 | 24 | 40 | — |
| category_medicine | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_merchants | 3 | 2 | 3 | 3944 | kfold | kfold | 2 | 5 | 15 | 25 | — |
| category_messaging | 22 | 179 | 22 | 3767 | holdout | holdout | 0 | 201 | 603 | 1005 | — |
| category_monitoring | 1 | 4 | 1 | 3942 | kfold | kfold | 4 | 5 | 15 | 25 | — |
| category_movies | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_music | 29 | 115 | 29 | 3831 | holdout | holdout | 0 | 144 | 432 | 720 | — |
| category_natural_language_processing | 2 | 0 | 2 | 3946 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_news_services | 12 | 33 | 12 | 3913 | holdout | holdout | 0 | 45 | 135 | 225 | — |
| category_nutrition | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_office | 11 | 16 | 11 | 3930 | holdout | holdout | 0 | 27 | 81 | 135 | — |
| category_other | 25 | 75 | 25 | 3871 | holdout | holdout | 0 | 100 | 300 | 500 | — |
| category_parsing | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_payments | 15 | 33 | 15 | 3913 | holdout | holdout | 0 | 48 | 144 | 240 | — |
| category_pdf | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_performance | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_personal_information_management | 7 | 17 | 7 | 3929 | holdout | holdout | 0 | 24 | 72 | 120 | — |
| category_photos | 26 | 359 | 26 | 3587 | holdout | holdout | 0 | 385 | 1155 | 1925 | — |
| category_platform_as_a_service | 1 | 2 | 1 | 3944 | kfold | kfold | 2 | 3 | 9 | 15 | — |
| category_politics | 1 | 3 | 1 | 3943 | kfold | kfold | 3 | 4 | 12 | 20 | — |
| category_predictions | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_printing | 2 | 2 | 2 | 3944 | kfold | kfold | 2 | 4 | 12 | 20 | — |
| category_products | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_project_management | 11 | 19 | 11 | 3927 | holdout | holdout | 0 | 30 | 90 | 150 | — |
| category_purchasing | 1 | 3 | 1 | 3943 | kfold | kfold | 3 | 4 | 12 | 20 | — |
| category_qa | 2 | 5 | 2 | 3941 | kfold | kfold | 5 | 7 | 21 | 35 | — |
| category_random | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_real_estate | 7 | 17 | 7 | 3929 | holdout | holdout | 0 | 24 | 72 | 120 | — |
| category_recognition | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_recommendations | 5 | 35 | 5 | 3911 | holdout | holdout | 0 | 40 | 120 | 200 | — |
| category_reference | 39 | 127 | 39 | 3819 | holdout | holdout | 0 | 166 | 498 | 830 | — |
| category_reporting | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_rewards | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_ride_share | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_sales | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_science | 8 | 15 | 8 | 3931 | holdout | holdout | 0 | 23 | 69 | 115 | — |
| category_screenshots | 1 | 5 | 1 | 3941 | kfold | kfold | 5 | 6 | 18 | 30 | — |
| category_search | 35 | 119 | 35 | 3827 | holdout | holdout | 0 | 154 | 462 | 770 | — |
| category_security | 8 | 12 | 8 | 3934 | holdout | holdout | 0 | 20 | 60 | 100 | — |
| category_semantics | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_sentiment | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_seo | 5 | 5 | 5 | 3941 | kfold | kfold | 5 | 10 | 30 | 50 | — |
| category_shipping | 5 | 14 | 5 | 3932 | kfold | kfold | 5 | 19 | 57 | 95 | — |
| category_social | 48 | 670 | 48 | 3276 | holdout | holdout | 0 | 718 | 2154 | 3276 | — |
| category_spam | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_sports | 9 | 15 | 9 | 3931 | holdout | holdout | 0 | 24 | 72 | 120 | — |
| category_spreadsheets | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_standards | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_statistics | 2 | 7 | 2 | 3939 | kfold | kfold | 5 | 9 | 27 | 45 | — |
| category_stocks | 2 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 3 | 9 | 15 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_storage | 7 | 51 | 7 | 3895 | holdout | holdout | 0 | 58 | 174 | 290 | — |
| category_streaming | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_support | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_tagging | 1 | 6 | 1 | 3940 | kfold | kfold | 5 | 7 | 21 | 35 | — |
| category_telephony | 18 | 327 | 18 | 3619 | holdout | holdout | 0 | 345 | 1035 | 1725 | — |
| category_text | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_tools | 38 | 113 | 38 | 3833 | holdout | holdout | 0 | 151 | 453 | 755 | — |
| category_translation | 2 | 19 | 2 | 3927 | holdout | holdout | 0 | 21 | 63 | 105 | — |
| category_transportation | 10 | 16 | 10 | 3930 | holdout | holdout | 0 | 26 | 78 | 130 | — |
| category_travel | 24 | 43 | 24 | 3903 | holdout | holdout | 0 | 67 | 201 | 335 | — |
| category_url_shortener | 2 | 21 | 2 | 3925 | holdout | holdout | 0 | 23 | 69 | 115 | — |
| category_video | 16 | 293 | 16 | 3653 | holdout | holdout | 0 | 309 | 927 | 1545 | — |
| category_voice | 1 | 0 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | 3 | 5 | one_positive_record |
| category_weather | 11 | 29 | 11 | 3917 | holdout | holdout | 0 | 40 | 120 | 200 | — |
| category_web_site_management | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_wi_fi | 1 | 1 | 1 | 3945 | kfold | fixed_config | 0 | 2 | 6 | 10 | fewer_than_two_nonanchored_mashup_positive_groups |
| category_widgets | 4 | 61 | 4 | 3885 | holdout | holdout | 0 | 65 | 195 | 325 | — |
| category_wiki | 3 | 6 | 3 | 3940 | kfold | kfold | 5 | 9 | 27 | 45 | — |


## 付録B. 新規実装の根拠（関数・行番号）

- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:44` — `sampling_order`
- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:50` — `sample_negatives`
- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:68` — `safe_weights`
- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:82` — `build_split`
- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:151` — `build_classifier`
- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:169` — `create`
- `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:257` — `validate`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:17` — `choose_config`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:48` — `imports`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:54` — `model_class`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:66` — `seed_everything`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:74` — `state_hash`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:81` — `legacy_inference_class`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:95` — `encode`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:103` — `make_loss`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:109` — `metrics_from_counts`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:116` — `evaluate`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:129` — `train_one`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:171` — `verify_inputs`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:180` — `load_classifier`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:194` — `bounded_subset`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:202` — `run`
- `scripts/hierarchical_api_selection/api_negative_sampling/training.py:285` — `main`
- `scripts/hierarchical_api_selection/api_negative_sampling/run_smoke_tests.py:11` — `main`
- `scripts/hierarchical_api_selection/api_negative_sampling/tests/test_phase3.py:11` — `member`
- `scripts/hierarchical_api_selection/api_negative_sampling/tests/test_phase3.py:14` — `population`
- `scripts/hierarchical_api_selection/api_negative_sampling/tests/test_phase3.py:17` — `PhaseThreeTests`

既存参考：`scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py:124` load_categoryは空used_apisを除外するため、今回の全0負例loaderには流用しない。同:159の分割後のラベル保持移動はgroup分離を壊し得るため再利用しない。同:272 train_kfold_then_finalの2lr/50epochを維持するが、旧実装の最終epoch50固定・最良重みcheckpoint不保存を、今回の中央値epoch/最良重み復元に変更した。`scripts/inference/mashup_dis_nosp_category_api.py:17` のクラス構造・forwardと保存state_dictの互換性を確認した。論文本体の参照は不要という最新依頼に従い、論文との直接一致は主張しない。

## 付録C. 最終照合用の追加記録

`checks/execution_evidence.json` に、実際のPython/ライブラリバージョン、作業開始時の既存14,138ファイルの変更0件、スモーク成果物20ファイルのSHA-256、最終再学習もFold試行と同じ事前学習初期値から始まったことのハッシュ照合を保存した。スモークは合計約33秒のCPU動作確認であり、データ検証や準備時間を含む作業全体の時間ではない。

本報告の最終校正では、生成用テンプレートに残った無関係な延べ件数の文言を「独立したサービス数ではない」へ整理し、既存関数名を `train_kfold_then_final` と明記した。またスモーク部分抽出の補正件数を実行JSONから追記した。データ・分割・学習結果の値は変更していない。
