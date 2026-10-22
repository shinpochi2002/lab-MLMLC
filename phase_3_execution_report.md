# Phase 3：API分類器への負例導入と学習条件選択の詳細実行報告

作成日時：2026-10-16 19:21:22 JST。対象リポジトリ：`/home/nakagawa/lab-MLMLC`。

本書はPhase 3で実施した実装・生成・小規模学習を、現在のソースコード、保存済みJSON/CSV、実行記録、SHA-256と再集計に基づいて説明する。**本書作成のためのモデル学習・再学習・推論評価・データ再生成は実施していない。新規作成するのは本報告書のみであり、Phase 3の保存済みコード・データ・モデル・報告書を変更していない。**

## 1. 実行結果の要点と読み方

**152 API分類器の負例候補、Trainの負例比率1:1/1:3/1:5、負例導入後の独立group分割を作成した。学習率・epoch選択と最終再学習のコードを実装し、実物の事前学習済みBERTによる4例のCPU小規模学習を完了した。全16階層の本格学習と、固定Validation100件/Test236件による推薦性能評価は未実施である。**

正式なデータ成果物は [api_negative_sampling_v2/README.md](data/hierarchical_api_selection/api_negative_sampling_v2/README.md)。初回生成の `api_negative_sampling_v1/` も保持したが、正式な学習入力として扱わない。版の経緯は第3節を参照。

| 項目 | 確認結果 | 数字の意味 |
| --- | --- | --- |
| API分類器 | 152 | 全16階層で共用。16倍しない |
| CV | 50分類器 | 分割データの成立。全Foldの実学習成功数ではない |
| Holdout | 47分類器 | 正例を含むMashup Validationを持つ |
| 共通固定設定 | 55分類器 | 通常の条件選択を行わず学習対象として保持 |
| 正例1レコードのみ | 10分類器 | 上記55個に含む。APIラベル単体の希少性とは別の集計 |
| 出力1ラベルのみ | 68分類器 | 全件候補データで正負両方の教師を確保 |
| データ自動検証 | 1,881,673条件passed | 保存済み検証のassertion件数。独立したテストケース数ではない |
| 単体テスト | 19件passed | 保存済み記録。今回再学習・再実行していない |
| 実物BERTの小規模学習 | 4種類、計10回のモデル学習 | 3種類×2lr＋4種類の最終動作確認 |
| 本格学習の見積もり | 3,405回 | 将来の計画上の回数。今回実行した回数ではない |

以下では「実装済み」「生成済み」「実学習で確認済み」「未実施/未検証」「今後の判断事項」を区別する。延べ件数には同じサービスの分類器間・Fold間の再利用が含まれる。

## 2. 研究条件と変更しなかったもの

| 条件 | 確認値 | 扱い |
| --- | --- | --- |
| 推薦候補API | 909 | 固定 |
| APIカテゴリ | 152 | 固定 |
| 階層構成 | 16 | 既存Balanced K-means定義を参照 |
| カテゴリ分類用サービス | 21907 | 教師・本文・groupを変更しない |
| Mashup由来学習サービス | 3946 | 負例生成元はこの集合だけ |
| 候補API自己説明文 | 909 | 自己API正例。負例にはしない |
| API分類用正例の独立サービス | 4855 | 3,946＋909。カテゴリ用集合の部分集合 |
| 候補外API説明文 | 17052 | カテゴリ分類にのみ利用。API負例にしない |
| 階層構造選定用固定Validation | 100 | 負例生成・CV・ES・条件選択に使用しない |
| 最終固定Test | 236 | 負例生成・CV・ES・条件選択に使用しない |
| seed | 42 | 分割・抽出・学習初期化の規則に記録 |


既存のPhase 1、旧全層共通Group 5-Fold、Phase 2の分類器別CV/Holdout、既存学習コード、既存モデル・checkpoint、既存ログ・評価結果は保持した。word2vec/Balanced K-meansの再実行は行っていない。旧分類器の重みを新規学習の初期値にしていない。

参照元：`data/hierarchical_api_selection/group_5fold_v1/source_records/unified_services.json`、同`api_classification_positive.json`、同`manifests/label_schema.json`、Phase 3の`manifests/source_manifest.json`。固定評価除外の読み取り確認は `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:62` の`load_phase_one`、Phase 3生成の呼び出しは `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:169`（`create`）。

## 3. 参照資料・成果物の版と生成記録

参照資料は `data_usage_audit.md`、`data_usage_redesign_proposal.md`、`phase_1_2_implementation_summary.md`、`classifier_cv_execution_report.md`、Phase 1/2保存データ、既存カテゴリ/API学習・推論コード。論文本体との直接一致を確認したという主張はしない。

| 記録 | 保存値 |
| --- | --- |
| Phase 3 v2生成記録 | 2026-10-16 19:05:37 JST |
| 最終データ検証完了記録 | 2026-10-16 19:10:25 JST |
| 参照Git commit | 5b8c7f5251128843b313b88304c3f2d4261e1cf0 |
| 正式出力 | data/hierarchical_api_selection/api_negative_sampling_v2/ |
| policy名 | api_negative_group_anchor_positive_threshold_v1 |
| 生成入力のSHA-256対象 | 875 |
| 新規実装/READMEのSHA-256対象 | 12 |


`api_negative_sampling_v1/`は初回生成を保存したもの。初回生成後、単一出力のスモーク対象として指定したweatherが実際には11出力であることを確認し、content（1出力）へ修正した。コードハッシュを既存manifestへ上書きせず、`api_negative_sampling_v2/`へ生成し直した。**負例定義・教師・分割policyを変更した版ではない。** 旧v1の最終検証は作成していないため、本格学習には正式v2を使う。根拠：[旧v1のREADME](data/hierarchical_api_selection/api_negative_sampling_v1/README.md)、[新規コードのREADME](scripts/hierarchical_api_selection/api_negative_sampling/README.md)。

既存再構成案の21,846/4,794は61 Mashup復元前の提案値。Phase 3の基準はPhase 1確定後の21,907/4,855であり、今回その差分を新しく追加・変更したものではない。

## 4. API正例・負例の定義とデータの追跡

### 4.1 カテゴリ別の教師

カテゴリcの固定出力API集合をA_c、Mashupの候補内使用API集合をA_mとする。教師は出力ラベル順に `1[api ∈ A_m ∩ A_c]` を並べたmulti-hot。交差が空なら全0負例、空でなければMashup正例とする。実装は `scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:151`（`build_classifier`）。

- `mashup_positive`：当該カテゴリの候補APIを1つ以上使用するMashup。
- `api_self_positive`：当該カテゴリの候補API自身の説明文。自己APIを正例とする。
- `mashup_negative`：当該カテゴリの候補APIの使用が観測されないMashup。

**負例は「観測された使用関係にない」という意味であり、そのAPIが本質的に不要であることを証明する教師ではない。** 候補外APIの説明文を全0教師へ置き換える処理はない。現在のPhase 1の`api_labels`は909候補へ限定されており、当該カテゴリとの交差により教師を作る。候補外の使用関係を後日入力へ追加する場合も、候補外APIを候補内APIの正例と見なすことはしない。

同一entity_idを同じ分類器の正例と負例へ二重登録しない。同じMashupが別カテゴリで正例/負例となることは許容する。本文はPhase 1のentity_idで参照し、各分割JSONに本文の複製は持たない。

### 4.2 独立サービス数とカテゴリ展開後の件数

| 集合 | 件数 | 集計単位 |
| --- | --- | --- |
| 元Mashup | 3946 | 独立entity_id |
| 候補API自己説明文 | 909 | 独立entity_id |
| API学習母集団の和集合 | 4855 | 上記の和。全条件共通 |
| カテゴリ別Mashup正例 | 5605 | 152分類器へ展開した延べレコード |
| カテゴリ別API自己正例 | 909 | 各APIの所属カテゴリへ割当 |
| カテゴリ別正例合計 | 6514 | 5,605＋909 |
| カテゴリ別Mashup負例候補 | 594187 | 152分類器への延べ割当 |
| 正負候補合計 | 600701 | 3,946×152＋909 |


6,514＋594,187を独立サービス数と呼ばない。同じMashupのカテゴリ別再利用による増加である。最終候補の全分類器のTrainを集合としてまとめると、比率1/3/5はいずれも4,855 unique entity_idを含む。比率間で増えるのはカテゴリごとにそのサービスを負例として利用する割当数である。

### 4.3 レコードと保存先

分類器ファイル `splits/api_classification/category_*.json` は、`output_labels`、`members`（entity_id、group_id、source_type、role、target_labels）、`partition`を持つ。比率別ファイルはFoldごとのTrain/Validation ID、負例選択ID・group、要求数/実数/実比率・不足・丸め、Train支持数/pos_weight、最終再学習候補IDを保存する。ラベル順は旧定義を保持し、サンプル内`target_labels`は集合として比較する。

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:151`（`build_classifier`）、`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:169`（`create`）、[classifier_manifest.json](data/hierarchical_api_selection/api_negative_sampling_v2/manifests/classifier_manifest.json)。

## 5. 分割を先に行い、Trainだけで負例を抽出する設計

### 5.1 API補助教師をTrainへ固定

当該カテゴリのAPI自己説明文を持つgroupを常にTrainに固定する。同じgroupのMashupもTrainに残す。ValidationにはMashupだけを使い、API自己説明文の識別性能をMashup指標へ混ぜない。

その後、残るgroupを「正例レコードを含むgroup」「負例レコードだけのgroup」の2層に分け、各層でgroup割当を行う。各ラベルの支持数等をsoftな均衡目標として既存の決定的なallocatorを使い、groupは分割しない。厳密なラベルごとの層別比率、MultilabelStratifiedKFoldと同一の割当を保証するものではない。

### 5.2 CV/Holdout/共通固定設定の判定

P＝自己API説明文を含む当該分類器の正例レコード数。G_pos＝API補助groupを除いた後の、検証へ割当可能なMashup正例group数。

| 条件 | 方式 | 備考 |
| --- | --- | --- |
| P>1、G_pos≥2、P<20 | min(5,G_pos)-fold | 少数groupではFold数を減らす |
| P>1、G_pos≥2、P≥20 | Holdout | 正例層・負例層それぞれ概ね80/20のgroup分割 |
| 上記以外 | fixed_config | 正例groupをTrainへ残し、負例だけの診断Validationを保存 |

Holdoutでは各層のgroup数Gに対し、G≥2ならValidation group数を `max(1,min(G-1,floor(0.2*G+0.5)))` として事前に容量を決める。G=1ならTrainに固定する。分割後にValidationから正例を移し戻す処理はない。正例Validationを必要とする方式で生成できないときは例外を出し、CV成功として進めない。

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:82`（`build_split`）、再利用元 `scripts/hierarchical_api_selection/pipeline.py:328`（`stratify`）。

### 5.3 n_splitsと実際の分割ペア数

保存形式の`n_splits`は**K-foldのKを表し、Holdout/fixed_configでは0**である。ただし両方式とも`partition.folds`には1組のTrain/Validationが保存される。`n_splits=0`を「検証用データが存在しない」と解釈しない。本書の付録ではKと実分割ペア数を別々に示す。

CVの50個では166組、Holdout47個では47組、合わせて**条件選択対象213組**。別にfixed_config55個の診断用55組があり、保存された分割ペアは合計268組である。

### 5.4 グローバルFoldとの関係

分類器ごとに独立して割当し、Phase 2の正例のみのFold IDをそのまま流用しない。旧全層共通Group 5-Foldも使用しない。同じ`fold_01`名でも分類器間で対応する共通Foldではない。API補助groupは常時Trainのため、全4,855サービスを一巡する通常のK-foldでもない。CV一巡の対象は検証へ回せるMashup集合である。

同じgroupが別分類器では別の役割になることはある。これは分類器別の条件選択という設計であり、階層モデル全体のOOF推薦性能を測る設計とは区別する。

## 6. 分割方式の変更前後と例外分類器

| Phase 2方式 | Phase 3方式 | 分類器数 |
| --- | --- | --- |
| holdout | holdout | 47 |
| kfold | fixed_config | 45 |
| kfold | kfold | 50 |
| unavailable | fixed_config | 10 |


| Phase 3方式 | K-foldのK | 分類器数 | 1分類器あたり実分割ペア数 |
| --- | --- | --- | --- |
| fixed_config | 0 | 55 | 1 |
| holdout | 0 | 47 | 1 |
| kfold | 2 | 20 | 2 |
| kfold | 3 | 10 | 3 |
| kfold | 4 | 4 | 4 |
| kfold | 5 | 16 | 5 |


APIのCVは95から50へ減った。負例追加でレコード数が増えたためではなく、**自己API説明文をTrainに固定し、検証可能なMashup正例groupを2以上必要とする基準を導入したため**である。旧CV45個が固定設定になり、旧分割不能10個も負例付き学習対象として保持した。

固定設定55個の内訳は、正例1レコードのみ10個と、正例2レコード以上でも独立した検証用Mashup正例groupが2未満の45個。正例レコードが複数あっても自己説明文中心なら通常のCV成功と数えない。

固定設定は **負例比率1、lr=2e-5、epoch=10**。固定評価、診断Validation、他分類器の実測から最適値を求めたものではなく、事前の保守的な実装用設定である。この3値の妥当性は未検証。`selection_enabled=false`と理由をmanifestに保存する。

正例1レコードのみの分類器は次の10個で、唯一の正例はいずれもAPI自己説明文。Mashup側の正例がないため、Mashupでの正例再現率を測れる状況ではない。

| 分類器 | 正例Mashup | 正例API | 負例候補 |
| --- | --- | --- | --- |
| category_booking | 0 | 1 | 3946 |
| category_hardware | 0 | 1 | 3946 |
| category_learning_management_systems | 0 | 1 | 3946 |
| category_machine_learning | 0 | 1 | 3946 |
| category_predictions | 0 | 1 | 3946 |
| category_recognition | 0 | 1 | 3946 |
| category_reporting | 0 | 1 | 3946 |
| category_sentiment | 0 | 1 | 3946 |
| category_text | 0 | 1 | 3946 |
| category_voice | 0 | 1 | 3946 |


## 7. 負例比率1:1/1:3/1:5の抽出と実件数

### 7.1 決定的抽出

抽出はTrain/Validationを確定した後、Trainの負例候補だけで行う。要求負例数は `Train正例数×比率`。Mashup正例と自己API正例の両方を正例数へ含める。

まず正例Trainと同groupに属する負例を必ず採用する。それ以外は、分類器UID・seed42・group IDから得るSHA-256順でgroupの接頭辞を採用する。要求数に達した最後のgroupを分割しないため、件数が超過することがある。候補不足なら全候補までとし、複製しない。比率1⊆3⊆5の包含関係を維持する。これは分類器依存の決定的疑似ランダム順で、APIラベルごとに負例数を別々に合わせる抽出ではない。

全条件で正例Train、Validation、その評価指標、lr候補、epoch上限、BERT初期化、閾値を共通にする。Train側の負例数に応じてTrain由来のpos_weightも変化するが、重み算出規則自体は全条件共通。

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:44`（`sampling_order`）、`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:50`（`sample_negatives`）。

### 7.2 最終再学習用候補の延べ件数

| 正例:負例の要求比率 | Mashup正例 | API自己正例 | 正例計 | 実負例 | Train計 | unique Trainサービス | unique 負例Mashup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 5605 | 909 | 6514 | 6514 | 13028 | 4855 | 3349 |
| 1:3 | 5605 | 909 | 6514 | 16660 | 23174 | 4855 | 3927 |
| 1:5 | 5605 | 909 | 6514 | 25935 | 32449 | 4855 | 3946 |


この表は**将来の最終再学習で使える候補IDを生成した件数**であり、全件を実学習した件数ではない。3条件は重なるため合算しない。fixed_config55個にも3条件のデータは存在するが、現在の学習policyで使う条件は比率1のみ。

### 7.3 条件選択用213ペアのFold Train延べ件数

| 要求比率 | 正例Mashup | 正例API | 正例計 | 実負例 | Train計 | 共通Validation計 | Validation正例 | Validation負例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 4947 | 1191 | 6138 | 6138 | 12276 | 234379 | 1286 | 233093 |
| 1:3 | 4947 | 1191 | 6138 | 16077 | 22215 | 234379 | 1286 | 233093 |
| 1:5 | 4947 | 1191 | 6138 | 25282 | 31420 | 234379 | 1286 | 233093 |


Validation延べ234,379件は213ペアへのカテゴリ別・Fold別再利用を含み、独立したMashup数ではない。同じValidationを2学習率・3比率で反復評価する。

### 7.4 例外55ペアの診断用データ

| 要求比率 | Train正例 | Train負例 | Train計 | 診断Validation計 | 診断Validation正例 | 診断Validation負例 |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 105 | 105 | 210 | 43443 | 0 | 43443 |
| 1:3 | 105 | 315 | 420 | 43443 | 0 | 43443 |
| 1:5 | 105 | 525 | 630 | 43443 | 0 | 43443 |


診断Validationはすべて負例で、比率/lr/epoch選択には使わない。実装した最終学習もこの診断データをESに使用しない。

### 7.5 要求比率を満たさない実測ケース

| 分類器 | 区分 | 要求比率 | 正例 | 要求負例 | 利用可能負例 | 実負例 | 実負例/正例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| category_mapping | fold_01 | 1:3 | 1386 | 4158 | 1821 | 1821 | 1.313853 |
| category_mapping | final_refit | 1:3 | 1720 | 5160 | 2278 | 2278 | 1.324419 |
| category_mapping | fold_01 | 1:5 | 1386 | 6930 | 1821 | 1821 | 1.313853 |
| category_mapping | final_refit | 1:5 | 1720 | 8600 | 2278 | 2278 | 1.324419 |
| category_social | fold_01 | 1:5 | 584 | 2920 | 2621 | 2621 | 4.488014 |
| category_social | final_refit | 1:5 | 718 | 3590 | 3276 | 3276 | 4.562674 |


mappingでは比率3と5の利用可能負例を使い切るため、実Train集合は同一になる。socialでは比率5で不足する。要求比率名だけを根拠に実比率3/5と解釈しない。

| group保持による超過分類器 | 区分 | 要求比率 | 正例 | 要求負例 | 実負例 | 超過 | 実比率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| category_identity | final_refit | 1:5 | 2 | 10 | 11 | 1 | 5.5 |


identityの比率5最終候補では最後のgroupを保持したため1件超過。現policyのidentityはfixed_configなので、この比率5を通常の最終学習で採用するわけではない。

根拠：[negative_sample_counts.csv](data/hierarchical_api_selection/api_negative_sampling_v2/reports/negative_sample_counts.csv)（1,260行、ヘッダ除く）、各比率JSONの`negative_sampling`。付録Bには268ペアの件数を掲載する。

## 8. 単一出力と安全なclass weight

### 8.1 出力1ラベルの68分類器

| 要求比率 | 分類器数 | Mashup正例延べ | API正例延べ | 正例計 | 負例計 | Train計 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 68 | 166 | 68 | 234 | 234 | 468 |
| 3 | 68 | 166 | 68 | 234 | 702 | 936 |
| 5 | 68 | 166 | 68 | 234 | 1171 | 1405 |


68個すべてが全件候補データで正負両方の教師を持つ。モデルは1出力の線形ヘッド＋Sigmoidのまま保持する。親カテゴリが予測されたことだけで自動推薦しない。閾値は>=0.5で固定し、未満なら当該APIを推薦しない。これは教師分布上の改善であって、推薦性能向上の実証ではない。

旧推論には`>0.5`と`>=0.5`が混在する箇所があり、新規計算は後者をpolicyに明記した。閾値ちょうど0.5の挙動を完全に同一と主張しない。

### 8.2 重みの計算方法

`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:68`（`safe_weights`） と `scripts/hierarchical_api_selection/api_negative_sampling/training.py:103`（`make_loss`）。各出力jについてTrainの正例数p_j、負例数n_jから次を計算する。

```text
p_j > 0 かつ n_j > 0 : pos_weight_j = clip(n_j / p_j, 1e-4, 1e4)
p_j = 0 または n_j = 0 : pos_weight_j = 1.0
```

空Trainは拒否する。重みはBCEWithLogitsLossに渡す。Validationの教師を使って重みを決めない。旧方式のn_j=0による重み0と正例損失項の消失を回避する。p_j=0の場合に学習可能な正例を補う仕組みではなく、数値上の安全性と正例不足の意味上の問題を区別する。

### 8.3 保存された全件データの分布

APIの全比率・Fold・finalの支持数は6,483行。正例0は0行、負例0は0行、補正は0行。実重み範囲は1.0～3997.0。自己API正例をTrainへ固定し、負例を追加した結果であり、補正が全件データで発動したという意味ではない。

カテゴリの既存661ペアへ同じ規則を適用して集計した支持数は3,085出力/ペア行、補正対象0分類器・0行（既存の計算量集計記録）。カテゴリの全件実学習を行った結果ではない。

### 8.4 スモーク部分集合では補正が発生

スモークは4件に縮小するため、複数出力の一部ラベルはTrain支持0になる。mappingは52出力中49出力、internet_thingsは5出力中2出力で正例0となり重み1へ補正した。これは通常候補データの正例不足を表す数字ではない。小規模学習で有限Lossが得られたことは、これらのラベルの識別を十分に学習できた証拠ではない。

## 9. 学習率・負例比率・epochの選択実装

### 9.1 共通の学習設定

| 項目 | 実装 |
| --- | --- |
| 初期値 | ローカルキャッシュのbert-base-uncasedのみ |
| モデル | BertModel.pooler_output→Linear(num_labels) |
| 損失 | BCEWithLogitsLoss、Train由来pos_weight |
| optimizer | torch.optim.AdamW、weight_decay=0、eps=1e-8 |
| lr候補 | 2e-5、3e-5 |
| 通常の最大epoch | 50 |
| Early Stopping | val_loss最小、patience3、改善判定は厳密な< |
| 通常のmax_length/batch_size | 512/8 |
| 判定 | Sigmoid>=0.5 |
| DataLoader | shuffleあり、seed42、num_workers=0 |
| 決定論的演算 | torch.use_deterministic_algorithms(True) |

モデル初期化は `scripts/hierarchical_api_selection/api_negative_sampling/training.py:54`（`model_class`）、学習・最良重み復元は `scripts/hierarchical_api_selection/api_negative_sampling/training.py:129`（`train_one`）、設定選択は `scripts/hierarchical_api_selection/api_negative_sampling/training.py:17`（`choose_config`）。

### 9.2 選択順序

1. 各比率×各lr×各Foldで、val_loss最小epochを記録し、その重みをメモリへ保存する。同Lossの場合は最初のepochを保持する。
2. 最良重みを復元し、最良epoch時のPrecision/Recall/micro-F1/val_lossを保存する。本格実行では各試行の最良checkpointをディスクにも保存する。
3. 各比率内でFold平均val_loss最小のlrを選ぶ。同点は平均micro-F1最大、その次は小さいlr。
4. 比率間では選択lrのFold平均micro-F1最大を選ぶ。同点はRecall最大、Precision最大、最後は小さい負例比率。比率間の主選択指標をLossにしない。
5. 選択比率/lrの最良epochの中央値をround-half-up（例2,5→3.5→4）し、1～50へ制限する。

平均はFold等重みの算術平均で、全Validationをpoolして計算するmicro-F1とは同一ではない。各Fold内のmicro-F1は全サンプル・全出力のTP/FP/FNから計算し、分母0は0として扱う。欠落Fold、重複Fold、非有限指標、不正なepochを拒否する。独立の設定選択CLIはJSONに渡された候補の比較であり、実BERT学習を起動するものではない。

### 9.3 Validationの目的と選択バイアス

同じローカルValidationをESとlr/比率選択に使う。この開発F1は独立した推薦性能推定ではない。固定Validation100件は将来の階層選定用、固定Test236件は将来の最終推薦性能用であり、上記処理には使わない。

### 9.4 最終再学習

選択後は、全正例を含み、ローカルValidationを除外していない全開発母集団から負例を選択比率で再抽出する。全負例候補を無条件に全件使う意味ではない。BERT事前学習初期値へ戻し、選択lr・集約epochで学習する。Foldモデルを初期値にしない。最終学習はESなし。

`run`は事前学習済みBERTのローカルsnapshotと一致する初期値以外を拒否する。学習済み分類器checkpointを初期値にする引数はない。新規出力パスが存在すれば拒否し、モデルはexclusive作成する。

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/training.py:202`（`run`）、`scripts/hierarchical_api_selection/api_negative_sampling/training.py:171`（`verify_inputs`）。今回実際に行った最終再学習の動作確認はスモーク用部分集合だけであり、全件での最終学習は未実行。

## 10. カテゴリ分類器への対応

653カテゴリ分類器のPhase 2教師と分割をそのまま参照し、負例を生成しない。既存は5-fold2分類器とHoldout651分類器、計661ペア。新規学習エンジンでは負例比率0（カテゴリ側で負例抽出を行わない識別値）として2lrを比較し、平均Loss最小・F1同点判定、中央値epoch、全教師での新規最終再学習を利用できる。

API自己説明文のTrain固定というPhase 3 API分割ルールを、既存カテゴリFoldへ遡及適用していない。**カテゴリ学習経路はコード実装・既存分割参照を確認した範囲であり、カテゴリの実物BERTスモークおよび全件学習は今回未実施。** API4例の成功をカテゴリ653個の学習成功としない。

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/training.py:180`（`load_classifier`）、`scripts/hierarchical_api_selection/api_negative_sampling/train_category_classifier.py`、Phase 2`manifests/classifier_manifest.json`。

## 11. 実物BERTによる小規模学習の実行内容

### 11.1 実行環境と制限

| 環境・設定 | 保存値 |
| --- | --- |
| Python | 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0] |
| Python実行パス | /home/nakagawa/venvs/swallow-py312/bin/python |
| numpy | 1.26.4 |
| pytorch-lightning | 2.2.1 |
| torch | 2.10.0+cu128 |
| transformers | 5.8.0 |
| 学習device | CPU、2スレッド |
| 事前学習済みモデル | bert-base-uncased、12層、hidden768 |
| max_length / batch_size | 32 / 2 |
| Train / Validation上限 | 各4件 |
| Fold数 | 選択対象は最初の1Foldのみ |
| 負例比率 | 1のみ |
| epoch上限 | 2 |
| BERT本体の更新 | 凍結せず全パラメータに勾配を通す |
| 大容量モデルのダウンロード | 実施していない |
| スモーク全体の保存経過時間 | 33.252462秒 |


ローカルsnapshotは各caseの`run_manifest.json`に絶対パス・事前学習済みファイルのSHA-256を保存した。実行時はローカルのみの読み込みとoffline環境変数を指定した。CUDA版torchのインストールはGPU学習の実施を意味しない。

スモーク全体は10回のモデル学習、合計19epoch、batch数から算出したoptimizer step合計36。同じモデルの追加継続学習10回ではなく、試行・最終動作確認ごとにBERTから初期化した。再ロード用モデルの初期化は「モデル学習回数」に含めない。

### 11.2 対象と実行結果

| case | 分類器 | 出力数 | 全件Mashup正例 | 全件API正例 | 分割方式 | 選択lr | 選択epoch設定 | 実final epoch | 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sufficient_multi | category_mapping | 52 | 1668 | 52 | holdout | 3e-05 | 2 | 2 | 12.163 |
| few_positive_multi | category_internet_things | 5 | 3 | 5 | kfold | 3e-05 | 2 | 2 | 9.552 |
| single_output | category_content | 1 | 55 | 1 | holdout | 2e-05 | 1 | 1 | 8.444 |
| one_positive | category_booking | 1 | 0 | 1 | fixed_config | 2e-05 | 10 | 2 | 2.973 |


mappingは十分な元正例を持つ複数出力、internet_thingsは少数正例の複数出力、contentは単一出力、bookingは正例1件。全件正例数と実際に学習した4件以下を混同しない。

bookingではCV/学習率選択を実行していない。共通設定epoch10を記録し、動作確認の実学習は2epochへ制限した。contentでは2lr試行の最良epochが1で、最終動作確認も1epoch。

### 11.3 最良epochでの開発指標（性能評価ではない）

| case | lr | Fold | 最良epoch | val_loss | Precision | Recall | micro-F1 | 最良重み復元 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sufficient_multi | 2e-05 | fold_01 | 2 | 0.690069228 | 0.009433962 | 1.000000000 | 0.018691589 | True | best_ratio_1_fold_01_lr_2e-05.pth |
| sufficient_multi | 3e-05 | fold_01 | 2 | 0.678047717 | 0.010416667 | 1.000000000 | 0.020618557 | True | メモリのみ |
| few_positive_multi | 2e-05 | fold_01 | 2 | 0.766923845 | 0.000000000 | 0.000000000 | 0.000000000 | True | best_ratio_1_fold_01_lr_2e-05.pth |
| few_positive_multi | 3e-05 | fold_01 | 2 | 0.758968651 | 0.000000000 | 0.000000000 | 0.000000000 | True | メモリのみ |
| single_output | 2e-05 | fold_01 | 1 | 0.656795681 | 0.000000000 | 0.000000000 | 0.000000000 | True | best_ratio_1_fold_01_lr_2e-05.pth |
| single_output | 3e-05 | fold_01 | 1 | 0.669226736 | 0.000000000 | 0.000000000 | 0.000000000 | True | メモリのみ |


これらは最大4件・1Fold・比率1だけの動作確認指標である。全比率の最良設定や研究用推薦性能として解釈しない。mappingの低Precision等を実モデルの性能低下の証拠として扱うこともできない。

### 11.4 実Train/Validation件数・有限Loss

| case | 学習区分 | 実Train | 実Validation | 実epoch | 各epochのTrain Loss |
| --- | --- | --- | --- | --- | --- |
| sufficient_multi | lr=2e-05 | 4 | 4 | 2 | 0.750831515; 0.722435147 |
| sufficient_multi | lr=3e-05 | 4 | 4 | 2 | 0.747573197; 0.728040159 |
| sufficient_multi | final_refit | 4 | 0 | 2 | 0.748990208; 0.718723893 |
| few_positive_multi | lr=2e-05 | 4 | 4 | 2 | 0.976116002; 0.833657205 |
| few_positive_multi | lr=3e-05 | 4 | 4 | 2 | 1.003801435; 0.831835628 |
| few_positive_multi | final_refit | 4 | 0 | 2 | 1.008415043; 0.829604983 |
| single_output | lr=2e-05 | 4 | 4 | 2 | 0.865532458; 0.573405519 |
| single_output | lr=3e-05 | 4 | 4 | 2 | 0.890430540; 0.553365588 |
| single_output | final_refit | 4 | 0 | 1 | 1.109152973 |
| one_positive | final_refit | 2 | 0 | 2 | 0.685490727; 0.484032184 |


すべて有限値。全件候補に対する全ラベル学習を完了したという意味ではない。

### 11.5 スモークの重み補正件数

| case | 学習区分 | 正例0出力 | 負例0出力 | 補正出力 |
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


同じ分類器を複数lrで繰り返した行を独立分類器数として合計しない。補正が必要だった実物BERTスモークの分類器はmappingとinternet_thingsの2個。

### 11.6 checkpoint復元・最終保存・既存推論との互換性

- 選択対象3caseの2lr試行では初期重みハッシュの一致と最良重み復元を確認した。
- contentではepoch2のLossが悪化し、epoch1の重みへ実際に復元した。単に最終epochを保存した例だけではない。
- スモークでは各caseの最初のlrの最良checkpointを保存し、2番目のlrはメモリで保存復元を確認した。bookingは選択なしで、Fold checkpointはない。
- 4caseすべてで最終重みを保存・再ロードし、照合入力のlogits完全一致を確認した。
- 既存推論ファイルからクラスASTだけを抽出し、strict loadと同一入力のlogits一致を確認した。既存推論モジュール全体の評価ループは実行していない。
- 最終モデルの初期重みハッシュもFold試行と同じBERT初期化条件であることを確認し、Fold学習済み重みの流用がないことを照合した。

保存は裸の`bert.*`/`classifier.*` state_dictと出力順の`api_labels.json`。単一出力のlogits形状は`[1,1]`。保存結果の`single_output_sigmoid_shape`というフィールドは、コード上はSigmoid後の値ではなく**logitsのshape**を保存しているため、本書ではその意味で読む。

互換性確認は同じ構造でのstrict loadと照合入力での一致であり、既存推論の全経路・探索ファイル名・GPU実行・全入力での推薦性能を検証した意味ではない。既存モデル探索パスは変更していないので、新モデルへ自動切替も行っていない。

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/training.py:81`（`legacy_inference_class`）、`scripts/hierarchical_api_selection/api_negative_sampling/training.py:202`（`run`）、`scripts/inference/mashup_dis_nosp_category_api.py:17`（`BertForMultilabelClassification`）、[smoke_results.json](results/hierarchical_api_selection/phase3_smoke_v1/smoke_results.json)、各caseの`training_results.json`。

## 12. 自動検証・単体テスト・今回の読み取り再照合

### 12.1 保存済みデータ検証

[checks/validation_results.json](data/hierarchical_api_selection/api_negative_sampling_v2/checks/validation_results.json)は`passed=true`、1,881,673条件passed、fixed_evaluation_used=false、group_overlaps=0、same_validation_across_ratios=true、same_seed_reproduced=true。準備段階の別記録`data_validation_results.json`も保持している。最終記録が準備記録を上書きしたものではない。

保存済み検証は、宣言入力/コードのハッシュ・既存ファイルメタデータ、全152分類器の同seed再構築、正例・負例教師、固定評価ID/本文除外、group非重複、同一Validation、比率包含、正例保持、group丸ごとの採用、Train支持数の重み計算、CVの非固定group一巡等を確認する。根拠：`scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py:257`（`validate`）。

### 12.2 保存済み単体テスト

`checks/unit_test_results.json`は19件、failure=0、error=0、skip=0。実装時の実行結果を参照し、今回の報告作成のために学習を伴うテストは再実行していない。

| 対象 | 確認内容 |
| --- | --- |
| 負例抽出 | 比率包含、同seed再現、group超過、不足、複製しないこと |
| 正負混在group | 正例と同groupの負例を同時採用 |
| 分割 | API補助group固定、MashupのみのValidation、少数Fold、Holdoutのgroup非重複 |
| 例外 | 正例1件の通常CV禁止とTrain保持、Mashup正例group不足の共通設定 |
| class weight | 通常比、正例0/負例0、上限clamp、空Train拒否 |
| 条件選択 | lrをLoss優先、比率をF1優先、同点、中央値epoch |
| 入力結果の異常 | Fold欠落・重複・NaNを拒否 |
| ファイル保護 | 既存出力への生成拒否、exclusiveファイル作成 |
| micro-F1 | 全負例かつ全非推薦の分母0を0とする |
| スモーク抽出 | 正例/負例があるとき両方を残す |
| ES復元 | 合成Loss列1.0,0.5,0.7,0.8,0.9でepoch5停止、epoch2重み復元 |
| 単一出力損失 | 両クラス/全正例/全負例で有限Loss・有限かつ非ゼロ勾配 |

ESのpatience到達までの確認は合成モデルで行った。実物BERTのスモーク上限は2epochであり、patience3による実物BERTの停止そのものを確認したとは述べない。

### 12.3 今回の報告作成時の再照合

集計日時：2026-10-16 19:14:07 JST。既存ソースを読み込み、新しいモデルを作成せずに照合した。保存済み検証件数と今回の再集計件数を混ぜない。

| 今回の照合対象 | 件数/結果 |
| --- | --- |
| 宣言入力SHA-256 | 875 |
| 新規コード/README SHA-256 | 12 |
| 正式データ成果物SHA-256 | 624 |
| スモーク成果物SHA-256 | 20 |
| 生成前ファイルメタデータ | 14766件、差分なし |
| 分類器別元group・API交差教師 | 600,701延べレコード |
| CSVとのTrain正負件数照合 | 1260 |
| Foldの比率間同一Validation照合 | 804 |
| FoldのTrain/Validation group非重複 | 804 |
| 独立した支持数再集計によるFold重み照合 | 3756 |
| 最終候補の全正例保持 | 456 |
| 最終再学習の初期値ハッシュ照合 | 4 |


新データ625ファイルのうち、最終検証JSON自身を除く624件に保存済みSHA-256がある。検証JSON自身の自己ハッシュが保存されているとは主張しない。今回の読み取り集計では合計3,145,195回のassertionが通ったが、ループ内の重複する性質確認を含む。これは保存済み1,881,673件の検証を再実行した件数でも、独立した単体テスト件数でもない。

### 12.4 既存ファイル保護の保証範囲

Phase 3開始時の既存14,138ファイル変更0件は`execution_evidence.json`に保存済み。v2生成直前の既存14,766ファイルも、今回保存メタデータと照合して差分0。差628件は初回v1生成の616ファイルと新規コード/README12ファイルが加わったもの。旧v1 READMEはv2生成後に新規追加されており、この14,766件の既存ファイル集合には含まれない。

保護対象のメタデータはsize/mtime_ns/mode/symlink。宣言入力・コードにはSHA-256もある。既存の全モデル重み内容をSHA-256した保証ではなく、全モデル内容の暗号学的同一性を別途立証したものではない。Gitの既存変更・未追跡成果物を削除・移動・resetしていない。

## 13. データリークの確認結果と残る評価上の制約

### 13.1 防いでいる混入

固定Validation100件/Test236件に一致するPhase 1の除外ID・正規化本文が、API正例/負例候補に含まれない。カテゴリ分類側もPhase 1除外済み集合を使う。負例の抽出前にgroupをTrain/Validationへ割り当て、同じ分類器の同じFoldでは同groupが両側へ入らない。

同名だけを理由に既存API説明文を削除する処理を追加していない。Phase 1の種別付きentity_idと正規化本文groupを維持する。候補外説明文からの全0教師や固定評価からの負例も作っていない。

### 13.2 本報告で追加集計したValidation正例ラベルの範囲

条件選択を行う97分類器の出力APIは合計848ラベル。そのうち**526ラベルは、その分類器の保存された全Validation集合の和集合にも正例がない**（影響51分類器）。残る55固定設定分類器の61ラベルは診断Validationが全負例のため、61ラベルすべてがValidation正例0。合計909ラベル中、検証側に正例を持つラベルは322、持たないものは587である。

これは本報告作成時の追加読み取り集計であり、既存分割を変更した結果ではない。CVでは全Fold Validationの和集合、Holdoutでは1回のValidationを対象にした数字で、単一Foldでの正例0数とも違う。原因には自己説明文をTrain固定すること、Mashup正例の少なさ、Holdoutの正例ラベルがTrain側にだけ残ることがある。

Trainの正例支持0がないことと、Validationで全出力の再現率を検証できることは別。micro-F1による選択はValidationに現れるラベルへ強く依存する。希少APIや検証正例0ラベルの品質をこの指標だけで保証しない。ラベル別支持数と未評価ラベルを後続実験にも残すことが望ましい。

### 13.3 残る制約

- 正規化本文/実体IDの既知重複を防ぐものであり、意味的近似・未知の別名・説明文中のAPI名など、すべての情報重複を完全検出したものではない。
- 分類器別にFoldが異なるので、全層のOOFモデルを組み合わせた独立推薦精度としてローカル指標を扱えない。
- 同一ValidationをES・lr・比率に反復使用し、条件選択バイアスが生じる。最終性能の確認とは区別する。
- 観測された未使用を負例とするため、実際には利用可能なAPIが負例に含まれる限界がある。
- API補助データの比率とpos_weightが連動するため、負例を増やした効果を件数だけで説明できない。
- 過去の既存モデルで固定評価が混入していた履歴や、過去のTest利用を、この再構成で消去・回復することはできない。

## 14. 本格学習の必要回数・計算量・容量

### 14.1 モデル学習回数

| 対象 | 条件選択対象ペア | 条件選択モデル回数 | 最終再学習回数 | 計 |
| --- | --- | --- | --- | --- |
| API | 213 | 1278 | 152 | 1430 |
| カテゴリ | 661 | 1322 | 653 | 1975 |
| 全体 | 874 | 2600 | 805 | 3405 |


API：213×3比率×2lr=1,278回、固定設定55個は比較しない。最終再学習は152個すべて1回ずつ。カテゴリ：661×2lr=1,322回と最終653回。API152個は16階層で共用し、16倍しない。カテゴリ共有候補による削減は適用していない。

### 14.2 最大epochからの算術上限

| 対象 | 条件選択 optimizer step上限 | 最終 optimizer step上限 | 条件選択 Train service-epoch | 条件選択 Validation service-epoch | 最終 Train service-epoch |
| --- | --- | --- | --- | --- | --- |
| API | 844800 | 201010 | 6591100 | 70313700 | 1593000 |
| カテゴリ | 9621200 | 6008500 | 76732400 | 19185300 | 47952850 |


batch8、通常最大50epoch。API最終は選択対象97個について最大Train条件（通常比率5）と50epoch、固定設定55個は比率1と10epochを仮定した上限。ESによる短縮は見込まない。全optimizer step上限は16,675,510。service-epochはレコードを1epoch処理する回数で、独立サービス数ではない。

これらは保存データの実件数と規定上限から計算した**見積もり**であり、実行ログの回数や所要時間ではない。GPU時間は未確認。32token・4件・CPUのスモークから512tokenの本格学習時間を外挿しない。実際のGPU/VRAM/実効batch/検証I/Oで1epochを測定する必要がある。

### 14.3 保存容量

今回のデータ成果物は149,980,381 bytes、スモーク成果物は3,066,484,779 bytes。スモークは4最終モデル＋3最良checkpointの計7個のpthを含む。コード/READMEは89,343 bytes。

既存見積もりの約438MB/モデルを使うと、最終805個だけで概算約353GB（10進単位）。さらに条件選択の2,600モデル分のcheckpointをすべて保持すると約1,139GBの追加概算となる。実際はヘッド次元・形式で変動する。optimizer状態等を含めない。checkpointの保持・整理方針を本格実験前に決める必要がある。

根拠：[training_cost_estimate.json](data/hierarchical_api_selection/api_negative_sampling_v2/reports/training_cost_estimate.json)、同md、`scripts/hierarchical_api_selection/api_negative_sampling/build_reports.py:main`。

## 15. 新規コード・成果物の一覧

### 15.1 コード

| 新規ファイル | 役割 |
| --- | --- |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | 全候補の教師生成、group分割、比率抽出、安全weight、検証 |
| [prepare_negative_samples.py](scripts/hierarchical_api_selection/api_negative_sampling/prepare_negative_samples.py) | 候補・分割・3比率を新規生成。既存出力を拒否 |
| [create_training_splits.py](scripts/hierarchical_api_selection/api_negative_sampling/create_training_splits.py) | 保存分割の同seed再構築を含む読み取り検証。再生成CLIではない |
| [validate_training_data.py](scripts/hierarchical_api_selection/api_negative_sampling/validate_training_data.py) | 読み取り検証、指定時だけ新規検証JSONを保存 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | BERTモデル、ES、重み復元、条件選択、最終再学習、互換性確認 |
| [train_api_classifier.py](scripts/hierarchical_api_selection/api_negative_sampling/train_api_classifier.py) | API学習エンジンのCLI入口 |
| [train_category_classifier.py](scripts/hierarchical_api_selection/api_negative_sampling/train_category_classifier.py) | 同エンジンのカテゴリ用入口 |
| [select_training_config.py](scripts/hierarchical_api_selection/api_negative_sampling/select_training_config.py) | 保存/入力JSONの条件選択。BERT学習を起動しない |
| [run_smoke_tests.py](scripts/hierarchical_api_selection/api_negative_sampling/run_smoke_tests.py) | 4caseの小規模CPU動作確認 |
| [build_reports.py](scripts/hierarchical_api_selection/api_negative_sampling/build_reports.py) | 件数・計算量・既存詳細報告の生成 |
| [tests/test_phase3.py](scripts/hierarchical_api_selection/api_negative_sampling/tests/test_phase3.py) | 19単体テスト |
| [README.md](scripts/hierarchical_api_selection/api_negative_sampling/README.md) | 設計と実行手順、v1/v2の扱い |


### 15.2 正式データ625ファイルの内訳

| 保存区分 | ファイル数 | 内容 |
| --- | --- | --- |
| manifests | 4 | source / negative_sampling / training_policy / classifier |
| splits/api_classification | 152 | 全候補レコード、分割前の正負、groupと分割 |
| negative_samples | 456 | 152分類器×3比率。各ファイルにFoldとfinal候補 |
| reports | 8 | CSV3、JSON2、Markdown3 |
| checks | 4 | ready検証、単体テスト、実行証拠、最終検証 |
| ルートREADME | 1 | 正式データ案内 |
| 計 | 625 | JSON618、CSV3、Markdown4 |


CSV行数（ヘッダ除く）：件数表1,260、分類器可否152、出力支持/weight6,483。件数表は `(実分割268＋最終候補152)×3比率=1,260` で整合する。新規コードはPython11＋README1の12ファイル、スモークはJSON13＋pth7の20ファイル。

### 15.3 管理・検証ファイルへのリンク

| ファイル | 内容 |
| --- | --- |
| [manifests/source_manifest.json](data/hierarchical_api_selection/api_negative_sampling_v2/manifests/source_manifest.json) | 入力/実装ハッシュ、生成前メタデータ、Git状態 |
| [manifests/training_policy.json](data/hierarchical_api_selection/api_negative_sampling_v2/manifests/training_policy.json) | 分割・例外・lr・epoch・損失・閾値の方針 |
| [manifests/classifier_manifest.json](data/hierarchical_api_selection/api_negative_sampling_v2/manifests/classifier_manifest.json) | 152分類器と3比率のパス/ハッシュ |
| [manifests/negative_sampling_manifest.json](data/hierarchical_api_selection/api_negative_sampling_v2/manifests/negative_sampling_manifest.json) | 負例元・条件別参照 |
| [reports/negative_sample_counts.csv](data/hierarchical_api_selection/api_negative_sampling_v2/reports/negative_sample_counts.csv) | 全Fold・全比率・finalの件数 |
| [reports/classifier_training_feasibility.csv](data/hierarchical_api_selection/api_negative_sampling_v2/reports/classifier_training_feasibility.csv) | 方式変更前後と例外理由 |
| [reports/train_label_weights.csv](data/hierarchical_api_selection/api_negative_sampling_v2/reports/train_label_weights.csv) | 出力ごとのTrain支持と重み |
| [reports/implementation_report.md](data/hierarchical_api_selection/api_negative_sampling_v2/reports/implementation_report.md) | Phase 3実装時に保存した詳細報告 |
| [checks/unit_test_results.json](data/hierarchical_api_selection/api_negative_sampling_v2/checks/unit_test_results.json) | 実行済み19単体テスト |
| [checks/data_validation_results.json](data/hierarchical_api_selection/api_negative_sampling_v2/checks/data_validation_results.json) | 学習前の準備検証 |
| [checks/execution_evidence.json](data/hierarchical_api_selection/api_negative_sampling_v2/checks/execution_evidence.json) | 環境、スモーク全成果物ハッシュ、既存保護証拠 |
| [checks/validation_results.json](data/hierarchical_api_selection/api_negative_sampling_v2/checks/validation_results.json) | 最終検証と624成果物ハッシュ |


本報告はリポジトリ直下へ新規作成し、検証済みデータディレクトリの625ファイルには追加しない。過去の実装報告・manifestを更新しない。

## 16. 実行コマンドと再確認手順

### 16.1 実装時の操作内容

入力・既存成果物を読み取り確認し、新規コードを作成、初回v1生成、単一出力対象の修正後にv2へ新規生成、19テスト記録、学習前データ検証、CPU小規模学習4例、報告/計算量生成、実行証拠保存、最終データ検証を行った。`README.md`には同じ機能を利用するコマンドを保存した。

小規模学習の起動コマンドは次の形式。**既存の出力先には再実行できない。**

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/nakagawa/venvs/swallow-py312/bin/python \
scripts/hierarchical_api_selection/api_negative_sampling/run_smoke_tests.py \
--output results/hierarchical_api_selection/phase3_smoke_v1
```

### 16.2 現状を読み取りだけで確認するコマンド

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
scripts/hierarchical_api_selection/api_negative_sampling/validate_training_data.py
```

デフォルトは正式v2の読み取り検証。`--write-report`/`--readiness-report`を付けない。学習を実行するコマンドではない。今回の報告作成では保存済み検証の確認と別の読み取り集計を実行した。

### 16.3 将来の再生成・本格学習

再生成する場合はprepareの`--output`で新しいパスを指定し、検証・学習側でも同じパスを指定する。既存v1/v2を削除・上書きしない。スモークも新規の出力ディレクトリを指定する。CUDAで決定論的学習を使う場合、README指定の`CUBLAS_WORKSPACE_CONFIG=:4096:8`等の環境条件を確認する。

本格API選択は3比率×2lr×全保存Foldを比較する実装であり、`--smoke --ratio 1`の結果を本格設定として採用しない。今回の保存モデルもスモーク用であり、既存推薦モデルへ置き換えない。

生成用`build_reports.py`はexclusive出力を使い、既存の報告書へ無条件再生成できない。また実装時報告には最終校正の追記があるため、テンプレート出力だけで過去報告をbyte単位で再現すると主張しない。データ/分割の同seed再現と、時刻・Git状態・報告文を含む全ファイルのbyte単位再現は別である。

## 17. 未実施・未検証と今後の判断事項

| 項目 | 状態 | 次に必要な判断/確認 |
| --- | --- | --- |
| 全152 API分類器の本格学習 | 未実施 | 3比率×2lr×全Foldで比較し、選択後に全開発集合で最終学習 |
| 全653カテゴリの本格学習 | 未実施 | 既存661ペアを使う。実物BERTのカテゴリ経路確認も未実施 |
| 全比率の実物BERT比較 | 未実施 | 実行は比率1のみ。比率選択規則は合成テストで確認 |
| 全Fold実物BERT学習 | 未実施 | スモークは最初の1Foldのみ |
| 共通設定55個の最適性 | 未確認 | 比率1/lr2e-5/epoch10の事前固定でよいか、他分類器から移すか |
| 正例1件10個のMashup再現率 | 検証不能 | 現学習集合にMashup正例なし。負例追加だけでは解決しない |
| 検証正例なし587APIラベル | 追加集計で確認 | ラベル別未評価を追跡し、micro-F1だけで品質保証しない |
| GPU所要時間/VRAM | 未確認 | 512tokenの代表1epoch計測とGPU条件確認 |
| checkpoint保持容量 | 未確定 | 全Fold最良重みと最終重みの保持/整理方針 |
| 既存推論への自動切替 | 未実施 | 探索パス・名前・ラベル順を明示して後続で接続 |
| 固定Validation100/Test236の評価 | 未実施 | モデル条件選択へ流用せず、後続の階層選定/最終評価で利用 |
| Eに対する評価プロット・平均集計 | 未実施 | 集計単位や階層比較手順は別途確認 |
| 学習の複数回実行間の完全再現 | 未検証 | 同初期重み・データ再現は確認。別環境でのbitwise再現とは別 |


判断が必要なものを既に確定・検証した事実として扱わない。現状で実装済みの既定値はmanifestに残し、変更する場合は新しいpolicy/成果物版で実験条件を追跡する。

## 18. 完了条件との対応

| 依頼上の完了条件 | 結果と限界 |
| --- | --- |
| 既存データ/コード/モデル/評価結果の保持 | 保存済み保護記録と今回照合で差分なし。全モデル内容のSHA保証ではない |
| 909 API/152カテゴリ/16階層の維持 | Phase 1の定義と入力ハッシュを保持 |
| 固定評価から負例を生成しない | ID/本文非混入とMashup3,946限定を確認 |
| group単位のTrain/Validation分離 | 保存された全268ペア・3比率で確認 |
| 3比率で同じValidation | 確認。実BERT全比率比較は未実行 |
| 正例1件を学習対象に保持 | 10個を共通設定で保持。bookingの実物BERTを確認 |
| 単一出力で推薦可否を学習 | 正負教師とSigmoid閾値。content/bookingで動作確認 |
| 重み0/NaN/Infの回避 | 全件weight有限。合成の片側支持0とスモーク部分集合で補正確認 |
| lr/epoch選択の実装と追跡 | 同点規則・中央値・最良復元を実装。3caseで2lr動作確認 |
| 本格実験を行わない | 4case、最大4件、1Fold、2epoch、CPUのみ |
| 由来/条件/保存先の追跡 | manifest、CSV、case別JSON、入力/成果物ハッシュを保存 |


## 付録A. 全152 API分類器の全件候補・方式一覧

負例1/3/5は最終再学習候補。各値のリンク先にFoldと最終候補のentity_idがある。K=0でもHoldout/診断の実分割は1ペア。Val正例0ラベルは当該分類器の全Validation和集合での数。

| 分類器 | 出力数 | 正例M | 正例API | 正例計 | 負例候補 | 旧方式 | 新方式 | K | 実ペア | 負例1 | 負例3 | 負例5 | Val正例0ラベル | 例外理由 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [category_accounting](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_accounting.json) | 2 | 1 | 2 | 3 | 3945 | kfold | fixed_config | 0 | 1 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_accounting.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_accounting.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_accounting.json) | 2 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_addresses](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_addresses.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_addresses.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_addresses.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_addresses.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_advertising](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_advertising.json) | 11 | 55 | 11 | 66 | 3891 | holdout | holdout | 0 | 1 | [66](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_advertising.json) | [198](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_advertising.json) | [330](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_advertising.json) | 9 | — |
| [category_air_travel](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_air_travel.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_air_travel.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_air_travel.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_air_travel.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_analytics](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_analytics.json) | 8 | 25 | 8 | 33 | 3921 | holdout | holdout | 0 | 1 | [33](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_analytics.json) | [99](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_analytics.json) | [165](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_analytics.json) | 7 | — |
| [category_animals](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_animals.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_animals.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_animals.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_animals.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_api](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_api.json) | 2 | 5 | 2 | 7 | 3941 | kfold | kfold | 5 | 5 | [7](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_api.json) | [21](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_api.json) | [35](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_api.json) | 0 | — |
| [category_application_development](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_application_development.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_application_development.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_application_development.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_application_development.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_applications](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_applications.json) | 3 | 3 | 3 | 6 | 3943 | kfold | kfold | 3 | 3 | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_applications.json) | [18](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_applications.json) | [30](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_applications.json) | 0 | — |
| [category_astronomy](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_astronomy.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_astronomy.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_astronomy.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_astronomy.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_authentication](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_authentication.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_authentication.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_authentication.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_authentication.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_auto](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_auto.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_auto.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_auto.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_auto.json) | 0 | — |
| [category_backend](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_backend.json) | 5 | 7 | 5 | 12 | 3939 | kfold | kfold | 5 | 5 | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_backend.json) | [36](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_backend.json) | [60](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_backend.json) | 0 | — |
| [category_barcodes](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_barcodes.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_barcodes.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_barcodes.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_barcodes.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_big_data](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_big_data.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_big_data.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_big_data.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_big_data.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_biometrics](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_biometrics.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_biometrics.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_biometrics.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_biometrics.json) | 0 | — |
| [category_bitcoin](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_bitcoin.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_bitcoin.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_bitcoin.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_bitcoin.json) | 0 | — |
| [category_blockchain](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_blockchain.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_blockchain.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_blockchain.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_blockchain.json) | 0 | — |
| [category_blogging](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_blogging.json) | 9 | 28 | 9 | 37 | 3918 | holdout | holdout | 0 | 1 | [37](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_blogging.json) | [111](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_blogging.json) | [185](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_blogging.json) | 5 | — |
| [category_booking](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_booking.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_booking.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_booking.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_booking.json) | 1 | one_positive_record |
| [category_bookmarks](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_bookmarks.json) | 5 | 47 | 5 | 52 | 3899 | holdout | holdout | 0 | 1 | [52](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_bookmarks.json) | [156](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_bookmarks.json) | [260](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_bookmarks.json) | 3 | — |
| [category_books](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_books.json) | 1 | 7 | 1 | 8 | 3939 | kfold | kfold | 5 | 5 | [8](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_books.json) | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_books.json) | [40](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_books.json) | 0 | — |
| [category_bots](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_bots.json) | 2 | 3 | 2 | 5 | 3943 | kfold | kfold | 3 | 3 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_bots.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_bots.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_bots.json) | 0 | — |
| [category_business](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_business.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_business.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_business.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_business.json) | 0 | — |
| [category_calendars](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_calendars.json) | 4 | 17 | 4 | 21 | 3929 | holdout | holdout | 0 | 1 | [21](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_calendars.json) | [63](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_calendars.json) | [105](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_calendars.json) | 3 | — |
| [category_chat](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_chat.json) | 7 | 12 | 7 | 19 | 3934 | kfold | kfold | 5 | 5 | [19](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_chat.json) | [57](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_chat.json) | [95](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_chat.json) | 0 | — |
| [category_classification](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_classification.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_classification.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_classification.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_classification.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_cloud](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_cloud.json) | 4 | 37 | 4 | 41 | 3909 | holdout | holdout | 0 | 1 | [41](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_cloud.json) | [123](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_cloud.json) | [205](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_cloud.json) | 2 | — |
| [category_collaboration](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_collaboration.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_collaboration.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_collaboration.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_collaboration.json) | 0 | — |
| [category_content](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_content.json) | 1 | 55 | 1 | 56 | 3891 | holdout | holdout | 0 | 1 | [56](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_content.json) | [168](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_content.json) | [280](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_content.json) | 0 | — |
| [category_coupons](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_coupons.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_coupons.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_coupons.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_coupons.json) | 0 | — |
| [category_covid_19](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_covid_19.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_covid_19.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_covid_19.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_covid_19.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_cryptocurrency](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_cryptocurrency.json) | 3 | 2 | 3 | 5 | 3944 | kfold | kfold | 2 | 2 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_cryptocurrency.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_cryptocurrency.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_cryptocurrency.json) | 0 | — |
| [category_cycling](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_cycling.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_cycling.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_cycling.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_cycling.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_data](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_data.json) | 2 | 3 | 2 | 5 | 3943 | kfold | kfold | 3 | 3 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_data.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_data.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_data.json) | 0 | — |
| [category_data_as_a_service](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_data_as_a_service.json) | 4 | 4 | 4 | 8 | 3942 | kfold | kfold | 4 | 4 | [8](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_data_as_a_service.json) | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_data_as_a_service.json) | [40](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_data_as_a_service.json) | 0 | — |
| [category_data_mining](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_data_mining.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_data_mining.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_data_mining.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_data_mining.json) | 0 | — |
| [category_database](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_database.json) | 12 | 42 | 12 | 54 | 3904 | holdout | holdout | 0 | 1 | [54](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_database.json) | [162](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_database.json) | [270](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_database.json) | 8 | — |
| [category_design](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_design.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_design.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_design.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_design.json) | 0 | — |
| [category_developers](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_developers.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_developers.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_developers.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_developers.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_documents](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_documents.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_documents.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_documents.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_documents.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_domains](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_domains.json) | 4 | 1 | 4 | 5 | 3945 | kfold | fixed_config | 0 | 1 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_domains.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_domains.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_domains.json) | 4 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_ecommerce](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_ecommerce.json) | 54 | 290 | 54 | 344 | 3656 | holdout | holdout | 0 | 1 | [344](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_ecommerce.json) | [1032](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_ecommerce.json) | [1720](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_ecommerce.json) | 40 | — |
| [category_education](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_education.json) | 12 | 26 | 12 | 38 | 3920 | holdout | holdout | 0 | 1 | [38](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_education.json) | [114](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_education.json) | [190](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_education.json) | 11 | — |
| [category_electronic_signature](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_electronic_signature.json) | 1 | 21 | 1 | 22 | 3925 | holdout | holdout | 0 | 1 | [22](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_electronic_signature.json) | [66](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_electronic_signature.json) | [110](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_electronic_signature.json) | 0 | — |
| [category_email](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_email.json) | 14 | 44 | 14 | 58 | 3902 | holdout | holdout | 0 | 1 | [58](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_email.json) | [174](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_email.json) | [290](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_email.json) | 7 | — |
| [category_enterprise](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_enterprise.json) | 17 | 14 | 17 | 31 | 3932 | holdout | holdout | 0 | 1 | [31](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_enterprise.json) | [93](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_enterprise.json) | [155](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_enterprise.json) | 15 | — |
| [category_entertainment](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_entertainment.json) | 5 | 14 | 5 | 19 | 3932 | kfold | kfold | 5 | 5 | [19](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_entertainment.json) | [57](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_entertainment.json) | [95](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_entertainment.json) | 1 | — |
| [category_environment](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_environment.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_environment.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_environment.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_environment.json) | 0 | — |
| [category_esports](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_esports.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_esports.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_esports.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_esports.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_events](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_events.json) | 6 | 19 | 6 | 25 | 3927 | holdout | holdout | 0 | 1 | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_events.json) | [75](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_events.json) | [125](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_events.json) | 5 | — |
| [category_fax](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_fax.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_fax.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_fax.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_fax.json) | 0 | — |
| [category_feedback](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_feedback.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_feedback.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_feedback.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_feedback.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_feeds](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_feeds.json) | 4 | 14 | 4 | 18 | 3932 | kfold | kfold | 5 | 5 | [18](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_feeds.json) | [54](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_feeds.json) | [90](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_feeds.json) | 0 | — |
| [category_file_sharing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_file_sharing.json) | 6 | 5 | 6 | 11 | 3941 | kfold | kfold | 5 | 5 | [11](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_file_sharing.json) | [33](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_file_sharing.json) | [55](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_file_sharing.json) | 1 | — |
| [category_financial](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_financial.json) | 20 | 35 | 20 | 55 | 3911 | holdout | holdout | 0 | 1 | [55](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_financial.json) | [165](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_financial.json) | [275](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_financial.json) | 17 | — |
| [category_fitness](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_fitness.json) | 2 | 18 | 2 | 20 | 3928 | holdout | holdout | 0 | 1 | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_fitness.json) | [60](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_fitness.json) | [100](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_fitness.json) | 1 | — |
| [category_food](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_food.json) | 4 | 6 | 4 | 10 | 3940 | kfold | kfold | 5 | 5 | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_food.json) | [30](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_food.json) | [50](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_food.json) | 2 | — |
| [category_forms](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_forms.json) | 1 | 4 | 1 | 5 | 3942 | kfold | kfold | 4 | 4 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_forms.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_forms.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_forms.json) | 0 | — |
| [category_framework](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_framework.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_framework.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_framework.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_framework.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_gambling](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_gambling.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_gambling.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_gambling.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_gambling.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_games](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_games.json) | 6 | 13 | 6 | 19 | 3933 | kfold | kfold | 5 | 5 | [19](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_games.json) | [57](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_games.json) | [95](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_games.json) | 0 | — |
| [category_geography](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_geography.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_geography.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_geography.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_geography.json) | 0 | — |
| [category_government](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_government.json) | 27 | 24 | 27 | 51 | 3922 | holdout | holdout | 0 | 1 | [51](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_government.json) | [153](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_government.json) | [255](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_government.json) | 25 | — |
| [category_hardware](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_hardware.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_hardware.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_hardware.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_hardware.json) | 1 | one_positive_record |
| [category_health](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_health.json) | 2 | 3 | 2 | 5 | 3943 | kfold | kfold | 3 | 3 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_health.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_health.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_health.json) | 0 | — |
| [category_healthcare](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_healthcare.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_healthcare.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_healthcare.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_healthcare.json) | 0 | — |
| [category_home_automation](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_home_automation.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_home_automation.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_home_automation.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_home_automation.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_hosting](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_hosting.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_hosting.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_hosting.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_hosting.json) | 0 | — |
| [category_identity](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_identity.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_identity.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_identity.json) | [11](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_identity.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_images](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_images.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_images.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_images.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_images.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_intelligence](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_intelligence.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_intelligence.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_intelligence.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_intelligence.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_internet_things](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_internet_things.json) | 5 | 3 | 5 | 8 | 3943 | kfold | kfold | 3 | 3 | [8](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_internet_things.json) | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_internet_things.json) | [40](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_internet_things.json) | 2 | — |
| [category_invoicing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_invoicing.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_invoicing.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_invoicing.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_invoicing.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_jobs](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_jobs.json) | 6 | 30 | 6 | 36 | 3916 | holdout | holdout | 0 | 1 | [36](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_jobs.json) | [108](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_jobs.json) | [180](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_jobs.json) | 5 | — |
| [category_language](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_language.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_language.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_language.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_language.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_learning_management_systems](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_learning_management_systems.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_learning_management_systems.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_learning_management_systems.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_learning_management_systems.json) | 1 | one_positive_record |
| [category_library](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_library.json) | 1 | 3 | 1 | 4 | 3943 | kfold | kfold | 3 | 3 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_library.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_library.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_library.json) | 0 | — |
| [category_location](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_location.json) | 2 | 3 | 2 | 5 | 3943 | kfold | kfold | 3 | 3 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_location.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_location.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_location.json) | 0 | — |
| [category_machine_learning](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_machine_learning.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_machine_learning.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_machine_learning.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_machine_learning.json) | 1 | one_positive_record |
| [category_mail](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_mail.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_mail.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_mail.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_mail.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_mapping](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_mapping.json) | 52 | 1668 | 52 | 1720 | 2278 | holdout | holdout | 0 | 1 | [1720](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_mapping.json) | [2278](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_mapping.json) | [2278](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_mapping.json) | 27 | — |
| [category_marketing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_marketing.json) | 2 | 4 | 2 | 6 | 3942 | kfold | kfold | 4 | 4 | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_marketing.json) | [18](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_marketing.json) | [30](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_marketing.json) | 0 | — |
| [category_media](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_media.json) | 8 | 20 | 8 | 28 | 3926 | holdout | holdout | 0 | 1 | [28](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_media.json) | [84](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_media.json) | [140](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_media.json) | 7 | — |
| [category_medical](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_medical.json) | 5 | 3 | 5 | 8 | 3943 | kfold | kfold | 3 | 3 | [8](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_medical.json) | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_medical.json) | [40](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_medical.json) | 1 | — |
| [category_medicine](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_medicine.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_medicine.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_medicine.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_medicine.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_merchants](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_merchants.json) | 3 | 2 | 3 | 5 | 3944 | kfold | kfold | 2 | 2 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_merchants.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_merchants.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_merchants.json) | 0 | — |
| [category_messaging](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_messaging.json) | 22 | 179 | 22 | 201 | 3767 | holdout | holdout | 0 | 1 | [201](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_messaging.json) | [603](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_messaging.json) | [1005](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_messaging.json) | 16 | — |
| [category_monitoring](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_monitoring.json) | 1 | 4 | 1 | 5 | 3942 | kfold | kfold | 4 | 4 | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_monitoring.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_monitoring.json) | [25](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_monitoring.json) | 0 | — |
| [category_movies](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_movies.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_movies.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_movies.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_movies.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_music](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_music.json) | 29 | 115 | 29 | 144 | 3831 | holdout | holdout | 0 | 1 | [144](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_music.json) | [432](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_music.json) | [720](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_music.json) | 19 | — |
| [category_natural_language_processing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_natural_language_processing.json) | 2 | 0 | 2 | 2 | 3946 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_natural_language_processing.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_natural_language_processing.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_natural_language_processing.json) | 2 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_news_services](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_news_services.json) | 12 | 33 | 12 | 45 | 3913 | holdout | holdout | 0 | 1 | [45](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_news_services.json) | [135](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_news_services.json) | [225](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_news_services.json) | 6 | — |
| [category_nutrition](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_nutrition.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_nutrition.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_nutrition.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_nutrition.json) | 0 | — |
| [category_office](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_office.json) | 11 | 16 | 11 | 27 | 3930 | holdout | holdout | 0 | 1 | [27](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_office.json) | [81](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_office.json) | [135](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_office.json) | 9 | — |
| [category_other](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_other.json) | 25 | 75 | 25 | 100 | 3871 | holdout | holdout | 0 | 1 | [100](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_other.json) | [300](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_other.json) | [500](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_other.json) | 16 | — |
| [category_parsing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_parsing.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_parsing.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_parsing.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_parsing.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_payments](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_payments.json) | 15 | 33 | 15 | 48 | 3913 | holdout | holdout | 0 | 1 | [48](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_payments.json) | [144](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_payments.json) | [240](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_payments.json) | 13 | — |
| [category_pdf](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_pdf.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_pdf.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_pdf.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_pdf.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_performance](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_performance.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_performance.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_performance.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_performance.json) | 0 | — |
| [category_personal_information_management](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_personal_information_management.json) | 7 | 17 | 7 | 24 | 3929 | holdout | holdout | 0 | 1 | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_personal_information_management.json) | [72](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_personal_information_management.json) | [120](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_personal_information_management.json) | 6 | — |
| [category_photos](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_photos.json) | 26 | 359 | 26 | 385 | 3587 | holdout | holdout | 0 | 1 | [385](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_photos.json) | [1155](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_photos.json) | [1925](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_photos.json) | 14 | — |
| [category_platform_as_a_service](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_platform_as_a_service.json) | 1 | 2 | 1 | 3 | 3944 | kfold | kfold | 2 | 2 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_platform_as_a_service.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_platform_as_a_service.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_platform_as_a_service.json) | 0 | — |
| [category_politics](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_politics.json) | 1 | 3 | 1 | 4 | 3943 | kfold | kfold | 3 | 3 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_politics.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_politics.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_politics.json) | 0 | — |
| [category_predictions](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_predictions.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_predictions.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_predictions.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_predictions.json) | 1 | one_positive_record |
| [category_printing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_printing.json) | 2 | 2 | 2 | 4 | 3944 | kfold | kfold | 2 | 2 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_printing.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_printing.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_printing.json) | 0 | — |
| [category_products](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_products.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_products.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_products.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_products.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_project_management](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_project_management.json) | 11 | 19 | 11 | 30 | 3927 | holdout | holdout | 0 | 1 | [30](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_project_management.json) | [90](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_project_management.json) | [150](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_project_management.json) | 10 | — |
| [category_purchasing](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_purchasing.json) | 1 | 3 | 1 | 4 | 3943 | kfold | kfold | 3 | 3 | [4](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_purchasing.json) | [12](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_purchasing.json) | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_purchasing.json) | 0 | — |
| [category_qa](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_qa.json) | 2 | 5 | 2 | 7 | 3941 | kfold | kfold | 5 | 5 | [7](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_qa.json) | [21](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_qa.json) | [35](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_qa.json) | 0 | — |
| [category_random](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_random.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_random.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_random.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_random.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_real_estate](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_real_estate.json) | 7 | 17 | 7 | 24 | 3929 | holdout | holdout | 0 | 1 | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_real_estate.json) | [72](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_real_estate.json) | [120](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_real_estate.json) | 5 | — |
| [category_recognition](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_recognition.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_recognition.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_recognition.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_recognition.json) | 1 | one_positive_record |
| [category_recommendations](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_recommendations.json) | 5 | 35 | 5 | 40 | 3911 | holdout | holdout | 0 | 1 | [40](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_recommendations.json) | [120](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_recommendations.json) | [200](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_recommendations.json) | 3 | — |
| [category_reference](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_reference.json) | 39 | 127 | 39 | 166 | 3819 | holdout | holdout | 0 | 1 | [166](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_reference.json) | [498](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_reference.json) | [830](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_reference.json) | 30 | — |
| [category_reporting](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_reporting.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_reporting.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_reporting.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_reporting.json) | 1 | one_positive_record |
| [category_rewards](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_rewards.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_rewards.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_rewards.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_rewards.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_ride_share](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_ride_share.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_ride_share.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_ride_share.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_ride_share.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_sales](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_sales.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_sales.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_sales.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_sales.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_science](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_science.json) | 8 | 15 | 8 | 23 | 3931 | holdout | holdout | 0 | 1 | [23](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_science.json) | [69](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_science.json) | [115](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_science.json) | 6 | — |
| [category_screenshots](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_screenshots.json) | 1 | 5 | 1 | 6 | 3941 | kfold | kfold | 5 | 5 | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_screenshots.json) | [18](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_screenshots.json) | [30](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_screenshots.json) | 0 | — |
| [category_search](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_search.json) | 35 | 119 | 35 | 154 | 3827 | holdout | holdout | 0 | 1 | [154](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_search.json) | [462](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_search.json) | [770](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_search.json) | 23 | — |
| [category_security](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_security.json) | 8 | 12 | 8 | 20 | 3934 | holdout | holdout | 0 | 1 | [20](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_security.json) | [60](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_security.json) | [100](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_security.json) | 7 | — |
| [category_semantics](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_semantics.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_semantics.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_semantics.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_semantics.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_sentiment](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_sentiment.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_sentiment.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_sentiment.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_sentiment.json) | 1 | one_positive_record |
| [category_seo](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_seo.json) | 5 | 5 | 5 | 10 | 3941 | kfold | kfold | 5 | 5 | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_seo.json) | [30](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_seo.json) | [50](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_seo.json) | 2 | — |
| [category_shipping](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_shipping.json) | 5 | 14 | 5 | 19 | 3932 | kfold | kfold | 5 | 5 | [19](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_shipping.json) | [57](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_shipping.json) | [95](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_shipping.json) | 0 | — |
| [category_social](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_social.json) | 48 | 670 | 48 | 718 | 3276 | holdout | holdout | 0 | 1 | [718](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_social.json) | [2154](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_social.json) | [3276](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_social.json) | 33 | — |
| [category_spam](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_spam.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_spam.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_spam.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_spam.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_sports](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_sports.json) | 9 | 15 | 9 | 24 | 3931 | holdout | holdout | 0 | 1 | [24](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_sports.json) | [72](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_sports.json) | [120](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_sports.json) | 7 | — |
| [category_spreadsheets](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_spreadsheets.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_spreadsheets.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_spreadsheets.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_spreadsheets.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_standards](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_standards.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_standards.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_standards.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_standards.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_statistics](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_statistics.json) | 2 | 7 | 2 | 9 | 3939 | kfold | kfold | 5 | 5 | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_statistics.json) | [27](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_statistics.json) | [45](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_statistics.json) | 0 | — |
| [category_stocks](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_stocks.json) | 2 | 1 | 2 | 3 | 3945 | kfold | fixed_config | 0 | 1 | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_stocks.json) | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_stocks.json) | [15](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_stocks.json) | 2 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_storage](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_storage.json) | 7 | 51 | 7 | 58 | 3895 | holdout | holdout | 0 | 1 | [58](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_storage.json) | [174](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_storage.json) | [290](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_storage.json) | 4 | — |
| [category_streaming](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_streaming.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_streaming.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_streaming.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_streaming.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_support](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_support.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_support.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_support.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_support.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_tagging](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_tagging.json) | 1 | 6 | 1 | 7 | 3940 | kfold | kfold | 5 | 5 | [7](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_tagging.json) | [21](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_tagging.json) | [35](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_tagging.json) | 0 | — |
| [category_telephony](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_telephony.json) | 18 | 327 | 18 | 345 | 3619 | holdout | holdout | 0 | 1 | [345](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_telephony.json) | [1035](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_telephony.json) | [1725](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_telephony.json) | 11 | — |
| [category_text](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_text.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_text.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_text.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_text.json) | 1 | one_positive_record |
| [category_tools](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_tools.json) | 38 | 113 | 38 | 151 | 3833 | holdout | holdout | 0 | 1 | [151](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_tools.json) | [453](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_tools.json) | [755](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_tools.json) | 31 | — |
| [category_translation](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_translation.json) | 2 | 19 | 2 | 21 | 3927 | holdout | holdout | 0 | 1 | [21](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_translation.json) | [63](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_translation.json) | [105](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_translation.json) | 1 | — |
| [category_transportation](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_transportation.json) | 10 | 16 | 10 | 26 | 3930 | holdout | holdout | 0 | 1 | [26](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_transportation.json) | [78](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_transportation.json) | [130](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_transportation.json) | 8 | — |
| [category_travel](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_travel.json) | 24 | 43 | 24 | 67 | 3903 | holdout | holdout | 0 | 1 | [67](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_travel.json) | [201](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_travel.json) | [335](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_travel.json) | 18 | — |
| [category_url_shortener](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_url_shortener.json) | 2 | 21 | 2 | 23 | 3925 | holdout | holdout | 0 | 1 | [23](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_url_shortener.json) | [69](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_url_shortener.json) | [115](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_url_shortener.json) | 1 | — |
| [category_video](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_video.json) | 16 | 293 | 16 | 309 | 3653 | holdout | holdout | 0 | 1 | [309](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_video.json) | [927](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_video.json) | [1545](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_video.json) | 13 | — |
| [category_voice](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_voice.json) | 1 | 0 | 1 | 1 | 3946 | unavailable | fixed_config | 0 | 1 | [1](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_voice.json) | [3](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_voice.json) | [5](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_voice.json) | 1 | one_positive_record |
| [category_weather](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_weather.json) | 11 | 29 | 11 | 40 | 3917 | holdout | holdout | 0 | 1 | [40](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_weather.json) | [120](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_weather.json) | [200](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_weather.json) | 8 | — |
| [category_web_site_management](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_web_site_management.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_web_site_management.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_web_site_management.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_web_site_management.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_wi_fi](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_wi_fi.json) | 1 | 1 | 1 | 2 | 3945 | kfold | fixed_config | 0 | 1 | [2](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_wi_fi.json) | [6](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_wi_fi.json) | [10](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_wi_fi.json) | 1 | fewer_than_two_nonanchored_mashup_positive_groups |
| [category_widgets](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_widgets.json) | 4 | 61 | 4 | 65 | 3885 | holdout | holdout | 0 | 1 | [65](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_widgets.json) | [195](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_widgets.json) | [325](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_widgets.json) | 2 | — |
| [category_wiki](data/hierarchical_api_selection/api_negative_sampling_v2/splits/api_classification/category_wiki.json) | 3 | 6 | 3 | 9 | 3940 | kfold | kfold | 5 | 5 | [9](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_1/category_wiki.json) | [27](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_3/category_wiki.json) | [45](data/hierarchical_api_selection/api_negative_sampling_v2/negative_samples/negative_ratio_5/category_wiki.json) | 0 | — |

## 付録B. 全268分割ペアの実件数

正例数とValidationは3条件で共通。負例1/3/5は当該FoldのTrain負例。fixed_configの行は診断用であり、学習条件選択を実行するペアではない。ここに最終再学習候補152件の表は合算しない。MはMashup、APIは自己説明文。

| 分類器 | ペア | 用途 | 正例M | 正例API | 正例計 | 負例1 | Train1 | 負例3 | Train3 | 負例5 | Train5 | Val正例 | Val負例 | Val計 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category_accounting | fold_01 | 診断のみ | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 0 | 789 | 789 |
| category_addresses | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_advertising | fold_01 | 条件選択 | 44 | 11 | 55 | 55 | 110 | 165 | 220 | 275 | 330 | 11 | 777 | 788 |
| category_air_travel | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_analytics | fold_01 | 条件選択 | 20 | 8 | 28 | 28 | 56 | 84 | 112 | 140 | 168 | 5 | 783 | 788 |
| category_animals | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_api | fold_01 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 789 | 790 |
| category_api | fold_02 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 787 | 788 |
| category_api | fold_03 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 789 | 790 |
| category_api | fold_04 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 788 | 789 |
| category_api | fold_05 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 788 | 789 |
| category_application_development | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_applications | fold_01 | 条件選択 | 2 | 3 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 1315 | 1316 |
| category_applications | fold_02 | 条件選択 | 2 | 3 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 1314 | 1315 |
| category_applications | fold_03 | 条件選択 | 2 | 3 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 1314 | 1315 |
| category_astronomy | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_authentication | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_auto | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_auto | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_backend | fold_01 | 条件選択 | 5 | 5 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 2 | 788 | 790 |
| category_backend | fold_02 | 条件選択 | 5 | 5 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 2 | 787 | 789 |
| category_backend | fold_03 | 条件選択 | 6 | 5 | 11 | 11 | 22 | 33 | 44 | 55 | 66 | 1 | 788 | 789 |
| category_backend | fold_04 | 条件選択 | 6 | 5 | 11 | 11 | 22 | 33 | 44 | 55 | 66 | 1 | 788 | 789 |
| category_backend | fold_05 | 条件選択 | 6 | 5 | 11 | 11 | 22 | 33 | 44 | 55 | 66 | 1 | 788 | 789 |
| category_barcodes | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_big_data | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_biometrics | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1973 | 1974 |
| category_biometrics | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1971 | 1972 |
| category_bitcoin | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1974 | 1975 |
| category_bitcoin | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1970 | 1971 |
| category_blockchain | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_blockchain | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_blogging | fold_01 | 条件選択 | 22 | 9 | 31 | 31 | 62 | 93 | 124 | 155 | 186 | 6 | 782 | 788 |
| category_booking | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_bookmarks | fold_01 | 条件選択 | 38 | 5 | 43 | 43 | 86 | 129 | 172 | 215 | 258 | 9 | 779 | 788 |
| category_books | fold_01 | 条件選択 | 5 | 1 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 2 | 787 | 789 |
| category_books | fold_02 | 条件選択 | 5 | 1 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 2 | 788 | 790 |
| category_books | fold_03 | 条件選択 | 6 | 1 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 790 | 791 |
| category_books | fold_04 | 条件選択 | 6 | 1 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 787 | 788 |
| category_books | fold_05 | 条件選択 | 6 | 1 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 787 | 788 |
| category_bots | fold_01 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1315 | 1316 |
| category_bots | fold_02 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1314 | 1315 |
| category_bots | fold_03 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1314 | 1315 |
| category_business | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_business | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_calendars | fold_01 | 条件選択 | 14 | 4 | 18 | 18 | 36 | 54 | 72 | 90 | 108 | 3 | 786 | 789 |
| category_chat | fold_01 | 条件選択 | 9 | 7 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 786 | 789 |
| category_chat | fold_02 | 条件選択 | 9 | 7 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 787 | 790 |
| category_chat | fold_03 | 条件選択 | 10 | 7 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 787 | 789 |
| category_chat | fold_04 | 条件選択 | 10 | 7 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 786 | 788 |
| category_chat | fold_05 | 条件選択 | 10 | 7 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 788 | 790 |
| category_classification | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_cloud | fold_01 | 条件選択 | 30 | 4 | 34 | 34 | 68 | 102 | 136 | 170 | 204 | 7 | 781 | 788 |
| category_collaboration | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1972 | 1973 |
| category_collaboration | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1972 | 1973 |
| category_content | fold_01 | 条件選択 | 44 | 1 | 45 | 45 | 90 | 135 | 180 | 225 | 270 | 11 | 777 | 788 |
| category_coupons | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1972 | 1973 |
| category_coupons | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1972 | 1973 |
| category_covid_19 | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_cryptocurrency | fold_01 | 条件選択 | 1 | 3 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1973 | 1974 |
| category_cryptocurrency | fold_02 | 条件選択 | 1 | 3 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1971 | 1972 |
| category_cycling | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_data | fold_01 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1315 | 1316 |
| category_data | fold_02 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1314 | 1315 |
| category_data | fold_03 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1314 | 1315 |
| category_data_as_a_service | fold_01 | 条件選択 | 3 | 4 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 987 | 988 |
| category_data_as_a_service | fold_02 | 条件選択 | 3 | 4 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 985 | 986 |
| category_data_as_a_service | fold_03 | 条件選択 | 3 | 4 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 984 | 985 |
| category_data_as_a_service | fold_04 | 条件選択 | 3 | 4 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 986 | 987 |
| category_data_mining | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1971 | 1972 |
| category_data_mining | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1973 | 1974 |
| category_database | fold_01 | 条件選択 | 34 | 12 | 46 | 46 | 92 | 138 | 184 | 230 | 276 | 8 | 781 | 789 |
| category_design | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_design | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_developers | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_documents | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_domains | fold_01 | 診断のみ | 1 | 4 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 0 | 790 | 790 |
| category_ecommerce | fold_01 | 条件選択 | 232 | 54 | 286 | 286 | 572 | 858 | 1144 | 1430 | 1716 | 58 | 731 | 789 |
| category_education | fold_01 | 条件選択 | 21 | 12 | 33 | 33 | 66 | 99 | 132 | 165 | 198 | 5 | 784 | 789 |
| category_electronic_signature | fold_01 | 条件選択 | 17 | 1 | 18 | 18 | 36 | 54 | 72 | 90 | 108 | 4 | 784 | 788 |
| category_email | fold_01 | 条件選択 | 35 | 14 | 49 | 49 | 98 | 147 | 196 | 245 | 294 | 9 | 781 | 790 |
| category_enterprise | fold_01 | 条件選択 | 11 | 17 | 28 | 28 | 56 | 84 | 112 | 140 | 168 | 3 | 785 | 788 |
| category_entertainment | fold_01 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 787 | 790 |
| category_entertainment | fold_02 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 786 | 789 |
| category_entertainment | fold_03 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 786 | 789 |
| category_entertainment | fold_04 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 786 | 789 |
| category_entertainment | fold_05 | 条件選択 | 12 | 5 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 787 | 789 |
| category_environment | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1974 | 1975 |
| category_environment | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1970 | 1971 |
| category_esports | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_events | fold_01 | 条件選択 | 15 | 6 | 21 | 21 | 42 | 63 | 84 | 105 | 126 | 4 | 786 | 790 |
| category_fax | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1972 | 1973 |
| category_fax | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1972 | 1973 |
| category_feedback | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_feeds | fold_01 | 条件選択 | 11 | 4 | 15 | 15 | 30 | 45 | 60 | 75 | 90 | 3 | 787 | 790 |
| category_feeds | fold_02 | 条件選択 | 11 | 4 | 15 | 15 | 30 | 45 | 60 | 75 | 90 | 3 | 787 | 790 |
| category_feeds | fold_03 | 条件選択 | 11 | 4 | 15 | 15 | 30 | 45 | 60 | 75 | 90 | 3 | 786 | 789 |
| category_feeds | fold_04 | 条件選択 | 11 | 4 | 15 | 15 | 30 | 45 | 60 | 75 | 90 | 3 | 786 | 789 |
| category_feeds | fold_05 | 条件選択 | 12 | 4 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 2 | 786 | 788 |
| category_file_sharing | fold_01 | 条件選択 | 4 | 6 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 1 | 788 | 789 |
| category_file_sharing | fold_02 | 条件選択 | 4 | 6 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 1 | 788 | 789 |
| category_file_sharing | fold_03 | 条件選択 | 4 | 6 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 1 | 790 | 791 |
| category_file_sharing | fold_04 | 条件選択 | 4 | 6 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 1 | 788 | 789 |
| category_file_sharing | fold_05 | 条件選択 | 4 | 6 | 10 | 10 | 20 | 30 | 40 | 50 | 60 | 1 | 787 | 788 |
| category_financial | fold_01 | 条件選択 | 28 | 20 | 48 | 48 | 96 | 144 | 192 | 240 | 288 | 7 | 781 | 788 |
| category_fitness | fold_01 | 条件選択 | 14 | 2 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 4 | 785 | 789 |
| category_food | fold_01 | 条件選択 | 4 | 4 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 2 | 787 | 789 |
| category_food | fold_02 | 条件選択 | 5 | 4 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 788 | 789 |
| category_food | fold_03 | 条件選択 | 5 | 4 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 790 | 791 |
| category_food | fold_04 | 条件選択 | 5 | 4 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 788 | 789 |
| category_food | fold_05 | 条件選択 | 5 | 4 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 787 | 788 |
| category_forms | fold_01 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 987 | 988 |
| category_forms | fold_02 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 984 | 985 |
| category_forms | fold_03 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 985 | 986 |
| category_forms | fold_04 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 986 | 987 |
| category_framework | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_gambling | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_games | fold_01 | 条件選択 | 10 | 6 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 788 | 791 |
| category_games | fold_02 | 条件選択 | 10 | 6 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 788 | 791 |
| category_games | fold_03 | 条件選択 | 10 | 6 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 785 | 788 |
| category_games | fold_04 | 条件選択 | 11 | 6 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 786 | 788 |
| category_games | fold_05 | 条件選択 | 11 | 6 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 786 | 788 |
| category_geography | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1971 | 1972 |
| category_geography | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1973 | 1974 |
| category_government | fold_01 | 条件選択 | 18 | 27 | 45 | 45 | 90 | 135 | 180 | 225 | 270 | 6 | 785 | 791 |
| category_hardware | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_health | fold_01 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1317 | 1318 |
| category_health | fold_02 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1313 | 1314 |
| category_health | fold_03 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1313 | 1314 |
| category_healthcare | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_healthcare | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_home_automation | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_hosting | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_hosting | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_identity | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_images | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_intelligence | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_internet_things | fold_01 | 条件選択 | 2 | 5 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 1315 | 1316 |
| category_internet_things | fold_02 | 条件選択 | 2 | 5 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 1314 | 1315 |
| category_internet_things | fold_03 | 条件選択 | 2 | 5 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 1314 | 1315 |
| category_invoicing | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_jobs | fold_01 | 条件選択 | 24 | 6 | 30 | 30 | 60 | 90 | 120 | 150 | 180 | 6 | 784 | 790 |
| category_language | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_learning_management_systems | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_library | fold_01 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1315 | 1316 |
| category_library | fold_02 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1314 | 1315 |
| category_library | fold_03 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1314 | 1315 |
| category_location | fold_01 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1315 | 1316 |
| category_location | fold_02 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1314 | 1315 |
| category_location | fold_03 | 条件選択 | 2 | 2 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1314 | 1315 |
| category_machine_learning | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_mail | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_mapping | fold_01 | 条件選択 | 1334 | 52 | 1386 | 1386 | 2772 | 1821 | 3207 | 1821 | 3207 | 334 | 457 | 791 |
| category_marketing | fold_01 | 条件選択 | 3 | 2 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 987 | 988 |
| category_marketing | fold_02 | 条件選択 | 3 | 2 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 985 | 986 |
| category_marketing | fold_03 | 条件選択 | 3 | 2 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 986 | 987 |
| category_marketing | fold_04 | 条件選択 | 3 | 2 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 984 | 985 |
| category_media | fold_01 | 条件選択 | 16 | 8 | 24 | 24 | 48 | 72 | 96 | 120 | 144 | 4 | 786 | 790 |
| category_medical | fold_01 | 条件選択 | 2 | 5 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 1315 | 1316 |
| category_medical | fold_02 | 条件選択 | 2 | 5 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 1314 | 1315 |
| category_medical | fold_03 | 条件選択 | 2 | 5 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 1 | 1314 | 1315 |
| category_medicine | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_merchants | fold_01 | 条件選択 | 1 | 3 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1971 | 1972 |
| category_merchants | fold_02 | 条件選択 | 1 | 3 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 1973 | 1974 |
| category_messaging | fold_01 | 条件選択 | 143 | 22 | 165 | 165 | 330 | 495 | 660 | 825 | 990 | 36 | 753 | 789 |
| category_monitoring | fold_01 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 986 | 987 |
| category_monitoring | fold_02 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 987 | 988 |
| category_monitoring | fold_03 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 984 | 985 |
| category_monitoring | fold_04 | 条件選択 | 3 | 1 | 4 | 4 | 8 | 12 | 16 | 20 | 24 | 1 | 985 | 986 |
| category_movies | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_music | fold_01 | 条件選択 | 92 | 29 | 121 | 121 | 242 | 363 | 484 | 605 | 726 | 23 | 768 | 791 |
| category_natural_language_processing | fold_01 | 診断のみ | 0 | 2 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 792 | 792 |
| category_news_services | fold_01 | 条件選択 | 26 | 12 | 38 | 38 | 76 | 114 | 152 | 190 | 228 | 7 | 781 | 788 |
| category_nutrition | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1971 | 1972 |
| category_nutrition | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1973 | 1974 |
| category_office | fold_01 | 条件選択 | 13 | 11 | 24 | 24 | 48 | 72 | 96 | 120 | 144 | 3 | 786 | 789 |
| category_other | fold_01 | 条件選択 | 60 | 25 | 85 | 85 | 170 | 255 | 340 | 425 | 510 | 15 | 773 | 788 |
| category_parsing | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_payments | fold_01 | 条件選択 | 26 | 15 | 41 | 41 | 82 | 123 | 164 | 205 | 246 | 7 | 781 | 788 |
| category_pdf | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_performance | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1973 | 1974 |
| category_performance | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1971 | 1972 |
| category_personal_information_management | fold_01 | 条件選択 | 14 | 7 | 21 | 21 | 42 | 63 | 84 | 105 | 126 | 3 | 786 | 789 |
| category_photos | fold_01 | 条件選択 | 287 | 26 | 313 | 313 | 626 | 939 | 1252 | 1565 | 1878 | 72 | 716 | 788 |
| category_platform_as_a_service | fold_01 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1972 | 1973 |
| category_platform_as_a_service | fold_02 | 条件選択 | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 1 | 1972 | 1973 |
| category_politics | fold_01 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1316 | 1317 |
| category_politics | fold_02 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1313 | 1314 |
| category_politics | fold_03 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1314 | 1315 |
| category_predictions | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_printing | fold_01 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1972 | 1973 |
| category_printing | fold_02 | 条件選択 | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1972 | 1973 |
| category_products | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_project_management | fold_01 | 条件選択 | 15 | 11 | 26 | 26 | 52 | 78 | 104 | 130 | 156 | 4 | 785 | 789 |
| category_purchasing | fold_01 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1315 | 1316 |
| category_purchasing | fold_02 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1315 | 1316 |
| category_purchasing | fold_03 | 条件選択 | 2 | 1 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 1 | 1313 | 1314 |
| category_qa | fold_01 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 789 | 790 |
| category_qa | fold_02 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 789 | 790 |
| category_qa | fold_03 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 788 | 789 |
| category_qa | fold_04 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 788 | 789 |
| category_qa | fold_05 | 条件選択 | 4 | 2 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 787 | 788 |
| category_random | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_real_estate | fold_01 | 条件選択 | 14 | 7 | 21 | 21 | 42 | 63 | 84 | 105 | 126 | 3 | 786 | 789 |
| category_recognition | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_recommendations | fold_01 | 条件選択 | 28 | 5 | 33 | 33 | 66 | 99 | 132 | 165 | 198 | 7 | 782 | 789 |
| category_reference | fold_01 | 条件選択 | 102 | 39 | 141 | 141 | 282 | 423 | 564 | 705 | 846 | 25 | 763 | 788 |
| category_reporting | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_rewards | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_ride_share | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_sales | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_science | fold_01 | 条件選択 | 12 | 8 | 20 | 20 | 40 | 60 | 80 | 100 | 120 | 3 | 786 | 789 |
| category_screenshots | fold_01 | 条件選択 | 4 | 1 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 790 | 791 |
| category_screenshots | fold_02 | 条件選択 | 4 | 1 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 787 | 788 |
| category_screenshots | fold_03 | 条件選択 | 4 | 1 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 787 | 788 |
| category_screenshots | fold_04 | 条件選択 | 4 | 1 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 789 | 790 |
| category_screenshots | fold_05 | 条件選択 | 4 | 1 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 1 | 788 | 789 |
| category_search | fold_01 | 条件選択 | 96 | 35 | 131 | 131 | 262 | 393 | 524 | 655 | 786 | 23 | 766 | 789 |
| category_security | fold_01 | 条件選択 | 10 | 8 | 18 | 18 | 36 | 54 | 72 | 90 | 108 | 2 | 787 | 789 |
| category_semantics | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_sentiment | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_seo | fold_01 | 条件選択 | 4 | 5 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 788 | 789 |
| category_seo | fold_02 | 条件選択 | 4 | 5 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 788 | 789 |
| category_seo | fold_03 | 条件選択 | 4 | 5 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 787 | 788 |
| category_seo | fold_04 | 条件選択 | 4 | 5 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 787 | 788 |
| category_seo | fold_05 | 条件選択 | 4 | 5 | 9 | 9 | 18 | 27 | 36 | 45 | 54 | 1 | 791 | 792 |
| category_shipping | fold_01 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 787 | 790 |
| category_shipping | fold_02 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 787 | 790 |
| category_shipping | fold_03 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 785 | 788 |
| category_shipping | fold_04 | 条件選択 | 11 | 5 | 16 | 16 | 32 | 48 | 64 | 80 | 96 | 3 | 787 | 790 |
| category_shipping | fold_05 | 条件選択 | 12 | 5 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 2 | 786 | 788 |
| category_social | fold_01 | 条件選択 | 536 | 48 | 584 | 584 | 1168 | 1752 | 2336 | 2621 | 3205 | 134 | 655 | 789 |
| category_spam | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_sports | fold_01 | 条件選択 | 12 | 9 | 21 | 21 | 42 | 63 | 84 | 105 | 126 | 3 | 788 | 791 |
| category_spreadsheets | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_standards | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_statistics | fold_01 | 条件選択 | 5 | 2 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 2 | 788 | 790 |
| category_statistics | fold_02 | 条件選択 | 5 | 2 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 2 | 787 | 789 |
| category_statistics | fold_03 | 条件選択 | 6 | 2 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 788 | 789 |
| category_statistics | fold_04 | 条件選択 | 6 | 2 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 789 | 790 |
| category_statistics | fold_05 | 条件選択 | 6 | 2 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 787 | 788 |
| category_stocks | fold_01 | 診断のみ | 1 | 2 | 3 | 3 | 6 | 9 | 12 | 15 | 18 | 0 | 789 | 789 |
| category_storage | fold_01 | 条件選択 | 41 | 7 | 48 | 48 | 96 | 144 | 192 | 240 | 288 | 10 | 778 | 788 |
| category_streaming | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_support | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 790 | 790 |
| category_tagging | fold_01 | 条件選択 | 4 | 1 | 5 | 5 | 10 | 15 | 20 | 25 | 30 | 2 | 789 | 791 |
| category_tagging | fold_02 | 条件選択 | 5 | 1 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 787 | 788 |
| category_tagging | fold_03 | 条件選択 | 5 | 1 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 788 | 789 |
| category_tagging | fold_04 | 条件選択 | 5 | 1 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 789 | 790 |
| category_tagging | fold_05 | 条件選択 | 5 | 1 | 6 | 6 | 12 | 18 | 24 | 30 | 36 | 1 | 787 | 788 |
| category_telephony | fold_01 | 条件選択 | 262 | 18 | 280 | 280 | 560 | 840 | 1120 | 1400 | 1680 | 65 | 723 | 788 |
| category_text | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_tools | fold_01 | 条件選択 | 90 | 38 | 128 | 128 | 256 | 384 | 512 | 640 | 768 | 23 | 766 | 789 |
| category_translation | fold_01 | 条件選択 | 15 | 2 | 17 | 17 | 34 | 51 | 68 | 85 | 102 | 4 | 785 | 789 |
| category_transportation | fold_01 | 条件選択 | 13 | 10 | 23 | 23 | 46 | 69 | 92 | 115 | 138 | 3 | 786 | 789 |
| category_travel | fold_01 | 条件選択 | 34 | 24 | 58 | 58 | 116 | 174 | 232 | 290 | 348 | 9 | 781 | 790 |
| category_url_shortener | fold_01 | 条件選択 | 17 | 2 | 19 | 19 | 38 | 57 | 76 | 95 | 114 | 4 | 785 | 789 |
| category_video | fold_01 | 条件選択 | 234 | 16 | 250 | 250 | 500 | 750 | 1000 | 1250 | 1500 | 59 | 730 | 789 |
| category_voice | fold_01 | 診断のみ | 0 | 1 | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 | 792 | 792 |
| category_weather | fold_01 | 条件選択 | 23 | 11 | 34 | 34 | 68 | 102 | 136 | 170 | 204 | 6 | 783 | 789 |
| category_web_site_management | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_wi_fi | fold_01 | 診断のみ | 1 | 1 | 2 | 2 | 4 | 6 | 8 | 10 | 12 | 0 | 789 | 789 |
| category_widgets | fold_01 | 条件選択 | 49 | 4 | 53 | 53 | 106 | 159 | 212 | 265 | 318 | 12 | 776 | 788 |
| category_wiki | fold_01 | 条件選択 | 4 | 3 | 7 | 7 | 14 | 21 | 28 | 35 | 42 | 2 | 788 | 790 |
| category_wiki | fold_02 | 条件選択 | 5 | 3 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 788 | 789 |
| category_wiki | fold_03 | 条件選択 | 5 | 3 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 789 | 790 |
| category_wiki | fold_04 | 条件選択 | 5 | 3 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 788 | 789 |
| category_wiki | fold_05 | 条件選択 | 5 | 3 | 8 | 8 | 16 | 24 | 32 | 40 | 48 | 1 | 787 | 788 |

## 付録C. 小規模学習に実際に使ったID

固定評価IDを含まない学習用entity_id。最終動作確認のValidationはない。

| case | 学習区分 | Train entity_id | Validation entity_id |
| --- | --- | --- | --- |
| sufficient_multi | lr=2e-05 | api:10422, api:11217, api:11751, mashup:100 | mashup:1, mashup:10, mashup:1001, mashup:1032 |
| sufficient_multi | lr=3e-05 | api:10422, api:11217, api:11751, mashup:100 | mashup:1, mashup:10, mashup:1001, mashup:1032 |
| sufficient_multi | final_refit | api:10422, api:11217, api:11751, mashup:1 | — |
| few_positive_multi | lr=2e-05 | api:14470, api:14536, api:18505, mashup:1007 | mashup:1, mashup:100, mashup:1002, mashup:533 |
| few_positive_multi | lr=3e-05 | api:14470, api:14536, api:18505, mashup:1007 | mashup:1, mashup:100, mashup:1002, mashup:533 |
| few_positive_multi | final_refit | api:14470, api:14536, api:18505, mashup:1203 | — |
| single_output | lr=2e-05 | api:18804, mashup:1155, mashup:1182, mashup:1230 | mashup:1, mashup:10, mashup:1006, mashup:1902 |
| single_output | lr=3e-05 | api:18804, mashup:1155, mashup:1182, mashup:1230 | mashup:1, mashup:10, mashup:1006, mashup:1902 |
| single_output | final_refit | api:18804, mashup:1035, mashup:1141, mashup:1155 | — |
| one_positive | final_refit | api:6051, mashup:3793 | — |

## 付録D. 根拠コードの関数名・行番号

本報告作成時の実コードをASTで取得した。データJSON/CSVの件数は全ファイルparseによるもので、ファイル先頭行から件数を推測したものではない。

| ファイル | 関数/クラス | 開始行 |
| --- | --- | --- |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | sampling_order | 44 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | sample_negatives | 50 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | safe_weights | 68 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | build_split | 82 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | build_classifier | 151 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | create | 169 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | validate | 257 |
| [pipeline.py](scripts/hierarchical_api_selection/api_negative_sampling/pipeline.py) | check | 259 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | choose_config | 17 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | imports | 48 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | model_class | 54 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | BertForMultilabelClassification | 56 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | __init__ | 57 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | forward | 61 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | seed_everything | 66 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | state_hash | 74 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | legacy_inference_class | 81 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | encode | 95 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | make_loss | 103 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | metrics_from_counts | 109 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | evaluate | 116 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | train_one | 129 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | verify_inputs | 171 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | load_classifier | 180 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | bounded_subset | 194 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | run | 202 |
| [training.py](scripts/hierarchical_api_selection/api_negative_sampling/training.py) | main | 285 |
| [run_smoke_tests.py](scripts/hierarchical_api_selection/api_negative_sampling/run_smoke_tests.py) | main | 11 |

## 付録E. 保存済み単体テスト19件の名称

| テスト関数 | 開始行 |
| --- | --- |
| test_ratio_nesting_and_reproduction | 18 |
| test_whole_group_and_shortage | 23 |
| test_mixed_positive_negative_group_mandatory | 27 |
| test_api_anchor_and_mashup_validation | 31 |
| test_reduced_folds_depend_on_mashup_positive_groups | 37 |
| test_single_positive_never_cv_or_select | 39 |
| test_one_mashup_positive_uses_common_default | 42 |
| test_holdout_positive_and_group_isolation | 45 |
| test_weights_all_positive_all_negative_and_normal | 51 |
| test_weight_extremes_finite | 54 |
| test_empty_train_rejected | 56 |
| test_empty_output_refused_before_reading | 58 |
| test_lr_loss_first_ratio_f1_first_median | 63 |
| test_ties_choose_smaller_ratio_and_lr | 72 |
| test_missing_duplicate_or_nonfinite_result_rejected | 75 |
| test_micro_f1_all_negative_explicit_zero | 80 |
| test_smoke_subset_keeps_both_roles | 82 |
| test_early_stopping_restores_earlier_best_epoch | 85 |
| test_single_output_loss_nonzero_gradient | 106 |

根拠：`scripts/hierarchical_api_selection/api_negative_sampling/tests/test_phase3.py`、`data/hierarchical_api_selection/api_negative_sampling_v2/checks/unit_test_results.json`。

## 付録F. スモーク成果物20ファイル

すべて`checks/execution_evidence.json`の保存ハッシュと一致した。

| 保存ファイル | bytes |
| --- | --- |
| [few_positive_multi/api_labels.json](results/hierarchical_api_selection/phase3_smoke_v1/few_positive_multi/api_labels.json) | 116 |
| [few_positive_multi/best_ratio_1_fold_01_lr_2e-05.pth](results/hierarchical_api_selection/phase3_smoke_v1/few_positive_multi/best_ratio_1_fold_01_lr_2e-05.pth) | 438022957 |
| [few_positive_multi/model_weights.pth](results/hierarchical_api_selection/phase3_smoke_v1/few_positive_multi/model_weights.pth) | 438022957 |
| [few_positive_multi/run_manifest.json](results/hierarchical_api_selection/phase3_smoke_v1/few_positive_multi/run_manifest.json) | 3742 |
| [few_positive_multi/training_results.json](results/hierarchical_api_selection/phase3_smoke_v1/few_positive_multi/training_results.json) | 8268 |
| [one_positive/api_labels.json](results/hierarchical_api_selection/phase3_smoke_v1/one_positive/api_labels.json) | 27 |
| [one_positive/model_weights.pth](results/hierarchical_api_selection/phase3_smoke_v1/one_positive/model_weights.pth) | 438010669 |
| [one_positive/run_manifest.json](results/hierarchical_api_selection/phase3_smoke_v1/one_positive/run_manifest.json) | 3637 |
| [one_positive/training_results.json](results/hierarchical_api_selection/phase3_smoke_v1/one_positive/training_results.json) | 1574 |
| [single_output/api_labels.json](results/hierarchical_api_selection/phase3_smoke_v1/single_output/api_labels.json) | 12 |
| [single_output/best_ratio_1_fold_01_lr_2e-05.pth](results/hierarchical_api_selection/phase3_smoke_v1/single_output/best_ratio_1_fold_01_lr_2e-05.pth) | 438010669 |
| [single_output/model_weights.pth](results/hierarchical_api_selection/phase3_smoke_v1/single_output/model_weights.pth) | 438010669 |
| [single_output/run_manifest.json](results/hierarchical_api_selection/phase3_smoke_v1/single_output/run_manifest.json) | 3622 |
| [single_output/training_results.json](results/hierarchical_api_selection/phase3_smoke_v1/single_output/training_results.json) | 6069 |
| [smoke_results.json](results/hierarchical_api_selection/phase3_smoke_v1/smoke_results.json) | 4840 |
| [sufficient_multi/api_labels.json](results/hierarchical_api_selection/phase3_smoke_v1/sufficient_multi/api_labels.json) | 1049 |
| [sufficient_multi/best_ratio_1_fold_01_lr_2e-05.pth](results/hierarchical_api_selection/phase3_smoke_v1/sufficient_multi/best_ratio_1_fold_01_lr_2e-05.pth) | 438167533 |
| [sufficient_multi/model_weights.pth](results/hierarchical_api_selection/phase3_smoke_v1/sufficient_multi/model_weights.pth) | 438167533 |
| [sufficient_multi/run_manifest.json](results/hierarchical_api_selection/phase3_smoke_v1/sufficient_multi/run_manifest.json) | 4761 |
| [sufficient_multi/training_results.json](results/hierarchical_api_selection/phase3_smoke_v1/sufficient_multi/training_results.json) | 34075 |
