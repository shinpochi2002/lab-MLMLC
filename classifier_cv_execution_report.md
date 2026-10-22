# 分類器単位Cross-Validation／Holdoutへの変更：詳細実行報告

## 1. 報告の目的と結論

本書は、階層的API推薦モデルのPhase 2を「全層共通Group 5-Fold」から「分類器ごとに独立したCV／Holdout」へ変更した作業の実行報告である。今回作成した実装・manifest・分割JSON・検証記録と、元のPhase 1データを読み取り照合して記載した。

**新しい分割を805分類器分作成した。カテゴリ分類器653個、API分類器152個が対象で、K-fold分割97個、Holdout 698個、独立分割不能10個である。既存Phase 1・旧共通Group 5-Fold・既存の学習コード・モデル・評価データは保持した。モデルの学習・再学習・推論評価は実施していない。**

「分割が成立した」ことと「教師分布上の問題がなく学習・評価できる」ことは異なる。特に単一出力・全正例の分類器では、CVのID割当が作れても損失重みや識別情報に問題が残る。以下では、この違いを分けて報告する。

本書の分類器数は、構成別カテゴリ分類器653個と、全16構成が共用するAPI分類器152個の合計。API分類器を16倍しては数えない。また同じサービスが異なる分類器へ再利用されるため、分類器の学習件数の総和を独立したサービス数とは呼ばない。

| 記録項目 | 保存値 |
| --- | --- |
| 対象リポジトリ | `/home/nakagawa/lab-MLMLC` |
| 本番生成開始の記録 | 2026-10-16 18:08:01 JST |
| 保存済み自動検証の完了記録 | 2026-10-16 18:08:42 JST |
| 参照Git commit | `5b8c7f5251128843b313b88304c3f2d4261e1cf0` |
| コード同定 | 新規Python 5ファイルのSHA-256をsource_manifestへ保存。上記commitだけでは未コミットの新規実装を同定できない |
| 分割policy | `classifier_local_threshold_20_v1` |
| 乱数seed | 42 |
| 本書作成時の再照合 | 保存済み成果物821件・宣言入力170件・新規Python 5ファイルのSHA-256に不一致なし。既存ファイルの保存メタデータに変更なし |

## 2. 依頼内容と今回の実施範囲

### 2.1 分割方式を変更した理由

今回の目的は、分類器ごとの学習条件をCVで選び、選択条件で最終分類モデルを作成するための準備である。階層モデル全体のFold別OOF推薦性能を測る目的ではない。

そのため、全21,907件へ共通Foldを割り当てる旧方式を使わず、**各分類器が実際に使う教師データを抽出した後、そのデータだけで独立して分割**した。同じサービスが異なる分類器で異なる役割に割り当てられることは許容する。一方、同じ分類器の同じFold内では、同一実体・同一本文groupをTrainとValidationへ分散させない。

### 2.2 実施した処理

1. 既存監査、再構成案、Phase 1・2の実施まとめ、旧成果物のREADME・manifest・検証結果、既存学習コードを確認。
2. 元のPhase 1データ・ラベル・本文groupと固定評価除外の結果を検証。
3. 既存16階層の経路から、すべてのカテゴリ分類器の教師を抽出。
4. API正例4,855サービスを152カテゴリのAPI分類器へ展開。
5. 各分類器に20件閾値と独立group数の規則を適用し、CV／Holdout／分割不能を判定。
6. FoldごとのTrain／Validation ID、正負支持数、未観測ラベル・定数教師等の状態を保存。
7. 分類器共有の候補を比較し、根拠をmanifestへ記録。
8. 集計CSV・比較報告・README・自動検証結果を作成。
9. 既存ファイルと生成物の保護・ハッシュを照合。

### 2.3 実施していない処理

- モデルの学習・再学習、学習済み重みやcheckpointの更新。
- 推論、API推薦精度の評価、Validation Lossやmicro-F1による実際の学習率選択。
- Balanced K-meansやword2vecによる階層構築。
- API分類用の負例サービスの生成、少数例への追加・複製、代替モデルや固定予測の実装。
- 固定Validation100件・Test236件の再分割・変更。
- 既存Phase 1・旧Group 5-Fold・監査レポート・既存コードの編集、移動、削除。

最新依頼は既存コード等を参照資料とし、以前の「論文本体を必ず参照」という条件を置き換えた。今回の既存仕様の説明はソースコードを根拠とするものであり、論文本体との直接照合済み・完全一致という主張はしない。

## 3. 維持したデータと研究条件

### 3.1 Phase 1の学習元データ

保存先：`data/hierarchical_api_selection/group_5fold_v1/`。今回このディレクトリを更新していない。

| データ・条件 | 件数・内容 | 今回の扱い |
| --- | --- | --- |
| カテゴリ分類用サービス | 21907 | 既存JSONを読み取り参照 |
| Mashup由来 | 3946 | 元のdescription・使用API・由来IDを維持 |
| 候補内API由来 | 909 | 自己APIを正例とする既存教師を維持 |
| 候補外API由来 | 17052 | カテゴリ分類専用。API正例へ含めない |
| API分類用正例 | 4855 | カテゴリ分類用21,907件の部分集合 |
| 復元済みMashup | 61 | Phase 1の復元結果を保持。再復元・再収集はしていない |
| 共通母集合の本文group | 21884 | group_idを保持。旧Fold IDは使わない |
| API正例の本文group | 4849 | 分類器内のgroup非重複に使用 |
| 最終推薦候補 | 909 | 候補API集合を変更しない |
| 最終カテゴリ | 152 | 固定カテゴリ集合を維持 |
| 既存階層構成 | 16 | 固定category_pathsを参照 |
| 固定構成選定Validation | 100 | 分類器教師・CV・Holdout・停止・条件選択から除外 |
| 固定最終Test | 236 | 同上 |

根拠：旧 `manifests/dataset_manifest.json`、`label_schema.json`、`exclusion_manifest.json`、`source_records/category_classification.json`、`api_classification_positive.json`、`groups.json`。これらを含む旧成果物の保存ハッシュを照合してから新分割を作成した。今回の `load_phase_one` はAPI正例が共通母集合の正確な部分集合であることも検査する。

### 3.2 引き継いだリーク対策と本文group

Phase 1では、固定評価336 Mashupの元実体、評価と同一本文の別名1件、空本文1件を除外した。既存拡張22,180件には評価実体332件が含まれ、残る4評価Mashupは同名APIの上書きで失われていたため復元しない扱いである。今回この除外結果を変更していない。

評価Mashupと同名でも別のAPI実体である4件は、元本文が異なるため保持されている。今回も名前だけの一律削除はしていない。

比較用本文は `HTML entity復号 → Unicode NFKC → 空白統一`。大文字小文字は保持する。元のモデル入力本文を正規化して書き換えてはいない。同一本文のレコードは教師が違っても同じgroupに置き、個々のラベルを自動統合しない。

今回新しく再生成したのは**分類器ごとの教師参照・target_labels・CV/Holdout割当**であり、Phase 1の母集合ではない。

## 4. 分類器別教師データの構成

### 4.1 カテゴリ分類器

各構成の固定`category_paths`を用い、親経路prefixに該当するサービスを抽出した。1サービスが同じ親分類器へ複数カテゴリから到達しても、当該分類器のmembersには1回だけ入り、教師はその親に属する子ラベルの集合となる。

- 第1段：全カテゴリ用サービスから上位クラスタを予測。
- 中間段：親経路に属するサービスから次段の経路ラベルを予測。
- 最終カテゴリ段：親経路に属するサービスから具体的なカテゴリ名を予測。
- 出力ラベル一覧は固定階層から確定し、Foldに現れたラベルだけへ縮小しない。

`classifier_pipeline.py:classifiers`は旧スクリプトの純粋な`project_classifiers` helperを参照するが、旧学習モジュールはimportしない。新しいIDは`category_classification/<hierarchy>/layer_<nn>_parent_<path>`で、構成・層・親を追跡できる。

### 4.2 API分類器

API正例4,855サービスから、API→カテゴリ表で152分類器へ振り分けた。各分類器の教師は、そのサービスの正解APIと、当該カテゴリの固定候補APIの共通部分である。

- Mashup：元の使用API集合を保持し、カテゴリ内の該当APIを教師にする。
- 候補API自己例：そのAPI自身を正例にする。
- 候補外API：API分類のmembersへ入れない。
- 各API分類器の出力は固定909候補のカテゴリ別部分集合。152分類器の出力ラベル数の和は909である。
- 各レコードに対し909次元の出力を新たに学習したという意味ではない。現行階層モデルに対応するカテゴリ別出力を固定している。

カテゴリ別API正例の延べ件数は**6,514行**。同じMashupが複数カテゴリへ割り当てられるため、4,855サービスとは異なる。API分類器は同一教師・同一固定出力・同一規定条件で全16構成から使う152個であり、2,432個として重複計数しない。

### 4.3 保存形式と入力の追跡

本文を805回分コピーせず、1分類器1 JSONに次を保存した。

| 項目 | 内容 |
| --- | --- |
| `classifier_uid` | 構成／層／親、またはAPIカテゴリで一意な分類器ID |
| `source_record_file`, `source_record_sha256` | Phase 1の本文・由来を参照する元JSONとhash |
| `members` | `entity_id`, `group_id`, `target_labels`, `source_type` |
| `output_labels` | 固定された出力順 |
| `partition.full_profile` | 全教師件数・group数・由来・各出力の正負支持数 |
| `partition.method`, `n_splits` | kfold／holdout／unavailableと分割回数 |
| `partition.folds` | Fold別のTrain／Validationのentity ID、支持数、警告 |
| `conditions` | 今回の分割規則と、将来の学習用に記録した既存条件 |
| `training_equivalence_signature` | 共有候補を比較する教師・意味・規定条件のhash |

`target_labels`は分類器固有の教師である。将来の学習では、entity_idから本文を参照したうえで、この教師を使う必要がある。元レコードの全カテゴリ・全APIを無条件に各分類器へ渡してはならない。

## 5. 実装したCV／Holdoutの規則

### 5.1 閾値とFold数

nを分類器のサービス件数、Gをそのmembers内の独立group数とする。

| 条件 | 実装した処理 |
| --- | --- |
| n=0またはG<2 | `unavailable`。Fold数0、独立分割不能の理由を保存 |
| 0<n<20かつG>=2 | k=`min(5,G)`のCV。出力ラベル1種類でも試行 |
| n>=20かつG>=2 | 1回のgroup対応Holdout。おおむねTrain80%／Validation20% |

「20件」の判定はサービス件数nであり、group数Gではない。GはFold数・独立分割可能性の判定に使う。今回の分割不能10個はすべてn=G=1。

### 5.2 分類器固有のgroup層化近似

分類器ごとの正例ラベル支持数と由来をgroup単位へ集約し、希少ラベルから不足量を優先して配置する既存の副作用のない`stratify` helperを、**各分類器について個別に呼び出した**。

1. 分類器のgroup IDをソートし、`random.Random(42)`で決定的な優先順位を作る。
2. 正解ラベルのgroup支持数を使って希少ラベルを優先する。
3. 各分割先の目標ラベル支持数に対する不足量を比較する。副次的に他ラベル・由来の不足、残りgroup容量、分割先番号で順序を決める。
4. group数の容量を守って1groupを丸ごと配置する。
5. 配置後にサンプルをTrainへ移動する補正は行わない。

group分離を厳守するため、`MultilabelStratifiedKFold`そのものと同一の割当を生成する実装ではない。完全な層化は保証せず、実際の支持数と、CV Fold数より正例group数が少ないラベルを保存している。

単一出力では、両クラスが実際に存在する場合に正例・負例のbinary支持数を考慮する。今回の実データの単一出力68分類器には負例がなく、全正例の定数教師である。この事実を層化で解決したようには扱わない。

### 5.3 Holdoutの80:20はgroup単位

Validationのgroup数は次の式で決める。

```text
G_validation = max(1, min(G-1, floor(0.2*G+0.5)))
G_train = G - G_validation
```

groupごとのレコード数が違うため、サービス件数で厳密な80:20になるとは限らない。実測された698 HoldoutのValidationサービス比率は、最小18.1818%（electronic_signature）、最大23.5294%（government）。目標20%から10パーセントポイントを超えてずれるHoldoutは0個だった。

CVでは、groupが同じでも複数サービスを含む場合にサービス比率が1/kからずれることがある。該当警告はFold単位で3件あり、無理にgroupを割って比率を合わせていない。

### 5.4 Fold IDとEarly Stoppingの意味

- `fold_01`等は**classifier_uidの中だけで有効**。異なる分類器の同名Foldは共通評価集合を意味しない。
- 旧`group_5fold_v1/splits/`の割当を新しいTrain／Validationへ流用していない。ただし保護検証・由来記録のため旧成果物のhashは読む。
- 新分割では旧方式の「共通外側Fold＋内部Early Stopping20%」を作っていない。
- 既存API学習仕様を参照し、分類器ローカルのValidationを条件選択・Early Stoppingに用いる想定をmanifestへ記録した。学習自体は未実施。
- そのValidation成績は選択に用いる開発指標であり、階層全体の独立したOOF推薦性能でも、最終Test精度でもない。
- 固定100件／236件はいずれにも含まれない。

## 6. 分割生成の実測結果

| 分類器種別 | 5-fold | 4-fold | 3-fold | 2-fold | CV計 | Holdout | 分割不能 | 合計 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| カテゴリ | 2 | 0 | 0 | 0 | 2 | 651 | 0 | 653 |
| API | 30 | 14 | 9 | 42 | 95 | 47 | 10 | 152 |
| 合計 | 32 | 14 | 9 | 42 | 97 | 698 | 10 | 805 |

上表のCV計97は、97モデルを学習・評価した数ではない。最大5-foldが32分類器、group数に合わせた2〜4-foldが65分類器である。

保存されたTrain／Validationペアは**1,025組**で、内訳はCV327 FoldとHoldout698組。これはモデルの実行回数ではなく分割インスタンス数である。CV97分類器はそれぞれ複数Foldを持つため、分類器数とは単位が違う。

### 6.1 全16構成のカテゴリ分類器数

| 既存構成ID | 分類器数 | CV | Holdout | 段ごとの分類器数 | 段ごとの延べサービス割当 |
| --- | --- | --- | --- | --- | --- |
| layer2_4_43 | 5 | 0 | 5 | 1 / 4 | 21907 / 23003 |
| layer2_6_25 | 7 | 0 | 7 | 1 / 6 | 21907 / 23157 |
| layer2_8_19 | 9 | 0 | 9 | 1 / 8 | 21907 / 23319 |
| layer2_12_13 | 13 | 0 | 13 | 1 / 12 | 21907 / 23414 |
| layer2_15_10 | 16 | 0 | 16 | 1 / 15 | 21907 / 23435 |
| layer2_24_7 | 25 | 0 | 25 | 1 / 24 | 21907 / 23524 |
| layer2_33_4 | 34 | 0 | 34 | 1 / 33 | 21907 / 23490 |
| layer2_51_3 | 52 | 1 | 51 | 1 / 51 | 21907 / 23494 |
| layer3_4_4_8 | 21 | 0 | 21 | 1 / 4 / 16 | 21907 / 23003 / 23362 |
| layer3_8_4_4 | 41 | 0 | 41 | 1 / 8 / 32 | 21907 / 23319 / 23458 |
| layer3_12_2_6 | 37 | 0 | 37 | 1 / 12 / 24 | 21907 / 23414 / 23475 |
| layer3_12_4_3 | 61 | 0 | 61 | 1 / 12 / 48 | 21907 / 23414 / 23507 |
| layer3_12_6_2 | 85 | 1 | 84 | 1 / 12 / 72 | 21907 / 23414 / 23555 |
| layer3_16_4_3 | 81 | 0 | 81 | 1 / 16 / 64 | 21907 / 23467 / 23561 |
| layer4_8_3_2_3 | 81 | 0 | 81 | 1 / 8 / 24 / 48 | 21907 / 23319 / 23450 / 23554 |
| layer4_12_2_2_3 | 85 | 0 | 85 | 1 / 12 / 24 / 48 | 21907 / 23414 / 23475 / 23548 |

各段の延べサービス割当は親分岐への再利用を含む。同じ21,907件が複数段へ入るため、段・構成を合計して独立データ数にしない。

カテゴリ分類器のCV対象は次の2個のみで、残る651個は20件以上のHoldoutとなった。

| 分類器ID | サービス | group | 出力数 | Fold数 | 各FoldのTrain/Validation |
| --- | --- | --- | --- | --- | --- |
| [category_classification/layer2_51_3/layer_02_parent_39](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_39.json) | 18 | 18 | 2 | 5 | fold_01: 14/4; fold_02: 14/4; fold_03: 14/4; fold_04: 15/3; fold_05: 15/3 |
| [category_classification/layer3_12_6_2/layer_03_parent_5_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_3.json) | 12 | 12 | 2 | 5 | fold_01: 9/3; fold_02: 9/3; fold_03: 10/2; fold_04: 10/2; fold_05: 10/2 |

### 6.2 API分類器の集計

152 API分類器のうち、95個がCV、47個がHoldout、10個が1groupで分割不能。教師はすべて正例サービスのみの暫定版で、負例追加後にはn・G・適用方式が変わり得る。

## 7. 単一出力・少数データ・未観測ラベルの実測

### 7.1 単一出力の分割と教師分布

| 区分 | 分類器数 | 意味 |
| --- | --- | --- |
| 単一出力の全分類器 | 68 | すべてAPI分類器。全開発教師は全正例、負例0 |
| 単一出力でCV分割を作成 | 56 | 新仕様に従って分割を試行。定数教師の警告を付与 |
| 単一出力でHoldoutを作成 | 2 | content 56サービス、electronic_signature 22サービス |
| 単一出力で独立分割不能 | 10 | いずれも1サービス・1group |
| 通常の単一出力binary両クラスがある実データ | 0 | 処理は合成テストで検査したが、実際の正例専用データに負例を追加したわけではない |

出力1種類の正例だけを集めると全ラベルが1となる。既存の`pos_weight=negative_count/positive_count`は0になり、その正例BCE項が0となる。単にCVを実行しても、正例と負例を識別する課題にはならない。

分割不能10個を除いた単一出力58個は、作成したすべてのTrain部分集合でもその出力に負例がない。損失重みを変更したとしても、負例0という教師分布そのものは残る。今回はいずれも学習せず、`constant_targets_not_normal_cv_success`やPhase 3判断フラグを保存した。

### 7.2 独立分割不能10分類器

| APIカテゴリ | n | G | 出力ラベル | 元entity ID | 理由 |
| --- | --- | --- | --- | --- | --- |
| [booking](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_booking.json) | 1 | 1 | bookingcom-content | api:6051 | one_independent_group |
| [hardware](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_hardware.json) | 1 | 1 | amazon-alexa-skill-voice-service | api:1935 | one_independent_group |
| [learning-management-systems](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_learning_management_systems.json) | 1 | 1 | moodle | api:17673 | one_independent_group |
| [machine-learning](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_machine_learning.json) | 1 | 1 | rxnlp-cluster-sentences-and-short-texts | api:13496 | one_independent_group |
| [predictions](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_predictions.json) | 1 | 1 | google-prediction | api:6207 | one_independent_group |
| [recognition](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_recognition.json) | 1 | 1 | microsoft-cognitive-services-emotion | api:8692 | one_independent_group |
| [reporting](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_reporting.json) | 1 | 1 | google-analytics-core-reporting | api:10016 | one_independent_group |
| [sentiment](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_sentiment.json) | 1 | 1 | getsentiment | api:21198 | one_independent_group |
| [text](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_text.json) | 1 | 1 | rxnlp-text-similarity | api:13932 | one_independent_group |
| [voice](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_voice.json) | 1 | 1 | google-assistant | api:7781 | one_independent_group |

これらにはFoldを保存していない。1例を複製したり同じgroupをTrain／Validationへ分けたり、固定評価例を追加していない。将来の例外処理はPhase 3で決定する。

### 7.3 Trainに正例0の出力が生じる11分類器

下表は、少なくとも1回の分割でTrain正例0となる出力がある分類器。ラベル欄は、その分類器の全Foldを通じた対象ラベルの和集合であり、毎Foldで全部が同時に欠落するという意味ではない。

| 分類器 | n/G | 方式 | Trainで正例0となり得る出力ラベル |
| --- | --- | --- | --- |
| [category_accounting](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_accounting.json) | 3/3 | 3-fold | myob-essentials-accounting |
| [category_domains](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_domains.json) | 5/5 | 5-fold | dnsimple, domain, domaintools |
| [category_enterprise](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_enterprise.json) | 31/31 | Holdout | batchbook, highrise |
| [category_entertainment](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_entertainment.json) | 19/19 | 5-fold | dog-ceo-dog |
| [category_file_sharing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_file_sharing.json) | 11/11 | 5-fold | 4shared |
| [category_food](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_food.json) | 10/10 | 5-fold | open-dining-network-food-ordering, untappd |
| [category_government](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_government.json) | 51/49 | Holdout | govtrackus, maplight, opensecrets |
| [category_internet_things](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_internet_things.json) | 8/8 | 5-fold | amazon-alexa-proactive-events, amazon-alexa-smart-home-skills |
| [category_medical](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_medical.json) | 8/8 | 5-fold | health-20 |
| [category_natural_language_processing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_natural_language_processing.json) | 2/2 | 2-fold | dialogflow-standard-webhooks, nlp-tools |
| [category_seo](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_seo.json) | 10/10 | 5-fold | authoritylabs-account, authoritylabs-partner |

旧実装のようにValidationからTrainへサンプルを移して補正せず、発生した状態をそのまま保存している。出力ラベル一覧自体は削除していない。

### 7.4 警告の件数と解釈

| 警告コード | 該当Fold数 | 意味 |
| --- | --- | --- |
| train_zero_positive_labels | 17 | 少なくとも1出力にTrain正例がない |
| train_zero_negative_labels_legacy_pos_weight_zero | 147 | 少なくとも1出力にTrain負例がなく、旧重みではその正例項の重みが0 |
| train_all_targets_constant | 145 | すべての出力が当該Train内で定数 |
| validation_all_targets_constant | 255 | すべての出力が当該Validation内で定数 |
| validation_no_within_label_discrimination | 255 | Validation内で正負両方が観測される出力がない |
| validation_zero_positive_labels | 211 | 少なくとも1出力にValidation正例がない |
| service_ratio_deviation_over_0_10_due_to_group_constraints | 3 | サービスのValidation比率が目標から10ポイント超ずれる |

これらは**Fold単位の重複するフラグ**であり、足して分類器数や失敗数にしない。例えば1つの小標本Validationには`validation_all_targets_constant`と`validation_no_within_label_discrimination`が同時に付く。

分類器単位では、Train正例0が生じるもの11個、Train負例0の出力が生じるもの61個。これらも重複し得る。単一出力68個と61個が一致しないのは、1groupで分割を作らなかった10個にはFold内統計がない一方、複数出力でも一部出力がTrainで全正例になる分類器があるためである。

`requires_phase_3_decision`は、定数教師・未観測ラベル・Validation内の定数分布・比率差など**何らかの判断対象がある**ことを示す広いフラグである。API分類器では151個に付き、カテゴリ分類器では0個だった。CV97個のうち95個にこのフラグがあるが、**「CV分割不能95個」や「95モデルの学習失敗」を意味しない**。

1例だけのValidationでは、観測された各出力が定数なのは自然なことで、フラグがあるだけでP/R/F1や損失を計算できないとは限らない。逆にフラグがない2カテゴリ分類器もモデルを学習していないので、学習成功を確認したわけではない。

JSONの`legacy_positive_loss_can_all_be_zero`は、Trainに可変出力がない場合の**正例項**についての警告である。複数出力で全負例の出力があれば負例BCE項は残るため、損失全体が必ず0になるという意味ではない。

## 8. 分類器の共有候補

API分類器152個は、全16構成で同じサービス・教師・出力一覧・規定学習条件を使うため、1つのAPI分類器データを各構成から参照する。

カテゴリ分類器については、次をsignatureに含めて比較した。

- task、固定出力ラベルとその順序。
- 元entity IDと、その分類器の正解ラベル。
- 元カテゴリと出力ラベルの意味対応。
- 規定した分割・将来学習条件。

**共有候補の等価群23組、関与するカテゴリ分類器51個**が見つかった。ただし今回は共有を適用せず、653個をそれぞれ保存した。仮に各等価群を1個へまとめれば28個分を削減できる計算だが、これは未実施の算術上の候補数である。選択されたlr、epoch、損失、前処理等も一致することを確認しなければ、最終学習モデルを共有可能とは確定しない。

## 9. 既存学習実装との整合点と相違点

| 項目 | 既存コードで確認した仕様 | 今回の扱い |
| --- | --- | --- |
| CV対象 | n<20かつ出力2種類以上 | n<20かつG>=2なら単一出力でも試行する明示変更 |
| n>=20 | 8:2 Holdout | 同じ閾値。group単位・seed42の近似層化 |
| Fold数 | max(2,min(5,n)) | min(5,G)。G<2では分割不能 |
| 層化 | MultilabelStratifiedKFold、groups非対応 | group対応の希少ラベル優先割当。厳密な同一割当ではない |
| Trainラベル保持 | Validationからサンプルを移す補正 | 移動せず、未観測ラベルを記録 |
| カテゴリ段 | 既存study配列を順序80:20で分割する版がある | 653分類器ごとにn/G判定を適用 |
| CVの学習率候補 | 2e-5、3e-5 | 将来学習用設定として保存。比較学習は未実施 |
| 条件選択 | 平均val_loss最小、同値なら平均micro-F1最大 | 同じ規則を保存。選択結果はまだ存在しない |
| Holdout学習率 | 2e-5固定 | 同値を将来学習条件として保存 |
| Early Stopping | val_loss、patience3、最大50epoch | 設定を記録。今回停止処理を実行していない |
| CV後の最終学習 | 選択lrで全データを50epoch、Validationなし | 既存参照条件として記録。再学習は未実施 |
| 出力空間 | データに観測されたAPIから組む版 | 固定909 APIのカテゴリ別出力・固定階層の出力順を保存 |
| API負例 | 現行autoローダーは空used_apisを除外 | 正例4,855サービスのみ。新しい負例は生成しない |
| 旧共通Fold | 全層で同じ外側・内部割当 | 今回の方式では使わない。旧成果物は保持 |

既存APIコードの根拠：`scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py`。

- 30–37行：閾値、K、lr、seed、patience。
- `load_category` 124–155行：空本文・空API除外、教師化。
- `split_with_all_labels` 159–209行：ラベル保持とValidationからの移動。
- `train_holdout` 222–270行：Holdout学習率・重み・停止設定。
- `train_kfold_then_final` 272–386行：CV、平均指標によるlr選択、全データ再学習。
- `train_one_category_auto` 399–430行、特に416行：n<20かつラベル数>1の条件。

既存カテゴリコードの例：`mashup_clu_mlc_nosp_nostop_layer3.py:127–146`は教師・分割前の重み計算・80:20スライス、163–176行は停止と保存。通常API版 `mashup_clu_mlc_nosp_nostop_category.py:173–178,212–226`は空API除外・分割前重み・順序80:20であり、CV版とは別の経路である。

既存コードには停止時の重みを保存し最良epochの重みを復元しない箇所もある。今回、最良モデル保存や最終epoch数を変更した学習実装を作ったわけではない。

## 10. 実行したスクリプトと処理順

| 新規ファイル | 役割 | 主要関数・行番号 |
| --- | --- | --- |
| [classifier_pipeline.py](scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py) | 教師抽出・独立分割・集計・検証の共通処理 | load_phase_one:62, classifiers:100, target_profile:141, make_partition:169, create:217, summarize:303, validate:491 |
| [create_classifier_splits.py](scripts/hierarchical_api_selection/classifier_cv/create_classifier_splits.py) | 教師・805分割JSON・manifestの作成 | main("create") |
| [summarize_classifier_splits.py](scripts/hierarchical_api_selection/classifier_cv/summarize_classifier_splits.py) | CSV・README・実装報告の作成 | main("summarize") |
| [validate_classifier_splits.py](scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py) | 読み取り検証、指定時のみ検証結果を新規保存 | main("validate") |
| [tests/test_classifier_splits.py](scripts/hierarchical_api_selection/classifier_cv/tests/test_classifier_splits.py) | 閾値・group分離・単一出力・上書き等の合成テスト | 13 test methods |
| [README.md](scripts/hierarchical_api_selection/classifier_cv/README.md) | 入力・出力・再生成・状態の説明 | — |

作業時は、既存13,309ファイルの状態を記録→新規実装→13件の合成テスト→一時ディレクトリで生成・集計・検証→本番専用ディレクトリへ同じ処理を実行→ハッシュ・既存ファイルの最終照合、の順で進めた。

本番は`set -e`付きの実行で、いずれかのコマンドが失敗した場合は後続処理を継続しない形とした。

```bash
set -e
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/create_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/summarize_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1 --write-report
```

モデル学習ライブラリのインストールや新しい依存導入はしていない。標準ライブラリと、既存Phase 1スクリプトの副作用のないhelperを使用した。

## 11. 自動検証と本書作成時の再照合

### 11.1 保存済み自動検証

`checks/validation_results.json`に**12,553項目、すべてpassed=true**が記録されている。これは多数の分類器・Fold・roleごとのチェックを展開した項目数で、12,553種類の異なる検証アルゴリズムや学習実験という意味ではない。

主な検査は次の通り。

- Phase 1の21,907件と4,855件が保持され、正例部分集合・entity ID一意性・本文groupが一致する。
- 固定100／236件との実体・比較用本文の混入がない。
- 909 API・152カテゴリ・16固定階層を維持。
- 805分類器の定義・固定出力・教師ID・target_labelsを再現できる。
- profileを独立な計数経路で再計算し、全教師・各Foldの正負支持数が一致する。
- Train／Validationは非空かつentity・groupが非重複で、和集合は当該分類器のmembers全体。
- CVはk<=Gを満たし、Validationのgroup割当が重複せず全groupを一巡する。
- 同一seedでの再計算と入力順反転で同一割当。
- 単一出力定数教師を正常なCV成功と主張しない状態が記録されている。
- CSV全行と教師からの再計算結果が一致する。
- 共有候補signatureと固定意味対応を再現できる。
- 旧全層共通Foldを使用せず、分割後のサンプル移動が0である。

保存済みリーク検査結果は、固定評価重複0、分類器内group重複0、seed再現true、旧共通Fold使用falseである。

### 11.2 合成テスト13件

実装時に実行して13件が通過した。保存済み自動検証JSONとは別のテストであり、学習の成否を調べるものではない。

| テスト関数 | 検査内容 |
| --- | --- |
| `test_single_record_never_attempts_twofold` | 1件で2-foldを強行しない |
| `test_multiple_records_one_text_group_still_unavailable` | 複数レコードでも1本文groupなら分割不能 |
| `test_fold_count_uses_independent_groups` | kを独立group数で制限する |
| `test_twenty_record_boundary_is_holdout` | 19件と20件で方式が切り替わる |
| `test_single_output_all_positive_attempts_but_does_not_claim_normal_success` | 単一出力全正例を分割できても正常学習成功としない |
| `test_single_output_binary_both_classes_supported` | 両クラスがある合成単一出力にも対応する |
| `test_fold_group_disjoint_and_validation_complete` | group非重複とValidation一巡 |
| `test_holdout_two_groups_never_rounds_validation_to_zero` | 2groupのHoldoutでValidationを0に丸めない |
| `test_no_silent_move_of_validation_only_label` | 未観測Trainラベルを無断移動で補正しない |
| `test_reproducible_reordered_and_input_not_mutated` | 入力順に依存せず元データを変更しない |
| `test_group_can_change_role_across_independent_classifiers` | 分類器固有nにより独立した方式を適用できる |
| `test_profile_validator_independently_reproduces_counts` | 独立な計数経路でprofileが一致 |
| `test_no_overwrite_and_no_inputs_read_when_output_exists` | 既存出力があれば停止し、無条件上書きしない |

「分類器をまたいでgroupが異なる役割になれる」テストは、異なるnでCVとHoldoutが独立に選択されることを検査するもので、あらゆる分類器ペアが必ず違うID割当になることを保証するテストではない。同じ教師・同じseedなら偶然または必然的に同じ割当になる場合もあり、それは共通Foldを流用したことにはならない。

### 11.3 本書作成時に追加で読み取り確認したこと

本書の作成にあたり、データ生成や学習を再実行せず、全805分割JSONの教師ID・件数をmanifest/CSVと再照合し、各Foldのentity/group分離とラベル支持数を再集計した。CVのValidation一巡も確認した。

加えて、保存済み821成果物のSHA-256、宣言入力170ファイルと新規Python 5ファイルのSHA-256を照合し、不一致0だった。保存された既存ファイル13,315件のsize・mtime・mode・linkにも変更を検出しなかった。

**保護件数の違い**：作業開始時の既存ファイルは13,309件。新規スクリプト6ファイルを作成した後、本番生成前に記録したsnapshotは13,315件。この6件分の差を、既存ファイルが増減・改変された問題と混同しない。前回作業の最終照合では開始時13,309件の変更なしを確認し、本書作成時は保存済みsnapshot13,315件を再照合した。

**検証の限界**：既存モデル本体の数百GBの内容ハッシュは取得していない。宣言入力・コードはSHA-256、その他の既存ファイルはメタデータ比較が保護検証の範囲である。近似本文・未知の別名をすべて検出した保証はなく、Phase 1の実体照合と完全一致正規化の範囲を引き継ぐ。

## 12. 作成した成果物とファイルの役割

保存先：`data/hierarchical_api_selection/classifier_cv_v1/`。

**データ側822ファイル**＝805分類器JSON＋共通manifest 4 JSON＋集計summary 1 JSON＋検証結果1 JSON＋CSV 8ファイル＋Markdown 3ファイル。新規スクリプト側6ファイルはこの822に含まない。本書はさらに別の新規報告ファイルであり、ハッシュ固定済みのデータディレクトリへ追記していない。

| 成果物 | 内容 |
| --- | --- |
| [README.md](data/hierarchical_api_selection/classifier_cv_v1/README.md) | 用途・入力・全スキーマ・生成/再生成手順・Phase 3引継ぎ |
| [manifests/source_manifest.json](data/hierarchical_api_selection/classifier_cv_v1/manifests/source_manifest.json) | 入力hash、コードhash、基点commit、条件、生成時刻 |
| [manifests/existing_file_snapshot.json](data/hierarchical_api_selection/classifier_cv_v1/manifests/existing_file_snapshot.json) | 生成前の既存13,315ファイルのメタデータ |
| [manifests/classifier_manifest.json](data/hierarchical_api_selection/classifier_cv_v1/manifests/classifier_manifest.json) | 805分類器の定義・出力・件数・分割ファイル・共有候補 |
| [manifests/split_manifest.json](data/hierarchical_api_selection/classifier_cv_v1/manifests/split_manifest.json) | 閾値・group優先・独立分割の規則・将来学習参照条件 |
| [reports/classifier_split_summary.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/classifier_split_summary.csv) | 805分類器の方式、n/G、支持数警告等の概要 |
| [reports/category_classifier_counts.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/category_classifier_counts.csv) | 653カテゴリ分類器の概要 |
| [reports/api_classifier_counts.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/api_classifier_counts.csv) | 152 API分類器の概要 |
| [reports/rare_classifier_summary.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/rare_classifier_summary.csv) | n<20またはPhase 3判断対象がある分類器 |
| [reports/unavailable_classifiers.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/unavailable_classifiers.csv) | 独立分割不能10個 |
| [reports/single_output_classifiers.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/single_output_classifiers.csv) | 単一出力68個の状態 |
| [reports/fold_sample_counts.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/fold_sample_counts.csv) | 各Fold/roleのサービス・group・由来件数 |
| [reports/label_supports.csv](data/hierarchical_api_selection/classifier_cv_v1/reports/label_supports.csv) | 全開発集合と各Fold/roleの全出力の正負支持数 |
| [reports/summary_counts.json](data/hierarchical_api_selection/classifier_cv_v1/reports/summary_counts.json) | 分類器・方式・警告等の機械可読総数 |
| [reports/existing_implementation_comparison.md](data/hierarchical_api_selection/classifier_cv_v1/reports/existing_implementation_comparison.md) | 既存実装との比較・根拠行 |
| [reports/implementation_report.md](data/hierarchical_api_selection/classifier_cv_v1/reports/implementation_report.md) | 生成時点の実装結果とPhase 3事項 |
| [checks/validation_results.json](data/hierarchical_api_selection/classifier_cv_v1/checks/validation_results.json) | 12,553チェック、検査範囲、成果物821件のhash |

個別分割は `splits/category_classification/<hierarchy>/layer_<nn>_parent_<path>.json` と `splits/api_classification/category_<name>.json`。全805件の具体的なパスは付録A・Bとclassifier_manifestに記載する。

検証結果内のhash一覧が821件なのは、検証JSON自身を自己hashに含めていないため。822ファイル全てのhashを内部に自己参照で固定したという意味ではない。

## 13. 再現・検証と次の実装への引継ぎ

### 13.1 再生成と読み取り検証

作成コマンドを同じ出力先へ再実行すると、既存出力を検出して停止する。無条件上書き・`--force`は実装していない。再生成は別の未使用ディレクトリを指定する。

```bash
set -e
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/create_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_reproduction
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/summarize_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_reproduction
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_reproduction --write-report
```

上記は再現用の例であり、その出力先に何もないことが前提。本書作成のためにこの再生成を実施したわけではない。作成時刻は変わるが、同一入力・同一seedなら教師・割当は同じになる。

既存結果の読み取り検証は以下。`--write-report`を付けないため成果物を更新しない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/classifier_cv/tests -v
```

### 13.2 Phase 3で判断・実装する事項

| 項目 | 現状 | 次に必要な判断 |
| --- | --- | --- |
| API負例 | 正例のみの暫定分割 | 既知Mashupからの負例、比率、sampling対象、全0教師の妥当性を決定 |
| 負例導入後の分割 | n/G・ラベル分布が変わり得る | 新版教師から再分割。現在のFoldを無条件に維持しない |
| 単一出力全正例 | 68分類器、うち分割あり58 | 損失重みと、識別教師をどう与えるかを決める |
| 1group | 10分類器で独立分割不能 | 学習対象外・代替等の研究方針を決める。今回は未実装 |
| Train正例0・負例0 | 対象出力を保存 | ラベル保持・損失・予測規則を決め、Validation移動で隠さない |
| 最良epoch・重み | 既存停止/再学習条件を参照記録のみ | 最良重み復元・epoch・全データ再学習を実装し検証 |
| 学習条件選択 | lr候補と比較規則を保存 | 実際の各分類器CV/Holdout学習を実行する次の承認範囲 |
| 分類器共有 | カテゴリ23候補群、共有未適用 | 選択後の条件まで一致した場合だけ共有を検討 |
| 固定評価 | 100/236を未変更・学習から隔離 | CV後の最終モデルで構成選定・推薦評価を実施する別工程 |
| Eとの比較・両評価平均 | 今回未計算 | 指標・平均単位・Testを使った再選定禁止を事前に固定 |

Phase 1が記録した `trackthis`・`rosetta` の固定カテゴリGTと元Mashup由来カテゴリの不一致も残る。今回、API正解・評価ファイルを修正していない。変更するなら評価版の承認・固定が必要である。

この段階ではAPI推薦精度の改善量、最適lr・epoch、未学習モデルの挙動を報告できない。分割作成・検証の完了は、モデルの再学習完了を意味しない。

## 付録A. カテゴリ分類器653個の全件一覧

nはサービス件数、Gは分類器内group数、Lは固定出力数。T/Vは各FoldのTrain/Validation件数の範囲であり、Holdoutではそれぞれ1つの実数。CVの詳細な各Foldはリンク先JSONを参照。リンク先の`members`が教師一覧、`output_labels`が出力順、`partition.folds`が割当である。

<details>
<summary>layer2_4_43：5分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_4_43/layer_01_parent_root.json) | 21907 | 21884 | 4 | Holdout | 17526 | 4381 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_4_43/layer_02_parent_0.json) | 6849 | 6845 | 38 | Holdout | 5478 | 1371 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_4_43/layer_02_parent_1.json) | 6303 | 6297 | 38 | Holdout | 5043 | 1260 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_4_43/layer_02_parent_2.json) | 7180 | 7172 | 38 | Holdout | 5745 | 1435 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_4_43/layer_02_parent_3.json) | 2671 | 2669 | 38 | Holdout | 2137 | 534 |

</details>

<details>
<summary>layer2_6_25：7分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_01_parent_root.json) | 21907 | 21884 | 6 | Holdout | 17525 | 4382 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_02_parent_0.json) | 2476 | 2476 | 25 | Holdout | 1981 | 495 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_02_parent_1.json) | 6764 | 6754 | 25 | Holdout | 5411 | 1353 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_02_parent_2.json) | 3759 | 3758 | 25 | Holdout | 3007 | 752 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_02_parent_3.json) | 4404 | 4401 | 25 | Holdout | 3524 | 880 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_02_parent_4.json) | 4395 | 4389 | 26 | Holdout | 3516 | 879 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_6_25/layer_02_parent_5.json) | 1359 | 1358 | 26 | Holdout | 1087 | 272 |

</details>

<details>
<summary>layer2_8_19：9分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_01_parent_root.json) | 21907 | 21884 | 8 | Holdout | 17526 | 4381 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_0.json) | 2044 | 2044 | 19 | Holdout | 1635 | 409 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_1.json) | 4252 | 4248 | 19 | Holdout | 3401 | 851 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_2.json) | 4606 | 4603 | 19 | Holdout | 3685 | 921 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_3.json) | 4504 | 4500 | 19 | Holdout | 3604 | 900 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_4.json) | 1913 | 1911 | 19 | Holdout | 1531 | 382 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_5.json) | 3422 | 3417 | 19 | Holdout | 2737 | 685 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_6.json) | 1758 | 1757 | 19 | Holdout | 1407 | 351 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_8_19/layer_02_parent_7.json) | 820 | 819 | 19 | Holdout | 656 | 164 |

</details>

<details>
<summary>layer2_12_13：13分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_01_parent_root.json) | 21907 | 21884 | 12 | Holdout | 17524 | 4383 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_0.json) | 1116 | 1116 | 12 | Holdout | 893 | 223 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_1.json) | 1299 | 1299 | 12 | Holdout | 1039 | 260 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_10.json) | 1327 | 1326 | 14 | Holdout | 1062 | 265 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_11.json) | 636 | 635 | 13 | Holdout | 509 | 127 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_2.json) | 3879 | 3875 | 12 | Holdout | 3104 | 775 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_3.json) | 2632 | 2630 | 12 | Holdout | 2105 | 527 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_4.json) | 2441 | 2438 | 12 | Holdout | 1952 | 489 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_5.json) | 3343 | 3341 | 12 | Holdout | 2674 | 669 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_6.json) | 1586 | 1584 | 13 | Holdout | 1269 | 317 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_7.json) | 1209 | 1209 | 13 | Holdout | 967 | 242 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_8.json) | 1969 | 1965 | 13 | Holdout | 1573 | 396 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_12_13/layer_02_parent_9.json) | 1977 | 1976 | 14 | Holdout | 1582 | 395 |

</details>

<details>
<summary>layer2_15_10：16分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_01_parent_root.json) | 21907 | 21884 | 15 | Holdout | 17523 | 4384 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_0.json) | 888 | 888 | 10 | Holdout | 710 | 178 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_1.json) | 1192 | 1192 | 10 | Holdout | 954 | 238 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_10.json) | 898 | 898 | 10 | Holdout | 718 | 180 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_11.json) | 852 | 851 | 10 | Holdout | 681 | 171 |
| [layer_02_parent_12](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_12.json) | 2161 | 2160 | 10 | Holdout | 1728 | 433 |
| [layer_02_parent_13](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_13.json) | 552 | 552 | 10 | Holdout | 442 | 110 |
| [layer_02_parent_14](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_14.json) | 621 | 620 | 12 | Holdout | 497 | 124 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_2.json) | 955 | 953 | 10 | Holdout | 764 | 191 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_3.json) | 3990 | 3988 | 10 | Holdout | 3191 | 799 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_4.json) | 1271 | 1269 | 10 | Holdout | 1017 | 254 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_5.json) | 4465 | 4460 | 10 | Holdout | 3572 | 893 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_6.json) | 1716 | 1716 | 10 | Holdout | 1373 | 343 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_7.json) | 1546 | 1544 | 10 | Holdout | 1237 | 309 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_8.json) | 941 | 941 | 10 | Holdout | 753 | 188 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_15_10/layer_02_parent_9.json) | 1387 | 1383 | 10 | Holdout | 1107 | 280 |

</details>

<details>
<summary>layer2_24_7：25分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_01_parent_root.json) | 21907 | 21884 | 24 | Holdout | 17525 | 4382 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_0.json) | 702 | 702 | 6 | Holdout | 562 | 140 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_1.json) | 577 | 577 | 7 | Holdout | 462 | 115 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_10.json) | 1155 | 1153 | 6 | Holdout | 924 | 231 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_11.json) | 1026 | 1024 | 7 | Holdout | 820 | 206 |
| [layer_02_parent_12](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_12.json) | 787 | 787 | 6 | Holdout | 630 | 157 |
| [layer_02_parent_13](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_13.json) | 2889 | 2886 | 6 | Holdout | 2311 | 578 |
| [layer_02_parent_14](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_14.json) | 272 | 272 | 6 | Holdout | 218 | 54 |
| [layer_02_parent_15](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_15.json) | 781 | 781 | 6 | Holdout | 625 | 156 |
| [layer_02_parent_16](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_16.json) | 432 | 432 | 6 | Holdout | 346 | 86 |
| [layer_02_parent_17](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_17.json) | 1311 | 1307 | 6 | Holdout | 1050 | 261 |
| [layer_02_parent_18](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_18.json) | 599 | 599 | 7 | Holdout | 479 | 120 |
| [layer_02_parent_19](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_19.json) | 386 | 386 | 6 | Holdout | 309 | 77 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_2.json) | 687 | 687 | 6 | Holdout | 550 | 137 |
| [layer_02_parent_20](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_20.json) | 1000 | 999 | 6 | Holdout | 800 | 200 |
| [layer_02_parent_21](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_21.json) | 1040 | 1040 | 6 | Holdout | 832 | 208 |
| [layer_02_parent_22](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_22.json) | 286 | 286 | 8 | Holdout | 229 | 57 |
| [layer_02_parent_23](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_23.json) | 416 | 415 | 8 | Holdout | 333 | 83 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_3.json) | 525 | 525 | 7 | Holdout | 420 | 105 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_4.json) | 645 | 645 | 6 | Holdout | 516 | 129 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_5.json) | 3158 | 3154 | 6 | Holdout | 2526 | 632 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_6.json) | 1421 | 1421 | 6 | Holdout | 1137 | 284 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_7.json) | 1358 | 1356 | 6 | Holdout | 1087 | 271 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_8.json) | 738 | 737 | 6 | Holdout | 591 | 147 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_24_7/layer_02_parent_9.json) | 1333 | 1333 | 6 | Holdout | 1066 | 267 |

</details>

<details>
<summary>layer2_33_4：34分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_01_parent_root.json) | 21907 | 21884 | 33 | Holdout | 17525 | 4382 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_0.json) | 558 | 558 | 4 | Holdout | 446 | 112 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_1.json) | 202 | 202 | 4 | Holdout | 162 | 40 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_10.json) | 544 | 544 | 4 | Holdout | 435 | 109 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_11.json) | 558 | 557 | 4 | Holdout | 447 | 111 |
| [layer_02_parent_12](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_12.json) | 1689 | 1687 | 4 | Holdout | 1352 | 337 |
| [layer_02_parent_13](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_13.json) | 573 | 573 | 4 | Holdout | 458 | 115 |
| [layer_02_parent_14](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_14.json) | 882 | 882 | 4 | Holdout | 706 | 176 |
| [layer_02_parent_15](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_15.json) | 449 | 449 | 4 | Holdout | 359 | 90 |
| [layer_02_parent_16](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_16.json) | 184 | 182 | 4 | Holdout | 148 | 36 |
| [layer_02_parent_17](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_17.json) | 173 | 173 | 4 | Holdout | 138 | 35 |
| [layer_02_parent_18](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_18.json) | 2534 | 2533 | 5 | Holdout | 2026 | 508 |
| [layer_02_parent_19](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_19.json) | 739 | 739 | 4 | Holdout | 591 | 148 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_2.json) | 356 | 356 | 4 | Holdout | 285 | 71 |
| [layer_02_parent_20](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_20.json) | 574 | 572 | 4 | Holdout | 460 | 114 |
| [layer_02_parent_21](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_21.json) | 97 | 97 | 4 | Holdout | 78 | 19 |
| [layer_02_parent_22](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_22.json) | 981 | 981 | 5 | Holdout | 785 | 196 |
| [layer_02_parent_23](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_23.json) | 221 | 221 | 4 | Holdout | 177 | 44 |
| [layer_02_parent_24](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_24.json) | 433 | 433 | 6 | Holdout | 346 | 87 |
| [layer_02_parent_25](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_25.json) | 261 | 261 | 4 | Holdout | 209 | 52 |
| [layer_02_parent_26](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_26.json) | 415 | 414 | 4 | Holdout | 331 | 84 |
| [layer_02_parent_27](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_27.json) | 182 | 182 | 5 | Holdout | 146 | 36 |
| [layer_02_parent_28](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_28.json) | 697 | 697 | 4 | Holdout | 558 | 139 |
| [layer_02_parent_29](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_29.json) | 1186 | 1182 | 4 | Holdout | 946 | 240 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_3.json) | 723 | 723 | 5 | Holdout | 578 | 145 |
| [layer_02_parent_30](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_30.json) | 1014 | 1014 | 4 | Holdout | 811 | 203 |
| [layer_02_parent_31](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_31.json) | 226 | 226 | 4 | Holdout | 181 | 45 |
| [layer_02_parent_32](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_32.json) | 1526 | 1524 | 16 | Holdout | 1220 | 306 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_4.json) | 366 | 366 | 4 | Holdout | 293 | 73 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_5.json) | 650 | 648 | 4 | Holdout | 518 | 132 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_6.json) | 1090 | 1089 | 4 | Holdout | 872 | 218 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_7.json) | 862 | 862 | 6 | Holdout | 690 | 172 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_8.json) | 739 | 739 | 4 | Holdout | 591 | 148 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_33_4/layer_02_parent_9.json) | 1806 | 1804 | 4 | Holdout | 1444 | 362 |

</details>

<details>
<summary>layer2_51_3：52分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_01_parent_root.json) | 21907 | 21884 | 51 | Holdout | 17525 | 4382 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_0.json) | 306 | 306 | 2 | Holdout | 245 | 61 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_1.json) | 252 | 252 | 2 | Holdout | 202 | 50 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_10.json) | 121 | 121 | 2 | Holdout | 97 | 24 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_11.json) | 286 | 286 | 2 | Holdout | 229 | 57 |
| [layer_02_parent_12](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_12.json) | 46 | 46 | 2 | Holdout | 37 | 9 |
| [layer_02_parent_13](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_13.json) | 54 | 54 | 2 | Holdout | 43 | 11 |
| [layer_02_parent_14](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_14.json) | 124 | 124 | 2 | Holdout | 99 | 25 |
| [layer_02_parent_15](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_15.json) | 173 | 172 | 2 | Holdout | 139 | 34 |
| [layer_02_parent_16](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_16.json) | 1570 | 1569 | 2 | Holdout | 1255 | 315 |
| [layer_02_parent_17](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_17.json) | 848 | 847 | 3 | Holdout | 678 | 170 |
| [layer_02_parent_18](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_18.json) | 513 | 511 | 2 | Holdout | 411 | 102 |
| [layer_02_parent_19](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_19.json) | 535 | 535 | 2 | Holdout | 428 | 107 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_2.json) | 434 | 433 | 2 | Holdout | 347 | 87 |
| [layer_02_parent_20](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_20.json) | 460 | 460 | 2 | Holdout | 368 | 92 |
| [layer_02_parent_21](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_21.json) | 1529 | 1527 | 2 | Holdout | 1224 | 305 |
| [layer_02_parent_22](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_22.json) | 1067 | 1067 | 14 | Holdout | 854 | 213 |
| [layer_02_parent_23](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_23.json) | 45 | 45 | 2 | Holdout | 36 | 9 |
| [layer_02_parent_24](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_24.json) | 170 | 170 | 3 | Holdout | 136 | 34 |
| [layer_02_parent_25](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_25.json) | 95 | 95 | 2 | Holdout | 76 | 19 |
| [layer_02_parent_26](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_26.json) | 68 | 68 | 2 | Holdout | 54 | 14 |
| [layer_02_parent_27](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_27.json) | 99 | 99 | 2 | Holdout | 79 | 20 |
| [layer_02_parent_28](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_28.json) | 92 | 90 | 2 | Holdout | 73 | 19 |
| [layer_02_parent_29](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_29.json) | 143 | 143 | 2 | Holdout | 114 | 29 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_3.json) | 150 | 150 | 2 | Holdout | 120 | 30 |
| [layer_02_parent_30](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_30.json) | 510 | 509 | 9 | Holdout | 407 | 103 |
| [layer_02_parent_31](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_31.json) | 380 | 380 | 2 | Holdout | 304 | 76 |
| [layer_02_parent_32](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_32.json) | 140 | 140 | 2 | Holdout | 112 | 28 |
| [layer_02_parent_33](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_33.json) | 2406 | 2405 | 2 | Holdout | 1925 | 481 |
| [layer_02_parent_34](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_34.json) | 126 | 126 | 2 | Holdout | 101 | 25 |
| [layer_02_parent_35](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_35.json) | 96 | 96 | 2 | Holdout | 77 | 19 |
| [layer_02_parent_36](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_36.json) | 218 | 218 | 2 | Holdout | 174 | 44 |
| [layer_02_parent_37](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_37.json) | 995 | 991 | 2 | Holdout | 796 | 199 |
| [layer_02_parent_38](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_38.json) | 188 | 188 | 2 | Holdout | 150 | 38 |
| [layer_02_parent_39](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_39.json) | 18 | 18 | 2 | 5-fold CV | 14–15 | 3–4 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_4.json) | 91 | 91 | 2 | Holdout | 73 | 18 |
| [layer_02_parent_40](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_40.json) | 83 | 83 | 2 | Holdout | 66 | 17 |
| [layer_02_parent_41](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_41.json) | 135 | 135 | 2 | Holdout | 108 | 27 |
| [layer_02_parent_42](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_42.json) | 2118 | 2114 | 14 | Holdout | 1693 | 425 |
| [layer_02_parent_43](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_43.json) | 851 | 851 | 2 | Holdout | 681 | 170 |
| [layer_02_parent_44](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_44.json) | 2575 | 2574 | 13 | Holdout | 2060 | 515 |
| [layer_02_parent_45](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_45.json) | 43 | 43 | 2 | Holdout | 34 | 9 |
| [layer_02_parent_46](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_46.json) | 666 | 666 | 8 | Holdout | 533 | 133 |
| [layer_02_parent_47](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_47.json) | 125 | 125 | 2 | Holdout | 100 | 25 |
| [layer_02_parent_48](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_48.json) | 492 | 492 | 2 | Holdout | 394 | 98 |
| [layer_02_parent_49](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_49.json) | 615 | 615 | 2 | Holdout | 492 | 123 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_5.json) | 237 | 237 | 2 | Holdout | 190 | 47 |
| [layer_02_parent_50](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_50.json) | 240 | 240 | 2 | Holdout | 192 | 48 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_6.json) | 119 | 119 | 2 | Holdout | 95 | 24 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_7.json) | 213 | 213 | 2 | Holdout | 170 | 43 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_8.json) | 217 | 217 | 2 | Holdout | 174 | 43 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer2_51_3/layer_02_parent_9.json) | 417 | 417 | 2 | Holdout | 334 | 83 |

</details>

<details>
<summary>layer3_4_4_8：21分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_01_parent_root.json) | 21907 | 21884 | 4 | Holdout | 17526 | 4381 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_02_parent_0.json) | 6849 | 6845 | 4 | Holdout | 5479 | 1370 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_02_parent_1.json) | 6303 | 6297 | 4 | Holdout | 5043 | 1260 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_02_parent_2.json) | 7180 | 7172 | 4 | Holdout | 5744 | 1436 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_02_parent_3.json) | 2671 | 2669 | 4 | Holdout | 2137 | 534 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_0_0.json) | 817 | 817 | 9 | Holdout | 654 | 163 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_0_1.json) | 1114 | 1114 | 9 | Holdout | 891 | 223 |
| [layer_03_parent_0_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_0_2.json) | 1014 | 1012 | 9 | Holdout | 811 | 203 |
| [layer_03_parent_0_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_0_3.json) | 3952 | 3950 | 11 | Holdout | 3161 | 791 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_1_0.json) | 1854 | 1852 | 9 | Holdout | 1483 | 371 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_1_1.json) | 1411 | 1408 | 9 | Holdout | 1129 | 282 |
| [layer_03_parent_1_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_1_2.json) | 1515 | 1515 | 9 | Holdout | 1212 | 303 |
| [layer_03_parent_1_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_1_3.json) | 1563 | 1562 | 11 | Holdout | 1251 | 312 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_2_0.json) | 3185 | 3182 | 9 | Holdout | 2549 | 636 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_2_1.json) | 703 | 703 | 9 | Holdout | 562 | 141 |
| [layer_03_parent_2_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_2_2.json) | 1966 | 1962 | 9 | Holdout | 1573 | 393 |
| [layer_03_parent_2_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_2_3.json) | 1587 | 1586 | 11 | Holdout | 1269 | 318 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_3_0.json) | 606 | 606 | 9 | Holdout | 485 | 121 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_3_1.json) | 1151 | 1150 | 9 | Holdout | 920 | 231 |
| [layer_03_parent_3_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_3_2.json) | 477 | 477 | 9 | Holdout | 382 | 95 |
| [layer_03_parent_3_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_4_4_8/layer_03_parent_3_3.json) | 447 | 446 | 11 | Holdout | 358 | 89 |

</details>

<details>
<summary>layer3_8_4_4：41分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_01_parent_root.json) | 21907 | 21884 | 8 | Holdout | 17526 | 4381 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_0.json) | 2044 | 2044 | 4 | Holdout | 1635 | 409 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_1.json) | 4252 | 4248 | 4 | Holdout | 3401 | 851 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_2.json) | 4606 | 4603 | 4 | Holdout | 3683 | 923 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_3.json) | 4504 | 4500 | 4 | Holdout | 3602 | 902 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_4.json) | 1913 | 1911 | 4 | Holdout | 1531 | 382 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_5.json) | 3422 | 3417 | 4 | Holdout | 2736 | 686 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_6.json) | 1758 | 1757 | 4 | Holdout | 1407 | 351 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_02_parent_7.json) | 820 | 819 | 4 | Holdout | 656 | 164 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_0_0.json) | 558 | 558 | 4 | Holdout | 446 | 112 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_0_1.json) | 202 | 202 | 4 | Holdout | 162 | 40 |
| [layer_03_parent_0_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_0_2.json) | 345 | 345 | 4 | Holdout | 276 | 69 |
| [layer_03_parent_0_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_0_3.json) | 940 | 940 | 7 | Holdout | 752 | 188 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_1_0.json) | 292 | 292 | 4 | Holdout | 234 | 58 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_1_1.json) | 1584 | 1581 | 4 | Holdout | 1268 | 316 |
| [layer_03_parent_1_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_1_2.json) | 257 | 257 | 4 | Holdout | 206 | 51 |
| [layer_03_parent_1_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_1_3.json) | 2129 | 2128 | 7 | Holdout | 1703 | 426 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_2_0.json) | 1056 | 1056 | 4 | Holdout | 845 | 211 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_2_1.json) | 1434 | 1432 | 4 | Holdout | 1147 | 287 |
| [layer_03_parent_2_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_2_2.json) | 963 | 962 | 4 | Holdout | 770 | 193 |
| [layer_03_parent_2_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_2_3.json) | 1208 | 1208 | 7 | Holdout | 966 | 242 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_3_0.json) | 829 | 829 | 4 | Holdout | 663 | 166 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_3_1.json) | 487 | 486 | 4 | Holdout | 389 | 98 |
| [layer_03_parent_3_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_3_2.json) | 131 | 131 | 4 | Holdout | 105 | 26 |
| [layer_03_parent_3_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_3_3.json) | 3074 | 3071 | 7 | Holdout | 2459 | 615 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_4_0.json) | 294 | 294 | 4 | Holdout | 235 | 59 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_4_1.json) | 648 | 646 | 5 | Holdout | 518 | 130 |
| [layer_03_parent_4_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_4_2.json) | 360 | 360 | 4 | Holdout | 288 | 72 |
| [layer_03_parent_4_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_4_3.json) | 623 | 623 | 6 | Holdout | 498 | 125 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_5_0.json) | 800 | 796 | 4 | Holdout | 641 | 159 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_5_1.json) | 1122 | 1122 | 4 | Holdout | 898 | 224 |
| [layer_03_parent_5_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_5_2.json) | 174 | 174 | 4 | Holdout | 139 | 35 |
| [layer_03_parent_5_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_5_3.json) | 1358 | 1357 | 7 | Holdout | 1087 | 271 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_6_0.json) | 95 | 95 | 4 | Holdout | 76 | 19 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_6_1.json) | 338 | 338 | 4 | Holdout | 270 | 68 |
| [layer_03_parent_6_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_6_2.json) | 419 | 419 | 4 | Holdout | 335 | 84 |
| [layer_03_parent_6_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_6_3.json) | 918 | 917 | 7 | Holdout | 735 | 183 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_7_0.json) | 223 | 223 | 4 | Holdout | 178 | 45 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_7_1.json) | 150 | 150 | 4 | Holdout | 120 | 30 |
| [layer_03_parent_7_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_7_2.json) | 170 | 170 | 4 | Holdout | 136 | 34 |
| [layer_03_parent_7_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_8_4_4/layer_03_parent_7_3.json) | 277 | 276 | 7 | Holdout | 221 | 56 |

</details>

<details>
<summary>layer3_12_2_6：37分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_01_parent_root.json) | 21907 | 21884 | 12 | Holdout | 17524 | 4383 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_0.json) | 1116 | 1116 | 2 | Holdout | 893 | 223 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_1.json) | 1299 | 1299 | 2 | Holdout | 1039 | 260 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_10.json) | 1327 | 1326 | 2 | Holdout | 1062 | 265 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_11.json) | 636 | 635 | 2 | Holdout | 509 | 127 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_2.json) | 3879 | 3875 | 2 | Holdout | 3103 | 776 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_3.json) | 2632 | 2630 | 2 | Holdout | 2105 | 527 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_4.json) | 2441 | 2438 | 2 | Holdout | 1953 | 488 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_5.json) | 3343 | 3341 | 2 | Holdout | 2675 | 668 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_6.json) | 1586 | 1584 | 2 | Holdout | 1269 | 317 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_7.json) | 1209 | 1209 | 2 | Holdout | 967 | 242 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_8.json) | 1969 | 1965 | 2 | Holdout | 1573 | 396 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_02_parent_9.json) | 1977 | 1976 | 2 | Holdout | 1582 | 395 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_0_0.json) | 615 | 615 | 6 | Holdout | 492 | 123 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_0_1.json) | 501 | 501 | 6 | Holdout | 401 | 100 |
| [layer_03_parent_10_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_10_0.json) | 1186 | 1185 | 7 | Holdout | 949 | 237 |
| [layer_03_parent_10_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_10_1.json) | 142 | 142 | 7 | Holdout | 114 | 28 |
| [layer_03_parent_11_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_11_0.json) | 193 | 193 | 6 | Holdout | 154 | 39 |
| [layer_03_parent_11_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_11_1.json) | 443 | 442 | 7 | Holdout | 355 | 88 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_1_0.json) | 685 | 685 | 6 | Holdout | 548 | 137 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_1_1.json) | 614 | 614 | 6 | Holdout | 491 | 123 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_2_0.json) | 1885 | 1882 | 6 | Holdout | 1508 | 377 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_2_1.json) | 2002 | 2001 | 6 | Holdout | 1602 | 400 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_3_0.json) | 983 | 983 | 6 | Holdout | 786 | 197 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_3_1.json) | 1670 | 1668 | 6 | Holdout | 1336 | 334 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_4_0.json) | 778 | 775 | 6 | Holdout | 623 | 155 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_4_1.json) | 1667 | 1667 | 6 | Holdout | 1334 | 333 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_5_0.json) | 688 | 687 | 6 | Holdout | 550 | 138 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_5_1.json) | 2662 | 2661 | 6 | Holdout | 2130 | 532 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_6_0.json) | 734 | 734 | 7 | Holdout | 587 | 147 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_6_1.json) | 854 | 852 | 6 | Holdout | 682 | 172 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_7_0.json) | 833 | 833 | 7 | Holdout | 666 | 167 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_7_1.json) | 379 | 379 | 6 | Holdout | 303 | 76 |
| [layer_03_parent_8_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_8_0.json) | 766 | 766 | 7 | Holdout | 613 | 153 |
| [layer_03_parent_8_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_8_1.json) | 1213 | 1209 | 6 | Holdout | 971 | 242 |
| [layer_03_parent_9_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_9_0.json) | 600 | 599 | 7 | Holdout | 479 | 121 |
| [layer_03_parent_9_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_2_6/layer_03_parent_9_1.json) | 1382 | 1382 | 7 | Holdout | 1106 | 276 |

</details>

<details>
<summary>layer3_12_4_3：61分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_01_parent_root.json) | 21907 | 21884 | 12 | Holdout | 17524 | 4383 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_0.json) | 1116 | 1116 | 4 | Holdout | 893 | 223 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_1.json) | 1299 | 1299 | 4 | Holdout | 1039 | 260 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_10.json) | 1327 | 1326 | 4 | Holdout | 1062 | 265 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_11.json) | 636 | 635 | 4 | Holdout | 509 | 127 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_2.json) | 3879 | 3875 | 4 | Holdout | 3103 | 776 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_3.json) | 2632 | 2630 | 4 | Holdout | 2105 | 527 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_4.json) | 2441 | 2438 | 4 | Holdout | 1953 | 488 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_5.json) | 3343 | 3341 | 4 | Holdout | 2675 | 668 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_6.json) | 1586 | 1584 | 4 | Holdout | 1269 | 317 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_7.json) | 1209 | 1209 | 4 | Holdout | 967 | 242 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_8.json) | 1969 | 1965 | 4 | Holdout | 1572 | 397 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_02_parent_9.json) | 1977 | 1976 | 4 | Holdout | 1582 | 395 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_0_0.json) | 366 | 366 | 3 | Holdout | 293 | 73 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_0_1.json) | 336 | 336 | 3 | Holdout | 269 | 67 |
| [layer_03_parent_0_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_0_2.json) | 115 | 115 | 3 | Holdout | 92 | 23 |
| [layer_03_parent_0_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_0_3.json) | 299 | 299 | 3 | Holdout | 239 | 60 |
| [layer_03_parent_10_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_10_0.json) | 66 | 66 | 3 | Holdout | 53 | 13 |
| [layer_03_parent_10_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_10_1.json) | 412 | 412 | 5 | Holdout | 330 | 82 |
| [layer_03_parent_10_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_10_2.json) | 692 | 691 | 3 | Holdout | 554 | 138 |
| [layer_03_parent_10_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_10_3.json) | 166 | 166 | 3 | Holdout | 133 | 33 |
| [layer_03_parent_11_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_11_0.json) | 116 | 116 | 3 | Holdout | 93 | 23 |
| [layer_03_parent_11_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_11_1.json) | 111 | 111 | 3 | Holdout | 89 | 22 |
| [layer_03_parent_11_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_11_2.json) | 235 | 234 | 3 | Holdout | 187 | 48 |
| [layer_03_parent_11_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_11_3.json) | 174 | 174 | 4 | Holdout | 139 | 35 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_1_0.json) | 433 | 433 | 3 | Holdout | 346 | 87 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_1_1.json) | 220 | 220 | 3 | Holdout | 176 | 44 |
| [layer_03_parent_1_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_1_2.json) | 315 | 315 | 3 | Holdout | 252 | 63 |
| [layer_03_parent_1_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_1_3.json) | 332 | 332 | 3 | Holdout | 266 | 66 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_2_0.json) | 1355 | 1354 | 3 | Holdout | 1084 | 271 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_2_1.json) | 1137 | 1136 | 3 | Holdout | 910 | 227 |
| [layer_03_parent_2_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_2_2.json) | 762 | 760 | 3 | Holdout | 609 | 153 |
| [layer_03_parent_2_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_2_3.json) | 632 | 632 | 3 | Holdout | 506 | 126 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_3_0.json) | 1046 | 1046 | 3 | Holdout | 837 | 209 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_3_1.json) | 995 | 993 | 3 | Holdout | 796 | 199 |
| [layer_03_parent_3_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_3_2.json) | 74 | 74 | 3 | Holdout | 59 | 15 |
| [layer_03_parent_3_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_3_3.json) | 539 | 539 | 3 | Holdout | 431 | 108 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_4_0.json) | 493 | 492 | 3 | Holdout | 395 | 98 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_4_1.json) | 969 | 969 | 3 | Holdout | 775 | 194 |
| [layer_03_parent_4_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_4_2.json) | 464 | 464 | 3 | Holdout | 371 | 93 |
| [layer_03_parent_4_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_4_3.json) | 519 | 517 | 3 | Holdout | 415 | 104 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_5_0.json) | 242 | 242 | 3 | Holdout | 194 | 48 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_5_1.json) | 463 | 463 | 3 | Holdout | 370 | 93 |
| [layer_03_parent_5_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_5_2.json) | 2245 | 2244 | 3 | Holdout | 1796 | 449 |
| [layer_03_parent_5_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_5_3.json) | 420 | 419 | 3 | Holdout | 336 | 84 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_6_0.json) | 533 | 531 | 3 | Holdout | 427 | 106 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_6_1.json) | 52 | 52 | 3 | Holdout | 42 | 10 |
| [layer_03_parent_6_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_6_2.json) | 783 | 783 | 4 | Holdout | 626 | 157 |
| [layer_03_parent_6_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_6_3.json) | 220 | 220 | 3 | Holdout | 176 | 44 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_7_0.json) | 356 | 356 | 3 | Holdout | 285 | 71 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_7_1.json) | 107 | 107 | 3 | Holdout | 86 | 21 |
| [layer_03_parent_7_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_7_2.json) | 571 | 571 | 3 | Holdout | 457 | 114 |
| [layer_03_parent_7_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_7_3.json) | 176 | 176 | 4 | Holdout | 141 | 35 |
| [layer_03_parent_8_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_8_0.json) | 407 | 407 | 3 | Holdout | 326 | 81 |
| [layer_03_parent_8_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_8_1.json) | 1259 | 1255 | 3 | Holdout | 1008 | 251 |
| [layer_03_parent_8_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_8_2.json) | 173 | 173 | 4 | Holdout | 138 | 35 |
| [layer_03_parent_8_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_8_3.json) | 142 | 142 | 3 | Holdout | 114 | 28 |
| [layer_03_parent_9_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_9_0.json) | 240 | 240 | 4 | Holdout | 192 | 48 |
| [layer_03_parent_9_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_9_1.json) | 277 | 277 | 4 | Holdout | 222 | 55 |
| [layer_03_parent_9_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_9_2.json) | 261 | 260 | 3 | Holdout | 209 | 52 |
| [layer_03_parent_9_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_4_3/layer_03_parent_9_3.json) | 1207 | 1207 | 3 | Holdout | 966 | 241 |

</details>

<details>
<summary>layer3_12_6_2：85分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_01_parent_root.json) | 21907 | 21884 | 12 | Holdout | 17524 | 4383 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_0.json) | 1116 | 1116 | 6 | Holdout | 893 | 223 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_1.json) | 1299 | 1299 | 6 | Holdout | 1039 | 260 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_10.json) | 1327 | 1326 | 6 | Holdout | 1062 | 265 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_11.json) | 636 | 635 | 6 | Holdout | 509 | 127 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_2.json) | 3879 | 3875 | 6 | Holdout | 3103 | 776 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_3.json) | 2632 | 2630 | 6 | Holdout | 2106 | 526 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_4.json) | 2441 | 2438 | 6 | Holdout | 1953 | 488 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_5.json) | 3343 | 3341 | 6 | Holdout | 2675 | 668 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_6.json) | 1586 | 1584 | 6 | Holdout | 1269 | 317 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_7.json) | 1209 | 1209 | 6 | Holdout | 967 | 242 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_8.json) | 1969 | 1965 | 6 | Holdout | 1576 | 393 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_02_parent_9.json) | 1977 | 1976 | 6 | Holdout | 1582 | 395 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_0_0.json) | 301 | 301 | 2 | Holdout | 241 | 60 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_0_1.json) | 257 | 257 | 2 | Holdout | 206 | 51 |
| [layer_03_parent_0_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_0_2.json) | 82 | 82 | 2 | Holdout | 66 | 16 |
| [layer_03_parent_0_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_0_3.json) | 254 | 254 | 2 | Holdout | 203 | 51 |
| [layer_03_parent_0_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_0_4.json) | 77 | 77 | 2 | Holdout | 62 | 15 |
| [layer_03_parent_0_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_0_5.json) | 145 | 145 | 2 | Holdout | 116 | 29 |
| [layer_03_parent_10_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_10_0.json) | 778 | 777 | 2 | Holdout | 622 | 156 |
| [layer_03_parent_10_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_10_1.json) | 26 | 26 | 2 | Holdout | 21 | 5 |
| [layer_03_parent_10_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_10_2.json) | 27 | 27 | 2 | Holdout | 22 | 5 |
| [layer_03_parent_10_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_10_3.json) | 301 | 301 | 2 | Holdout | 241 | 60 |
| [layer_03_parent_10_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_10_4.json) | 173 | 173 | 4 | Holdout | 138 | 35 |
| [layer_03_parent_10_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_10_5.json) | 31 | 31 | 2 | Holdout | 25 | 6 |
| [layer_03_parent_11_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_11_0.json) | 93 | 93 | 2 | Holdout | 74 | 19 |
| [layer_03_parent_11_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_11_1.json) | 91 | 91 | 2 | Holdout | 73 | 18 |
| [layer_03_parent_11_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_11_2.json) | 191 | 191 | 2 | Holdout | 153 | 38 |
| [layer_03_parent_11_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_11_3.json) | 157 | 157 | 3 | Holdout | 126 | 31 |
| [layer_03_parent_11_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_11_4.json) | 49 | 48 | 2 | Holdout | 38 | 11 |
| [layer_03_parent_11_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_11_5.json) | 55 | 55 | 2 | Holdout | 44 | 11 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_1_0.json) | 199 | 199 | 2 | Holdout | 159 | 40 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_1_1.json) | 238 | 238 | 2 | Holdout | 190 | 48 |
| [layer_03_parent_1_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_1_2.json) | 256 | 256 | 2 | Holdout | 205 | 51 |
| [layer_03_parent_1_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_1_3.json) | 247 | 247 | 2 | Holdout | 198 | 49 |
| [layer_03_parent_1_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_1_4.json) | 116 | 116 | 2 | Holdout | 93 | 23 |
| [layer_03_parent_1_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_1_5.json) | 244 | 244 | 2 | Holdout | 195 | 49 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_2_0.json) | 494 | 492 | 2 | Holdout | 396 | 98 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_2_1.json) | 540 | 540 | 2 | Holdout | 432 | 108 |
| [layer_03_parent_2_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_2_2.json) | 132 | 132 | 2 | Holdout | 106 | 26 |
| [layer_03_parent_2_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_2_3.json) | 71 | 71 | 2 | Holdout | 57 | 14 |
| [layer_03_parent_2_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_2_4.json) | 1360 | 1359 | 2 | Holdout | 1088 | 272 |
| [layer_03_parent_2_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_2_5.json) | 1290 | 1289 | 2 | Holdout | 1032 | 258 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_3_0.json) | 40 | 40 | 2 | Holdout | 32 | 8 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_3_1.json) | 611 | 611 | 2 | Holdout | 489 | 122 |
| [layer_03_parent_3_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_3_2.json) | 1356 | 1354 | 2 | Holdout | 1085 | 271 |
| [layer_03_parent_3_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_3_3.json) | 141 | 141 | 2 | Holdout | 113 | 28 |
| [layer_03_parent_3_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_3_4.json) | 175 | 175 | 2 | Holdout | 140 | 35 |
| [layer_03_parent_3_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_3_5.json) | 377 | 377 | 2 | Holdout | 302 | 75 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_4_0.json) | 426 | 426 | 2 | Holdout | 341 | 85 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_4_1.json) | 451 | 451 | 2 | Holdout | 361 | 90 |
| [layer_03_parent_4_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_4_2.json) | 260 | 259 | 2 | Holdout | 208 | 52 |
| [layer_03_parent_4_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_4_3.json) | 366 | 366 | 2 | Holdout | 293 | 73 |
| [layer_03_parent_4_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_4_4.json) | 801 | 801 | 2 | Holdout | 641 | 160 |
| [layer_03_parent_4_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_4_5.json) | 141 | 139 | 2 | Holdout | 112 | 29 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_0.json) | 235 | 235 | 2 | Holdout | 188 | 47 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_1.json) | 156 | 156 | 2 | Holdout | 125 | 31 |
| [layer_03_parent_5_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_2.json) | 334 | 334 | 2 | Holdout | 267 | 67 |
| [layer_03_parent_5_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_3.json) | 12 | 12 | 2 | 5-fold CV | 9–10 | 2–3 |
| [layer_03_parent_5_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_4.json) | 2248 | 2247 | 2 | Holdout | 1799 | 449 |
| [layer_03_parent_5_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_5_5.json) | 386 | 385 | 2 | Holdout | 308 | 78 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_6_0.json) | 537 | 535 | 2 | Holdout | 430 | 107 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_6_1.json) | 458 | 458 | 2 | Holdout | 366 | 92 |
| [layer_03_parent_6_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_6_2.json) | 78 | 78 | 2 | Holdout | 62 | 16 |
| [layer_03_parent_6_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_6_3.json) | 180 | 180 | 2 | Holdout | 144 | 36 |
| [layer_03_parent_6_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_6_4.json) | 258 | 258 | 3 | Holdout | 206 | 52 |
| [layer_03_parent_6_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_6_5.json) | 76 | 76 | 2 | Holdout | 61 | 15 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_7_0.json) | 398 | 398 | 3 | Holdout | 318 | 80 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_7_1.json) | 89 | 89 | 2 | Holdout | 71 | 18 |
| [layer_03_parent_7_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_7_2.json) | 353 | 353 | 2 | Holdout | 282 | 71 |
| [layer_03_parent_7_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_7_3.json) | 104 | 104 | 2 | Holdout | 83 | 21 |
| [layer_03_parent_7_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_7_4.json) | 53 | 53 | 2 | Holdout | 42 | 11 |
| [layer_03_parent_7_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_7_5.json) | 216 | 216 | 2 | Holdout | 173 | 43 |
| [layer_03_parent_8_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_8_0.json) | 69 | 69 | 2 | Holdout | 55 | 14 |
| [layer_03_parent_8_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_8_1.json) | 124 | 124 | 3 | Holdout | 99 | 25 |
| [layer_03_parent_8_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_8_2.json) | 1041 | 1037 | 2 | Holdout | 834 | 207 |
| [layer_03_parent_8_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_8_3.json) | 596 | 596 | 2 | Holdout | 477 | 119 |
| [layer_03_parent_8_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_8_4.json) | 25 | 25 | 2 | Holdout | 20 | 5 |
| [layer_03_parent_8_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_8_5.json) | 124 | 124 | 2 | Holdout | 99 | 25 |
| [layer_03_parent_9_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_9_0.json) | 236 | 235 | 2 | Holdout | 189 | 47 |
| [layer_03_parent_9_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_9_1.json) | 144 | 144 | 2 | Holdout | 115 | 29 |
| [layer_03_parent_9_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_9_2.json) | 1046 | 1046 | 4 | Holdout | 837 | 209 |
| [layer_03_parent_9_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_9_3.json) | 43 | 43 | 2 | Holdout | 34 | 9 |
| [layer_03_parent_9_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_9_4.json) | 316 | 316 | 2 | Holdout | 253 | 63 |
| [layer_03_parent_9_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_12_6_2/layer_03_parent_9_5.json) | 200 | 200 | 2 | Holdout | 160 | 40 |

</details>

<details>
<summary>layer3_16_4_3：81分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_01_parent_root.json) | 21907 | 21884 | 16 | Holdout | 17527 | 4380 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_0.json) | 864 | 864 | 4 | Holdout | 691 | 173 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_1.json) | 1114 | 1114 | 4 | Holdout | 891 | 223 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_10.json) | 1214 | 1210 | 4 | Holdout | 972 | 242 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_11.json) | 1371 | 1371 | 4 | Holdout | 1097 | 274 |
| [layer_02_parent_12](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_12.json) | 690 | 689 | 4 | Holdout | 552 | 138 |
| [layer_02_parent_13](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_13.json) | 2062 | 2061 | 4 | Holdout | 1649 | 413 |
| [layer_02_parent_14](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_14.json) | 604 | 604 | 4 | Holdout | 483 | 121 |
| [layer_02_parent_15](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_15.json) | 567 | 566 | 4 | Holdout | 454 | 113 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_2.json) | 1014 | 1012 | 4 | Holdout | 812 | 202 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_3.json) | 3755 | 3753 | 4 | Holdout | 3004 | 751 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_4.json) | 1379 | 1379 | 4 | Holdout | 1103 | 276 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_5.json) | 1398 | 1396 | 4 | Holdout | 1119 | 279 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_6.json) | 4598 | 4593 | 4 | Holdout | 3677 | 921 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_7.json) | 1187 | 1185 | 4 | Holdout | 950 | 237 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_8.json) | 453 | 453 | 4 | Holdout | 362 | 91 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_02_parent_9.json) | 1197 | 1197 | 4 | Holdout | 958 | 239 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_0_0.json) | 301 | 301 | 2 | Holdout | 241 | 60 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_0_1.json) | 304 | 304 | 3 | Holdout | 243 | 61 |
| [layer_03_parent_0_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_0_2.json) | 103 | 103 | 2 | Holdout | 82 | 21 |
| [layer_03_parent_0_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_0_3.json) | 156 | 156 | 3 | Holdout | 125 | 31 |
| [layer_03_parent_10_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_10_0.json) | 148 | 148 | 3 | Holdout | 118 | 30 |
| [layer_03_parent_10_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_10_1.json) | 125 | 125 | 2 | Holdout | 100 | 25 |
| [layer_03_parent_10_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_10_2.json) | 731 | 727 | 2 | Holdout | 586 | 145 |
| [layer_03_parent_10_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_10_3.json) | 213 | 213 | 2 | Holdout | 170 | 43 |
| [layer_03_parent_11_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_11_0.json) | 334 | 334 | 3 | Holdout | 267 | 67 |
| [layer_03_parent_11_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_11_1.json) | 50 | 50 | 2 | Holdout | 40 | 10 |
| [layer_03_parent_11_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_11_2.json) | 389 | 389 | 2 | Holdout | 311 | 78 |
| [layer_03_parent_11_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_11_3.json) | 602 | 602 | 2 | Holdout | 482 | 120 |
| [layer_03_parent_12_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_12_0.json) | 407 | 406 | 4 | Holdout | 325 | 82 |
| [layer_03_parent_12_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_12_1.json) | 87 | 87 | 2 | Holdout | 70 | 17 |
| [layer_03_parent_12_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_12_2.json) | 123 | 123 | 2 | Holdout | 98 | 25 |
| [layer_03_parent_12_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_12_3.json) | 73 | 73 | 2 | Holdout | 58 | 15 |
| [layer_03_parent_13_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_13_0.json) | 80 | 80 | 2 | Holdout | 64 | 16 |
| [layer_03_parent_13_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_13_1.json) | 148 | 148 | 2 | Holdout | 118 | 30 |
| [layer_03_parent_13_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_13_2.json) | 903 | 902 | 2 | Holdout | 722 | 181 |
| [layer_03_parent_13_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_13_3.json) | 933 | 933 | 3 | Holdout | 746 | 187 |
| [layer_03_parent_14_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_14_0.json) | 127 | 127 | 4 | Holdout | 102 | 25 |
| [layer_03_parent_14_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_14_1.json) | 287 | 287 | 2 | Holdout | 230 | 57 |
| [layer_03_parent_14_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_14_2.json) | 99 | 99 | 2 | Holdout | 79 | 20 |
| [layer_03_parent_14_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_14_3.json) | 91 | 91 | 2 | Holdout | 73 | 18 |
| [layer_03_parent_15_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_15_0.json) | 79 | 79 | 3 | Holdout | 63 | 16 |
| [layer_03_parent_15_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_15_1.json) | 223 | 222 | 3 | Holdout | 179 | 44 |
| [layer_03_parent_15_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_15_2.json) | 103 | 103 | 3 | Holdout | 82 | 21 |
| [layer_03_parent_15_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_15_3.json) | 162 | 162 | 3 | Holdout | 130 | 32 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_1_0.json) | 125 | 125 | 2 | Holdout | 100 | 25 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_1_1.json) | 190 | 190 | 2 | Holdout | 152 | 38 |
| [layer_03_parent_1_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_1_2.json) | 547 | 547 | 3 | Holdout | 438 | 109 |
| [layer_03_parent_1_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_1_3.json) | 253 | 253 | 2 | Holdout | 202 | 51 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_2_0.json) | 91 | 91 | 2 | Holdout | 73 | 18 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_2_1.json) | 544 | 542 | 3 | Holdout | 435 | 109 |
| [layer_03_parent_2_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_2_2.json) | 153 | 153 | 2 | Holdout | 122 | 31 |
| [layer_03_parent_2_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_2_3.json) | 226 | 226 | 2 | Holdout | 181 | 45 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_3_0.json) | 977 | 977 | 3 | Holdout | 782 | 195 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_3_1.json) | 1516 | 1515 | 4 | Holdout | 1213 | 303 |
| [layer_03_parent_3_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_3_2.json) | 113 | 113 | 2 | Holdout | 90 | 23 |
| [layer_03_parent_3_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_3_3.json) | 1164 | 1163 | 2 | Holdout | 930 | 234 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_4_0.json) | 461 | 461 | 2 | Holdout | 369 | 92 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_4_1.json) | 448 | 448 | 2 | Holdout | 358 | 90 |
| [layer_03_parent_4_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_4_2.json) | 54 | 54 | 2 | Holdout | 43 | 11 |
| [layer_03_parent_4_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_4_3.json) | 420 | 420 | 3 | Holdout | 336 | 84 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_5_0.json) | 262 | 261 | 2 | Holdout | 210 | 52 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_5_1.json) | 55 | 55 | 2 | Holdout | 44 | 11 |
| [layer_03_parent_5_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_5_2.json) | 927 | 926 | 3 | Holdout | 741 | 186 |
| [layer_03_parent_5_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_5_3.json) | 160 | 160 | 2 | Holdout | 128 | 32 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_6_0.json) | 902 | 902 | 2 | Holdout | 722 | 180 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_6_1.json) | 1022 | 1020 | 2 | Holdout | 818 | 204 |
| [layer_03_parent_6_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_6_2.json) | 2248 | 2247 | 2 | Holdout | 1799 | 449 |
| [layer_03_parent_6_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_6_3.json) | 482 | 480 | 3 | Holdout | 386 | 96 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_7_0.json) | 162 | 162 | 2 | Holdout | 130 | 32 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_7_1.json) | 332 | 332 | 2 | Holdout | 266 | 66 |
| [layer_03_parent_7_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_7_2.json) | 577 | 575 | 2 | Holdout | 462 | 115 |
| [layer_03_parent_7_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_7_3.json) | 118 | 118 | 3 | Holdout | 94 | 24 |
| [layer_03_parent_8_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_8_0.json) | 221 | 221 | 3 | Holdout | 177 | 44 |
| [layer_03_parent_8_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_8_1.json) | 121 | 121 | 2 | Holdout | 97 | 24 |
| [layer_03_parent_8_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_8_2.json) | 84 | 84 | 2 | Holdout | 67 | 17 |
| [layer_03_parent_8_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_8_3.json) | 27 | 27 | 2 | Holdout | 22 | 5 |
| [layer_03_parent_9_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_9_0.json) | 388 | 388 | 2 | Holdout | 310 | 78 |
| [layer_03_parent_9_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_9_1.json) | 457 | 457 | 2 | Holdout | 366 | 91 |
| [layer_03_parent_9_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_9_2.json) | 47 | 47 | 2 | Holdout | 38 | 9 |
| [layer_03_parent_9_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer3_16_4_3/layer_03_parent_9_3.json) | 306 | 306 | 3 | Holdout | 245 | 61 |

</details>

<details>
<summary>layer4_8_3_2_3：81分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_01_parent_root.json) | 21907 | 21884 | 8 | Holdout | 17526 | 4381 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_0.json) | 2044 | 2044 | 3 | Holdout | 1635 | 409 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_1.json) | 4252 | 4248 | 3 | Holdout | 3401 | 851 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_2.json) | 4606 | 4603 | 3 | Holdout | 3684 | 922 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_3.json) | 4504 | 4500 | 3 | Holdout | 3602 | 902 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_4.json) | 1913 | 1911 | 3 | Holdout | 1531 | 382 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_5.json) | 3422 | 3417 | 3 | Holdout | 2735 | 687 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_6.json) | 1758 | 1757 | 3 | Holdout | 1407 | 351 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_02_parent_7.json) | 820 | 819 | 3 | Holdout | 656 | 164 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_0_0.json) | 703 | 703 | 2 | Holdout | 562 | 141 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_0_1.json) | 566 | 566 | 2 | Holdout | 453 | 113 |
| [layer_03_parent_0_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_0_2.json) | 776 | 776 | 2 | Holdout | 621 | 155 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_1_0.json) | 411 | 411 | 2 | Holdout | 329 | 82 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_1_1.json) | 2101 | 2098 | 2 | Holdout | 1680 | 421 |
| [layer_03_parent_1_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_1_2.json) | 1750 | 1749 | 2 | Holdout | 1400 | 350 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_2_0.json) | 1110 | 1110 | 2 | Holdout | 888 | 222 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_2_1.json) | 1797 | 1796 | 2 | Holdout | 1438 | 359 |
| [layer_03_parent_2_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_2_2.json) | 1748 | 1746 | 2 | Holdout | 1399 | 349 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_3_0.json) | 959 | 959 | 2 | Holdout | 767 | 192 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_3_1.json) | 971 | 970 | 2 | Holdout | 777 | 194 |
| [layer_03_parent_3_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_3_2.json) | 2593 | 2590 | 2 | Holdout | 2074 | 519 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_4_0.json) | 463 | 463 | 2 | Holdout | 370 | 93 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_4_1.json) | 894 | 892 | 2 | Holdout | 716 | 178 |
| [layer_03_parent_4_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_4_2.json) | 564 | 564 | 2 | Holdout | 451 | 113 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_5_0.json) | 1114 | 1109 | 2 | Holdout | 892 | 222 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_5_1.json) | 1181 | 1181 | 2 | Holdout | 945 | 236 |
| [layer_03_parent_5_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_5_2.json) | 1159 | 1159 | 2 | Holdout | 927 | 232 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_6_0.json) | 354 | 354 | 2 | Holdout | 283 | 71 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_6_1.json) | 909 | 908 | 2 | Holdout | 726 | 183 |
| [layer_03_parent_6_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_6_2.json) | 507 | 507 | 2 | Holdout | 406 | 101 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_7_0.json) | 202 | 201 | 2 | Holdout | 162 | 40 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_7_1.json) | 341 | 341 | 2 | Holdout | 273 | 68 |
| [layer_03_parent_7_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_03_parent_7_2.json) | 277 | 277 | 2 | Holdout | 222 | 55 |
| [layer_04_parent_0_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_0_0_0.json) | 366 | 366 | 3 | Holdout | 293 | 73 |
| [layer_04_parent_0_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_0_0_1.json) | 337 | 337 | 3 | Holdout | 270 | 67 |
| [layer_04_parent_0_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_0_1_0.json) | 241 | 241 | 3 | Holdout | 193 | 48 |
| [layer_04_parent_0_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_0_1_1.json) | 325 | 325 | 3 | Holdout | 260 | 65 |
| [layer_04_parent_0_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_0_2_0.json) | 525 | 525 | 4 | Holdout | 420 | 105 |
| [layer_04_parent_0_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_0_2_1.json) | 251 | 251 | 3 | Holdout | 201 | 50 |
| [layer_04_parent_1_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_1_0_0.json) | 289 | 289 | 3 | Holdout | 231 | 58 |
| [layer_04_parent_1_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_1_0_1.json) | 122 | 122 | 3 | Holdout | 98 | 24 |
| [layer_04_parent_1_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_1_1_0.json) | 500 | 498 | 3 | Holdout | 399 | 101 |
| [layer_04_parent_1_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_1_1_1.json) | 1601 | 1600 | 3 | Holdout | 1281 | 320 |
| [layer_04_parent_1_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_1_2_0.json) | 488 | 488 | 4 | Holdout | 390 | 98 |
| [layer_04_parent_1_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_1_2_1.json) | 1268 | 1267 | 3 | Holdout | 1015 | 253 |
| [layer_04_parent_2_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_2_0_0.json) | 502 | 502 | 3 | Holdout | 402 | 100 |
| [layer_04_parent_2_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_2_0_1.json) | 655 | 655 | 3 | Holdout | 524 | 131 |
| [layer_04_parent_2_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_2_1_0.json) | 561 | 560 | 3 | Holdout | 449 | 112 |
| [layer_04_parent_2_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_2_1_1.json) | 1241 | 1241 | 3 | Holdout | 993 | 248 |
| [layer_04_parent_2_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_2_2_0.json) | 1109 | 1107 | 3 | Holdout | 887 | 222 |
| [layer_04_parent_2_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_2_2_1.json) | 639 | 639 | 4 | Holdout | 511 | 128 |
| [layer_04_parent_3_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_3_0_0.json) | 787 | 787 | 3 | Holdout | 630 | 157 |
| [layer_04_parent_3_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_3_0_1.json) | 172 | 172 | 3 | Holdout | 138 | 34 |
| [layer_04_parent_3_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_3_1_0.json) | 331 | 330 | 3 | Holdout | 264 | 67 |
| [layer_04_parent_3_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_3_1_1.json) | 640 | 640 | 3 | Holdout | 512 | 128 |
| [layer_04_parent_3_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_3_2_0.json) | 2217 | 2216 | 4 | Holdout | 1774 | 443 |
| [layer_04_parent_3_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_3_2_1.json) | 411 | 409 | 3 | Holdout | 329 | 82 |
| [layer_04_parent_4_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_4_0_0.json) | 179 | 179 | 3 | Holdout | 143 | 36 |
| [layer_04_parent_4_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_4_0_1.json) | 285 | 285 | 4 | Holdout | 228 | 57 |
| [layer_04_parent_4_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_4_1_0.json) | 634 | 632 | 3 | Holdout | 507 | 127 |
| [layer_04_parent_4_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_4_1_1.json) | 264 | 264 | 3 | Holdout | 211 | 53 |
| [layer_04_parent_4_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_4_2_0.json) | 364 | 364 | 3 | Holdout | 291 | 73 |
| [layer_04_parent_4_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_4_2_1.json) | 200 | 200 | 3 | Holdout | 160 | 40 |
| [layer_04_parent_5_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_5_0_0.json) | 908 | 903 | 3 | Holdout | 723 | 185 |
| [layer_04_parent_5_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_5_0_1.json) | 207 | 207 | 3 | Holdout | 166 | 41 |
| [layer_04_parent_5_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_5_1_0.json) | 996 | 996 | 3 | Holdout | 797 | 199 |
| [layer_04_parent_5_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_5_1_1.json) | 188 | 188 | 3 | Holdout | 150 | 38 |
| [layer_04_parent_5_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_5_2_0.json) | 109 | 109 | 3 | Holdout | 87 | 22 |
| [layer_04_parent_5_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_5_2_1.json) | 1050 | 1050 | 4 | Holdout | 840 | 210 |
| [layer_04_parent_6_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_6_0_0.json) | 76 | 76 | 3 | Holdout | 61 | 15 |
| [layer_04_parent_6_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_6_0_1.json) | 278 | 278 | 3 | Holdout | 222 | 56 |
| [layer_04_parent_6_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_6_1_0.json) | 785 | 784 | 3 | Holdout | 628 | 157 |
| [layer_04_parent_6_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_6_1_1.json) | 125 | 125 | 3 | Holdout | 100 | 25 |
| [layer_04_parent_6_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_6_2_0.json) | 140 | 140 | 3 | Holdout | 112 | 28 |
| [layer_04_parent_6_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_6_2_1.json) | 367 | 367 | 4 | Holdout | 294 | 73 |
| [layer_04_parent_7_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_7_0_0.json) | 75 | 74 | 4 | Holdout | 60 | 15 |
| [layer_04_parent_7_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_7_0_1.json) | 127 | 127 | 3 | Holdout | 102 | 25 |
| [layer_04_parent_7_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_7_1_0.json) | 110 | 110 | 3 | Holdout | 88 | 22 |
| [layer_04_parent_7_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_7_1_1.json) | 231 | 231 | 3 | Holdout | 185 | 46 |
| [layer_04_parent_7_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_7_2_0.json) | 130 | 130 | 3 | Holdout | 104 | 26 |
| [layer_04_parent_7_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_8_3_2_3/layer_04_parent_7_2_1.json) | 148 | 148 | 3 | Holdout | 118 | 30 |

</details>

<details>
<summary>layer4_12_2_2_3：85分類器</summary>

| 分類器（構成以下の識別部） | n | G | L | 方式 | T | V |
| --- | --- | --- | --- | --- | --- | --- |
| [layer_01_parent_root](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_01_parent_root.json) | 21907 | 21884 | 12 | Holdout | 17524 | 4383 |
| [layer_02_parent_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_0.json) | 1116 | 1116 | 2 | Holdout | 893 | 223 |
| [layer_02_parent_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_1.json) | 1299 | 1299 | 2 | Holdout | 1039 | 260 |
| [layer_02_parent_10](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_10.json) | 1327 | 1326 | 2 | Holdout | 1062 | 265 |
| [layer_02_parent_11](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_11.json) | 636 | 635 | 2 | Holdout | 509 | 127 |
| [layer_02_parent_2](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_2.json) | 3879 | 3875 | 2 | Holdout | 3103 | 776 |
| [layer_02_parent_3](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_3.json) | 2632 | 2630 | 2 | Holdout | 2105 | 527 |
| [layer_02_parent_4](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_4.json) | 2441 | 2438 | 2 | Holdout | 1953 | 488 |
| [layer_02_parent_5](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_5.json) | 3343 | 3341 | 2 | Holdout | 2675 | 668 |
| [layer_02_parent_6](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_6.json) | 1586 | 1584 | 2 | Holdout | 1269 | 317 |
| [layer_02_parent_7](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_7.json) | 1209 | 1209 | 2 | Holdout | 967 | 242 |
| [layer_02_parent_8](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_8.json) | 1969 | 1965 | 2 | Holdout | 1573 | 396 |
| [layer_02_parent_9](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_02_parent_9.json) | 1977 | 1976 | 2 | Holdout | 1582 | 395 |
| [layer_03_parent_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_0_0.json) | 615 | 615 | 2 | Holdout | 492 | 123 |
| [layer_03_parent_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_0_1.json) | 501 | 501 | 2 | Holdout | 401 | 100 |
| [layer_03_parent_10_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_10_0.json) | 1186 | 1185 | 2 | Holdout | 949 | 237 |
| [layer_03_parent_10_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_10_1.json) | 142 | 142 | 2 | Holdout | 114 | 28 |
| [layer_03_parent_11_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_11_0.json) | 193 | 193 | 2 | Holdout | 154 | 39 |
| [layer_03_parent_11_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_11_1.json) | 443 | 442 | 2 | Holdout | 355 | 88 |
| [layer_03_parent_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_1_0.json) | 685 | 685 | 2 | Holdout | 548 | 137 |
| [layer_03_parent_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_1_1.json) | 614 | 614 | 2 | Holdout | 491 | 123 |
| [layer_03_parent_2_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_2_0.json) | 1885 | 1882 | 2 | Holdout | 1509 | 376 |
| [layer_03_parent_2_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_2_1.json) | 2002 | 2001 | 2 | Holdout | 1602 | 400 |
| [layer_03_parent_3_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_3_0.json) | 983 | 983 | 2 | Holdout | 786 | 197 |
| [layer_03_parent_3_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_3_1.json) | 1670 | 1668 | 2 | Holdout | 1336 | 334 |
| [layer_03_parent_4_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_4_0.json) | 778 | 775 | 2 | Holdout | 623 | 155 |
| [layer_03_parent_4_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_4_1.json) | 1667 | 1667 | 2 | Holdout | 1334 | 333 |
| [layer_03_parent_5_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_5_0.json) | 688 | 687 | 2 | Holdout | 550 | 138 |
| [layer_03_parent_5_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_5_1.json) | 2662 | 2661 | 2 | Holdout | 2129 | 533 |
| [layer_03_parent_6_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_6_0.json) | 734 | 734 | 2 | Holdout | 587 | 147 |
| [layer_03_parent_6_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_6_1.json) | 854 | 852 | 2 | Holdout | 684 | 170 |
| [layer_03_parent_7_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_7_0.json) | 833 | 833 | 2 | Holdout | 666 | 167 |
| [layer_03_parent_7_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_7_1.json) | 379 | 379 | 2 | Holdout | 303 | 76 |
| [layer_03_parent_8_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_8_0.json) | 766 | 766 | 2 | Holdout | 613 | 153 |
| [layer_03_parent_8_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_8_1.json) | 1213 | 1209 | 2 | Holdout | 970 | 243 |
| [layer_03_parent_9_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_9_0.json) | 600 | 599 | 2 | Holdout | 479 | 121 |
| [layer_03_parent_9_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_03_parent_9_1.json) | 1382 | 1382 | 2 | Holdout | 1106 | 276 |
| [layer_04_parent_0_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_0_0_0.json) | 366 | 366 | 3 | Holdout | 293 | 73 |
| [layer_04_parent_0_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_0_0_1.json) | 249 | 249 | 3 | Holdout | 199 | 50 |
| [layer_04_parent_0_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_0_1_0.json) | 202 | 202 | 3 | Holdout | 162 | 40 |
| [layer_04_parent_0_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_0_1_1.json) | 299 | 299 | 3 | Holdout | 239 | 60 |
| [layer_04_parent_10_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_10_0_0.json) | 827 | 826 | 4 | Holdout | 662 | 165 |
| [layer_04_parent_10_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_10_0_1.json) | 367 | 367 | 3 | Holdout | 294 | 73 |
| [layer_04_parent_10_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_10_1_0.json) | 50 | 50 | 4 | Holdout | 40 | 10 |
| [layer_04_parent_10_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_10_1_1.json) | 92 | 92 | 3 | Holdout | 74 | 18 |
| [layer_04_parent_11_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_11_0_0.json) | 102 | 102 | 3 | Holdout | 82 | 20 |
| [layer_04_parent_11_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_11_0_1.json) | 91 | 91 | 3 | Holdout | 73 | 18 |
| [layer_04_parent_11_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_11_1_0.json) | 212 | 212 | 4 | Holdout | 170 | 42 |
| [layer_04_parent_11_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_11_1_1.json) | 231 | 230 | 3 | Holdout | 184 | 47 |
| [layer_04_parent_1_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_1_0_0.json) | 562 | 562 | 3 | Holdout | 450 | 112 |
| [layer_04_parent_1_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_1_0_1.json) | 123 | 123 | 3 | Holdout | 98 | 25 |
| [layer_04_parent_1_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_1_1_0.json) | 282 | 282 | 3 | Holdout | 226 | 56 |
| [layer_04_parent_1_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_1_1_1.json) | 332 | 332 | 3 | Holdout | 266 | 66 |
| [layer_04_parent_2_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_2_0_0.json) | 631 | 629 | 3 | Holdout | 504 | 127 |
| [layer_04_parent_2_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_2_0_1.json) | 1255 | 1254 | 3 | Holdout | 1004 | 251 |
| [layer_04_parent_2_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_2_1_0.json) | 1168 | 1167 | 3 | Holdout | 935 | 233 |
| [layer_04_parent_2_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_2_1_1.json) | 834 | 834 | 3 | Holdout | 667 | 167 |
| [layer_04_parent_3_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_3_0_0.json) | 642 | 642 | 3 | Holdout | 514 | 128 |
| [layer_04_parent_3_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_3_0_1.json) | 386 | 386 | 3 | Holdout | 309 | 77 |
| [layer_04_parent_3_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_3_1_0.json) | 1078 | 1076 | 3 | Holdout | 863 | 215 |
| [layer_04_parent_3_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_3_1_1.json) | 596 | 596 | 3 | Holdout | 477 | 119 |
| [layer_04_parent_4_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_4_0_0.json) | 256 | 256 | 3 | Holdout | 205 | 51 |
| [layer_04_parent_4_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_4_0_1.json) | 522 | 519 | 3 | Holdout | 418 | 104 |
| [layer_04_parent_4_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_4_1_0.json) | 878 | 878 | 3 | Holdout | 702 | 176 |
| [layer_04_parent_4_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_4_1_1.json) | 789 | 789 | 3 | Holdout | 631 | 158 |
| [layer_04_parent_5_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_5_0_0.json) | 536 | 535 | 3 | Holdout | 429 | 107 |
| [layer_04_parent_5_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_5_0_1.json) | 152 | 152 | 3 | Holdout | 122 | 30 |
| [layer_04_parent_5_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_5_1_0.json) | 445 | 445 | 3 | Holdout | 356 | 89 |
| [layer_04_parent_5_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_5_1_1.json) | 2223 | 2222 | 3 | Holdout | 1778 | 445 |
| [layer_04_parent_6_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_6_0_0.json) | 614 | 614 | 4 | Holdout | 491 | 123 |
| [layer_04_parent_6_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_6_0_1.json) | 120 | 120 | 3 | Holdout | 96 | 24 |
| [layer_04_parent_6_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_6_1_0.json) | 692 | 690 | 3 | Holdout | 553 | 139 |
| [layer_04_parent_6_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_6_1_1.json) | 162 | 162 | 3 | Holdout | 130 | 32 |
| [layer_04_parent_7_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_7_0_0.json) | 190 | 190 | 4 | Holdout | 152 | 38 |
| [layer_04_parent_7_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_7_0_1.json) | 643 | 643 | 3 | Holdout | 514 | 129 |
| [layer_04_parent_7_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_7_1_0.json) | 117 | 117 | 3 | Holdout | 94 | 23 |
| [layer_04_parent_7_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_7_1_1.json) | 263 | 263 | 3 | Holdout | 210 | 53 |
| [layer_04_parent_8_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_8_0_0.json) | 688 | 688 | 4 | Holdout | 550 | 138 |
| [layer_04_parent_8_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_8_0_1.json) | 78 | 78 | 3 | Holdout | 62 | 16 |
| [layer_04_parent_8_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_8_1_0.json) | 808 | 804 | 3 | Holdout | 647 | 161 |
| [layer_04_parent_8_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_8_1_1.json) | 410 | 410 | 3 | Holdout | 328 | 82 |
| [layer_04_parent_9_0_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_9_0_0.json) | 494 | 493 | 4 | Holdout | 394 | 100 |
| [layer_04_parent_9_0_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_9_0_1.json) | 106 | 106 | 3 | Holdout | 85 | 21 |
| [layer_04_parent_9_1_0](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_9_1_0.json) | 1046 | 1046 | 4 | Holdout | 837 | 209 |
| [layer_04_parent_9_1_1](data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/layer4_12_2_2_3/layer_04_parent_9_1_1.json) | 339 | 339 | 3 | Holdout | 271 | 68 |

</details>

## 付録B. API分類器152個の全件一覧

各FoldのT/Vは`Train/Validation`をFold順に記載。Holdoutは1組、分割不能は「—」。正例0・負例0列は、少なくとも1つのFoldでTrainの該当支持数が0となる**出力ラベルの和集合数**であり、Fold数の和ではない。分割不能10個ではFoldがないため0と表示されるが、正常な教師分布を意味しない。

| APIカテゴリ | n | G | L | 方式 | 各FoldのT/V | Train正例0出力 | Train負例0出力 | 定数教師 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [accounting](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_accounting.json) | 3 | 3 | 2 | 3-fold CV | 2/1; 2/1; 2/1 | 1 | 1 | 可変出力あり |
| [addresses](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_addresses.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [advertising](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_advertising.json) | 66 | 66 | 11 | Holdout | 53/13 | 0 | 0 | 可変出力あり |
| [air_travel](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_air_travel.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [analytics](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_analytics.json) | 33 | 33 | 8 | Holdout | 26/7 | 0 | 0 | 可変出力あり |
| [animals](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_animals.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [api](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_api.json) | 7 | 7 | 2 | 5-fold CV | 5/2; 5/2; 6/1; 6/1; 6/1 | 0 | 0 | 可変出力あり |
| [application_development](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_application_development.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [applications](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_applications.json) | 6 | 6 | 3 | 5-fold CV | 4/2; 5/1; 5/1; 5/1; 5/1 | 0 | 0 | 可変出力あり |
| [astronomy](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_astronomy.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [authentication](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_authentication.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [auto](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_auto.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [backend](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_backend.json) | 12 | 12 | 5 | 5-fold CV | 9/3; 9/3; 10/2; 10/2; 10/2 | 0 | 0 | 可変出力あり |
| [barcodes](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_barcodes.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [big_data](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_big_data.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [biometrics](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_biometrics.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [bitcoin](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_bitcoin.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [blockchain](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_blockchain.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [blogging](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_blogging.json) | 37 | 37 | 9 | Holdout | 30/7 | 0 | 0 | 可変出力あり |
| [booking](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_booking.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [bookmarks](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_bookmarks.json) | 52 | 52 | 5 | Holdout | 42/10 | 0 | 0 | 可変出力あり |
| [books](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_books.json) | 8 | 8 | 1 | 5-fold CV | 6/2; 6/2; 6/2; 7/1; 7/1 | 0 | 1 | 全出力定数 |
| [bots](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_bots.json) | 5 | 5 | 2 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 0 | 可変出力あり |
| [business](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_business.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [calendars](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_calendars.json) | 21 | 21 | 4 | Holdout | 17/4 | 0 | 0 | 可変出力あり |
| [chat](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_chat.json) | 19 | 19 | 7 | 5-fold CV | 15/4; 15/4; 15/4; 15/4; 16/3 | 0 | 0 | 可変出力あり |
| [classification](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_classification.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [cloud](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_cloud.json) | 41 | 41 | 4 | Holdout | 33/8 | 0 | 0 | 可変出力あり |
| [collaboration](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_collaboration.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [content](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_content.json) | 56 | 56 | 1 | Holdout | 45/11 | 0 | 1 | 全出力定数 |
| [coupons](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_coupons.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [covid_19](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_covid_19.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [cryptocurrency](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_cryptocurrency.json) | 5 | 5 | 3 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 0 | 可変出力あり |
| [cycling](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_cycling.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [data](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_data.json) | 5 | 5 | 2 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 0 | 可変出力あり |
| [data_as_a_service](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_data_as_a_service.json) | 8 | 8 | 4 | 5-fold CV | 6/2; 6/2; 6/2; 7/1; 7/1 | 0 | 0 | 可変出力あり |
| [data_mining](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_data_mining.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [database](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_database.json) | 54 | 54 | 12 | Holdout | 43/11 | 0 | 0 | 可変出力あり |
| [design](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_design.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [developers](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_developers.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [documents](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_documents.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [domains](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_domains.json) | 5 | 5 | 4 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 3 | 0 | 可変出力あり |
| [ecommerce](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_ecommerce.json) | 344 | 344 | 54 | Holdout | 275/69 | 0 | 0 | 可変出力あり |
| [education](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_education.json) | 38 | 38 | 12 | Holdout | 30/8 | 0 | 0 | 可変出力あり |
| [electronic_signature](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_electronic_signature.json) | 22 | 22 | 1 | Holdout | 18/4 | 0 | 1 | 全出力定数 |
| [email](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_email.json) | 58 | 58 | 14 | Holdout | 46/12 | 0 | 0 | 可変出力あり |
| [enterprise](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_enterprise.json) | 31 | 31 | 17 | Holdout | 25/6 | 2 | 0 | 可変出力あり |
| [entertainment](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_entertainment.json) | 19 | 19 | 5 | 5-fold CV | 15/4; 15/4; 15/4; 15/4; 16/3 | 1 | 0 | 可変出力あり |
| [environment](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_environment.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [esports](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_esports.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [events](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_events.json) | 25 | 25 | 6 | Holdout | 20/5 | 0 | 0 | 可変出力あり |
| [fax](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_fax.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [feedback](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_feedback.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [feeds](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_feeds.json) | 18 | 18 | 4 | 5-fold CV | 14/4; 14/4; 14/4; 15/3; 15/3 | 0 | 0 | 可変出力あり |
| [file_sharing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_file_sharing.json) | 11 | 11 | 6 | 5-fold CV | 8/3; 9/2; 9/2; 9/2; 9/2 | 1 | 0 | 可変出力あり |
| [financial](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_financial.json) | 55 | 55 | 20 | Holdout | 44/11 | 0 | 0 | 可変出力あり |
| [fitness](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_fitness.json) | 20 | 20 | 2 | Holdout | 16/4 | 0 | 0 | 可変出力あり |
| [food](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_food.json) | 10 | 10 | 4 | 5-fold CV | 8/2; 8/2; 8/2; 8/2; 8/2 | 2 | 0 | 可変出力あり |
| [forms](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_forms.json) | 5 | 5 | 1 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 1 | 全出力定数 |
| [framework](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_framework.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [gambling](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_gambling.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [games](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_games.json) | 19 | 19 | 6 | 5-fold CV | 15/4; 15/4; 15/4; 15/4; 16/3 | 0 | 0 | 可変出力あり |
| [geography](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_geography.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [government](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_government.json) | 51 | 49 | 27 | Holdout | 39/12 | 3 | 0 | 可変出力あり |
| [hardware](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_hardware.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [health](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_health.json) | 5 | 5 | 2 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 0 | 可変出力あり |
| [healthcare](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_healthcare.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [home_automation](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_home_automation.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [hosting](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_hosting.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [identity](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_identity.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [images](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_images.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [intelligence](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_intelligence.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [internet_things](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_internet_things.json) | 8 | 8 | 5 | 5-fold CV | 6/2; 6/2; 6/2; 7/1; 7/1 | 2 | 0 | 可変出力あり |
| [invoicing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_invoicing.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [jobs](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_jobs.json) | 36 | 36 | 6 | Holdout | 29/7 | 0 | 0 | 可変出力あり |
| [language](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_language.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [learning_management_systems](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_learning_management_systems.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [library](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_library.json) | 4 | 4 | 1 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 1 | 全出力定数 |
| [location](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_location.json) | 5 | 5 | 2 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 0 | 可変出力あり |
| [machine_learning](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_machine_learning.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [mail](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_mail.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [mapping](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_mapping.json) | 1720 | 1720 | 52 | Holdout | 1376/344 | 0 | 0 | 可変出力あり |
| [marketing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_marketing.json) | 6 | 6 | 2 | 5-fold CV | 4/2; 5/1; 5/1; 5/1; 5/1 | 0 | 0 | 可変出力あり |
| [media](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_media.json) | 28 | 28 | 8 | Holdout | 22/6 | 0 | 0 | 可変出力あり |
| [medical](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_medical.json) | 8 | 8 | 5 | 5-fold CV | 6/2; 6/2; 6/2; 7/1; 7/1 | 1 | 0 | 可変出力あり |
| [medicine](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_medicine.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [merchants](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_merchants.json) | 5 | 5 | 3 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 0 | 可変出力あり |
| [messaging](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_messaging.json) | 201 | 201 | 22 | Holdout | 161/40 | 0 | 0 | 可変出力あり |
| [monitoring](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_monitoring.json) | 5 | 5 | 1 | 5-fold CV | 4/1; 4/1; 4/1; 4/1; 4/1 | 0 | 1 | 全出力定数 |
| [movies](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_movies.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [music](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_music.json) | 144 | 144 | 29 | Holdout | 115/29 | 0 | 0 | 可変出力あり |
| [natural_language_processing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_natural_language_processing.json) | 2 | 2 | 2 | 2-fold CV | 1/1; 1/1 | 2 | 2 | 可変出力あり |
| [news_services](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_news_services.json) | 45 | 45 | 12 | Holdout | 36/9 | 0 | 0 | 可変出力あり |
| [nutrition](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_nutrition.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [office](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_office.json) | 27 | 27 | 11 | Holdout | 22/5 | 0 | 0 | 可変出力あり |
| [other](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_other.json) | 100 | 100 | 25 | Holdout | 80/20 | 0 | 0 | 可変出力あり |
| [parsing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_parsing.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [payments](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_payments.json) | 48 | 48 | 15 | Holdout | 38/10 | 0 | 0 | 可変出力あり |
| [pdf](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_pdf.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [performance](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_performance.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [personal_information_management](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_personal_information_management.json) | 24 | 24 | 7 | Holdout | 19/5 | 0 | 0 | 可変出力あり |
| [photos](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_photos.json) | 385 | 385 | 26 | Holdout | 308/77 | 0 | 0 | 可変出力あり |
| [platform_as_a_service](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_platform_as_a_service.json) | 3 | 3 | 1 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 1 | 全出力定数 |
| [politics](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_politics.json) | 4 | 4 | 1 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 1 | 全出力定数 |
| [predictions](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_predictions.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [printing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_printing.json) | 4 | 4 | 2 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 0 | 可変出力あり |
| [products](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_products.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [project_management](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_project_management.json) | 30 | 30 | 11 | Holdout | 24/6 | 0 | 0 | 可変出力あり |
| [purchasing](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_purchasing.json) | 4 | 4 | 1 | 4-fold CV | 3/1; 3/1; 3/1; 3/1 | 0 | 1 | 全出力定数 |
| [qa](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_qa.json) | 7 | 7 | 2 | 5-fold CV | 5/2; 5/2; 6/1; 6/1; 6/1 | 0 | 0 | 可変出力あり |
| [random](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_random.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [real_estate](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_real_estate.json) | 24 | 24 | 7 | Holdout | 19/5 | 0 | 0 | 可変出力あり |
| [recognition](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_recognition.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [recommendations](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_recommendations.json) | 40 | 40 | 5 | Holdout | 32/8 | 0 | 0 | 可変出力あり |
| [reference](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_reference.json) | 166 | 166 | 39 | Holdout | 133/33 | 0 | 0 | 可変出力あり |
| [reporting](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_reporting.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [rewards](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_rewards.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [ride_share](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_ride_share.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [sales](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_sales.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [science](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_science.json) | 23 | 23 | 8 | Holdout | 18/5 | 0 | 0 | 可変出力あり |
| [screenshots](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_screenshots.json) | 6 | 6 | 1 | 5-fold CV | 4/2; 5/1; 5/1; 5/1; 5/1 | 0 | 1 | 全出力定数 |
| [search](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_search.json) | 154 | 152 | 35 | Holdout | 124/30 | 0 | 0 | 可変出力あり |
| [security](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_security.json) | 20 | 20 | 8 | Holdout | 16/4 | 0 | 0 | 可変出力あり |
| [semantics](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_semantics.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [sentiment](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_sentiment.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [seo](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_seo.json) | 10 | 10 | 5 | 5-fold CV | 8/2; 8/2; 8/2; 8/2; 8/2 | 2 | 0 | 可変出力あり |
| [shipping](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_shipping.json) | 19 | 19 | 5 | 5-fold CV | 15/4; 15/4; 15/4; 15/4; 16/3 | 0 | 0 | 可変出力あり |
| [social](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_social.json) | 718 | 717 | 48 | Holdout | 574/144 | 0 | 0 | 可変出力あり |
| [spam](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_spam.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [sports](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_sports.json) | 24 | 24 | 9 | Holdout | 19/5 | 0 | 0 | 可変出力あり |
| [spreadsheets](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_spreadsheets.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [standards](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_standards.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [statistics](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_statistics.json) | 9 | 9 | 2 | 5-fold CV | 7/2; 7/2; 7/2; 7/2; 8/1 | 0 | 0 | 可変出力あり |
| [stocks](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_stocks.json) | 3 | 3 | 2 | 3-fold CV | 2/1; 2/1; 2/1 | 0 | 2 | 可変出力あり |
| [storage](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_storage.json) | 58 | 58 | 7 | Holdout | 46/12 | 0 | 0 | 可変出力あり |
| [streaming](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_streaming.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [support](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_support.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [tagging](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_tagging.json) | 7 | 7 | 1 | 5-fold CV | 5/2; 5/2; 6/1; 6/1; 6/1 | 0 | 1 | 全出力定数 |
| [telephony](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_telephony.json) | 345 | 344 | 18 | Holdout | 275/70 | 0 | 0 | 可変出力あり |
| [text](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_text.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [tools](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_tools.json) | 151 | 151 | 38 | Holdout | 121/30 | 0 | 0 | 可変出力あり |
| [translation](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_translation.json) | 21 | 21 | 2 | Holdout | 17/4 | 0 | 0 | 可変出力あり |
| [transportation](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_transportation.json) | 26 | 26 | 10 | Holdout | 21/5 | 0 | 0 | 可変出力あり |
| [travel](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_travel.json) | 67 | 67 | 24 | Holdout | 54/13 | 0 | 0 | 可変出力あり |
| [url_shortener](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_url_shortener.json) | 23 | 23 | 2 | Holdout | 18/5 | 0 | 0 | 可変出力あり |
| [video](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_video.json) | 309 | 309 | 16 | Holdout | 247/62 | 0 | 0 | 可変出力あり |
| [voice](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_voice.json) | 1 | 1 | 1 | 分割不能 | — | 0 | 0 | 全出力定数 |
| [weather](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_weather.json) | 40 | 40 | 11 | Holdout | 32/8 | 0 | 0 | 可変出力あり |
| [web_site_management](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_web_site_management.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [wi_fi](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_wi_fi.json) | 2 | 2 | 1 | 2-fold CV | 1/1; 1/1 | 0 | 1 | 全出力定数 |
| [widgets](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_widgets.json) | 65 | 65 | 4 | Holdout | 52/13 | 0 | 0 | 可変出力あり |
| [wiki](data/hierarchical_api_selection/classifier_cv_v1/splits/api_classification/category_wiki.json) | 9 | 9 | 3 | 5-fold CV | 7/2; 7/2; 7/2; 7/2; 8/1 | 0 | 0 | 可変出力あり |

## 付録C. カテゴリ分類器共有候補の全23群

表のsignatureは先頭16文字のみを表示する。完全なhashと判定条件はclassifier_manifestに保存されている。いずれも共有は未適用である。

| 等価群signature（短縮） | 分類器数 | 候補ID一覧 |
| --- | --- | --- |
| `04a86bc936edd30d` | 2 | category_classification/layer3_12_2_6/layer_02_parent_0<br>category_classification/layer4_12_2_2_3/layer_02_parent_0 |
| `054c8d3cc7a672ee` | 2 | category_classification/layer3_12_2_6/layer_02_parent_11<br>category_classification/layer4_12_2_2_3/layer_02_parent_11 |
| `1f11fe92975f9694` | 2 | category_classification/layer2_33_4/layer_02_parent_0<br>category_classification/layer3_8_4_4/layer_03_parent_0_0 |
| `29c7d92c477a704d` | 2 | category_classification/layer3_12_4_3/layer_03_parent_1_3<br>category_classification/layer4_12_2_2_3/layer_04_parent_1_1_1 |
| `337bf2eb7a382f56` | 2 | category_classification/layer2_4_43/layer_01_parent_root<br>category_classification/layer3_4_4_8/layer_01_parent_root |
| `41a934056900d1f7` | 2 | category_classification/layer3_12_6_2/layer_03_parent_9_2<br>category_classification/layer4_12_2_2_3/layer_04_parent_9_1_0 |
| `4564faee940d63d8` | 2 | category_classification/layer2_33_4/layer_02_parent_1<br>category_classification/layer3_8_4_4/layer_03_parent_0_1 |
| `4b64c23a51072918` | 2 | category_classification/layer3_12_2_6/layer_02_parent_8<br>category_classification/layer4_12_2_2_3/layer_02_parent_8 |
| `4d598053bc152d7a` | 2 | category_classification/layer3_12_2_6/layer_02_parent_6<br>category_classification/layer4_12_2_2_3/layer_02_parent_6 |
| `687c26699b9020c5` | 5 | category_classification/layer2_12_13/layer_01_parent_root<br>category_classification/layer3_12_2_6/layer_01_parent_root<br>category_classification/layer3_12_4_3/layer_01_parent_root<br>category_classification/layer3_12_6_2/layer_01_parent_root<br>category_classification/layer4_12_2_2_3/layer_01_parent_root |
| `6ac088de6f1abe48` | 2 | category_classification/layer3_12_2_6/layer_02_parent_10<br>category_classification/layer4_12_2_2_3/layer_02_parent_10 |
| `7d927be0e52348e3` | 2 | category_classification/layer3_12_2_6/layer_02_parent_1<br>category_classification/layer4_12_2_2_3/layer_02_parent_1 |
| `89a989cb96c4b850` | 2 | category_classification/layer3_12_2_6/layer_02_parent_3<br>category_classification/layer4_12_2_2_3/layer_02_parent_3 |
| `925016ff86c35efd` | 3 | category_classification/layer3_12_4_3/layer_03_parent_0_0<br>category_classification/layer4_8_3_2_3/layer_04_parent_0_0_0<br>category_classification/layer4_12_2_2_3/layer_04_parent_0_0_0 |
| `98bc511fff552676` | 2 | category_classification/layer3_12_2_6/layer_02_parent_9<br>category_classification/layer4_12_2_2_3/layer_02_parent_9 |
| `b2632298136ac7b5` | 2 | category_classification/layer3_12_2_6/layer_02_parent_5<br>category_classification/layer4_12_2_2_3/layer_02_parent_5 |
| `b2ad4a185834e0d5` | 2 | category_classification/layer3_12_2_6/layer_02_parent_2<br>category_classification/layer4_12_2_2_3/layer_02_parent_2 |
| `b4d19dd599137ab0` | 3 | category_classification/layer2_8_19/layer_01_parent_root<br>category_classification/layer3_8_4_4/layer_01_parent_root<br>category_classification/layer4_8_3_2_3/layer_01_parent_root |
| `e2f2a7cd26ec9d6d` | 2 | category_classification/layer3_12_2_6/layer_02_parent_7<br>category_classification/layer4_12_2_2_3/layer_02_parent_7 |
| `e71d1f140701bcef` | 2 | category_classification/layer3_12_6_2/layer_03_parent_5_4<br>category_classification/layer3_16_4_3/layer_03_parent_6_2 |
| `edeb754d9336c3ae` | 2 | category_classification/layer3_12_4_3/layer_03_parent_0_3<br>category_classification/layer4_12_2_2_3/layer_04_parent_0_1_1 |
| `f36e9f9575e02848` | 2 | category_classification/layer3_12_2_6/layer_02_parent_4<br>category_classification/layer4_12_2_2_3/layer_02_parent_4 |
| `f8529914d3b357f5` | 2 | category_classification/layer3_12_6_2/layer_03_parent_0_0<br>category_classification/layer3_16_4_3/layer_03_parent_0_0 |

## 付録D. 集計値の検算と参照関数

```text
分類器総数 = 653 + 152 = 805
方式別総数 = 97 + 698 + 10 = 805
CV方式内訳 = 32(5-fold) + 14(4-fold) + 9(3-fold) + 42(2-fold) = 97
CV Fold数 = 32*5 + 14*4 + 9*3 + 42*2 = 327
Train/Validationペア = 327 + 698 = 1,025
単一出力分類器 = 56(CV) + 2(Holdout) + 10(不能) = 68
カテゴリ分類器 = 2(CV) + 651(Holdout) = 653
API分類器 = 95(CV) + 47(Holdout) + 10(不能) = 152
カテゴリサービス = 3,946(Mashup) + 909(候補内API) + 17,052(候補外API) = 21,907
API正例サービス = 3,946 + 909 = 4,855
共有候補関与分類器 = 51、等価群 = 23、候補上の削減数 = 51 - 23 = 28（未適用）
```

今回の実装の主要根拠：

| 処理 | 根拠 |
| --- | --- |
| load_phase_one | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:62` |
| classifiers | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:100` |
| training_fingerprint | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:93` |
| target_profile | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:141` |
| local_features | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:159` |
| make_partition | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:169` |
| create | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:217` |
| row_summary | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:284` |
| summarize | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:303` |
| independent_profile | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:468` |
| validate | `scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py:491` |

group割当helperの根拠は `scripts/hierarchical_api_selection/pipeline.py:stratify`。新処理はこのhelperを分類器ローカルに呼び出すが、同ファイルの`plan_folds`は使用していない。元のファイルを書き換えずに再利用している。

本書に記載した件数は保存済みファイルから再集計した値である。見つからない値や学習結果を推測で埋めていない。本書の作成では既存コード・データ・検証結果を変更せず、この新規Markdownだけをリポジトリ直下へ保存した。
