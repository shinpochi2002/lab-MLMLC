# 階層的API推薦モデル：Phase 1・2の実施内容まとめ

## 1. 実施したこと

モデルの再学習に向けて、既存データから学習元データを再構成し、全16階層構成・全カテゴリ分類層・API分類層で共有するGroup 5-Foldを作成した。

- **Phase 1**：元データの由来照合、同名APIに上書きされたMashup61件の復元、固定評価との重複除外、教師ラベルの整理。
- **Phase 2**：共通groupの作成、seed42の5-Fold割当、内部Early Stopping割当、Fold別・分類器別件数の集計、自動検証。

既存の推薦候補909 API、152カテゴリ、Balanced K-meansによる16階層構成を維持した。固定Validation100件・Test236件は変更せず、学習候補とFold分割から除外した。

## 2. Phase 1：データの再構成

### 元レコードの識別と復元

既存のカテゴリ拡張データ22,180件は、Mashup4,219件とAPI17,961件からなる。同名衝突によって失われたMashup65件を元レコードと照合し、固定評価に属する4件を復元対象から外して、残り61件を復元した。

MashupとAPIには `mashup:<source_id>`、`api:<source_id>` の種別付きIDを付けた。元IDの一意性を検証し、元ファイル・元キー・ハッシュも保存した。同名のMashupとAPIを異なる実体として保持できる。

### 評価データとの重複除外

分岐・カテゴリへの展開前に、以下を除外した。

- 固定評価Mashup336実体。既存22,180件に含まれる332件に加え、同名衝突で既に失われていた4件も復元しないことを確認した。
- 固定評価の `touropia` と同一本文を持つ別名Mashup `vacationic` 1件。正解APIは異なるが、本文の漏洩を防ぐため学習側を除外した。
- 空本文の `awardwallet-app` 1件。

本文照合にはHTML entity復号、Unicode NFKC、空白の統一を使用した。モデル入力として保存する元の本文は変更していない。

評価Mashupと同名のAPI4件（`metro-mapper`、`trackthis`、`distance-calculator`、`rosetta`）は、本文が異なる別実体のため保持した。

### 完成したデータ件数

| データ種別 | 既存拡張データ内 | 既存分から除外 | 復元 | 完成件数 |
| --- | ---: | ---: | ---: | ---: |
| Mashup由来 | 4,219 | 334 | 61 | 3,946 |
| 909候補内API由来 | 909 | 0 | 0 | 909 |
| 909候補外API由来 | 17,052 | 0 | 0 | 17,052 |
| カテゴリ分類用合計 | 22,180 | 334 | 61 | 21,907 |
| API分類用正例の部分集合 | 5,128 | 334 | 61 | 4,855 |

既存分からの除外334件は、評価実体332件＋別名本文一致1件＋空本文1件。復元前の全元実体を対象にした除外記録は、評価336件＋別名1件＋空本文1件＝338件である。集計対象の違いによる数値差で、除外漏れではない。

カテゴリ分類用21,907件のうち、Mashup3,946件と候補API自己説明文909件の計4,855件をAPI分類用正例とした。候補外17,052 APIはカテゴリ分類にのみ利用し、API分類の全0負例にはしていない。

API分類用4,855件はカテゴリ分類用データの部分集合であり、独立した追加サービスとして合計しない。カテゴリ別API分類器へ展開した正例は延べ6,514行である。

## 3. Phase 2：共通Group 5-Fold

### groupの作成

正規化後の本文が同じレコードは、由来や正解ラベルが異なっても同じgroupへまとめた。個々のレコードと教師ラベルは保持し、ラベルの自動統合はしていない。

- 共通母集合：21,907サービス、21,884 group。
- API分類用正例：4,855サービス、4,849 group。
- group ID：group内のentity ID一覧をソートした内容のSHA-256から生成。

### Foldと内部Early Stopping

seed42で、カテゴリ・APIのマルチラベル支持数とデータ由来の偏りを抑える割当処理を実装した。groupの分離を最優先し、group数の容量制約を設けた。希少ラベルの完全な層化は保証しない。

各Foldの外側Trainを、さらにgroup単位で約80%の実学習Trainと約20%の内部Early Stopping用Validationへ分けた。全16構成とAPI分類は同じ外側・内部割当を継承する。

以下は共通母集合のサービス単位の件数である。

| Fold | 外側Train全体 | 実学習Train | 内部Early Stopping | 外側Validation |
| --- | ---: | ---: | ---: | ---: |
| fold_01 | 17522 | 14020 | 3502 | 4385 |
| fold_02 | 17528 | 14022 | 3506 | 4379 |
| fold_03 | 17529 | 14024 | 3505 | 4378 |
| fold_04 | 17528 | 14026 | 3502 | 4379 |
| fold_05 | 17521 | 14017 | 3504 | 4386 |

外側Train全体＝実学習Train＋内部Early Stopping。各Foldでこの3用途は外側Validationと分離される。固定Validation100件・Test236件はこの表の外側Validationや内部Early Stoppingとは別で、どのFoldにも含まれない。

同じサービスが複数の層・分岐で使われる場合も、同一Fold内では同じ役割を維持する。各サービスが複数FoldのTrainとして再利用されることは通常のCVの仕様であり、同一Fold内の学習・評価間の混入とは区別する。

## 4. 少数データ・学習不能状態の集計

復元後に再集計した結果、正例1件のみのAPIは**87個**、データ1件のみのAPI分類カテゴリは**10個**となった。復元前の調査値98個・11個との差は、61 Mashupの復元に伴う教師例の増加による。

| Fold | 実学習Trainが空のAPI分類器 | 内部検証が成立しないAPI分類器 |
| --- | ---: | ---: |
| fold_01 | 2 | 78 |
| fold_02 | 2 | 79 |
| fold_03 | 3 | 82 |
| fold_04 | 2 | 79 |
| fold_05 | 1 | 80 |

上表は全構成で共有する152 API分類器の件数で、16構成分を掛けていない。「内部検証が成立しない」は実学習Trainまたは内部Early Stoppingのデータが空の状態を指す。ラベルごとの正例0件は別の列で記録した。カテゴリ分類器では、全16構成のいずれにもTrain空・内部検証不能は検出されなかった。

少数データを別用途へ強制移動する処理や、学習のフォールバックは実装していない。既知Mashupから生成できるAPI負例候補の件数は集計したが、負例レコードは生成していない。

## 5. 作成したファイル

データの保存先は [data/hierarchical_api_selection/group_5fold_v1/](data/hierarchical_api_selection/group_5fold_v1/) 。この中に45ファイルを新規作成した。

| 保存先 | 内容 |
| --- | --- |
| [README.md](data/hierarchical_api_selection/group_5fold_v1/README.md) | 目的、構成、スキーマ、除外条件、再生成手順、Phase 3への引継ぎ |
| `manifests/` | 元ファイル・コードのハッシュ、データ件数、除外理由、固定ラベル・階層、分類器定義、分割条件 |
| `source_records/` | 共通母集合、カテゴリ分類用データ、API正例部分集合、group一覧、復元61件 |
| `splits/` | group・entityの共通Fold割当と、各FoldのTrain・外側Validation・内部Early Stoppingのgroup一覧 |
| [reports/data_usage_summary.csv](data/hierarchical_api_selection/group_5fold_v1/reports/data_usage_summary.csv) | データの元件数・除外・復元・完成件数 |
| [reports/fold_distribution.csv](data/hierarchical_api_selection/group_5fold_v1/reports/fold_distribution.csv) | Fold・用途別のサービス数、group数、由来別件数・割合 |
| [reports/classifier_sample_counts.csv](data/hierarchical_api_selection/group_5fold_v1/reports/classifier_sample_counts.csv) | 全16構成・全層・全分類器の用途別件数と負例候補数 |
| `reports/category_label_distribution.csv`・`api_label_distribution.csv` | 各用途の全ラベル支持数。正例0件も記録 |
| `reports/rare_classifier_summary.csv`・`unavailable_classifiers.csv` | 希少ラベルと学習・内部検証不能の分類器一覧 |
| [reports/data_leakage_audit.md](data/hierarchical_api_selection/group_5fold_v1/reports/data_leakage_audit.md) | 実体・本文・同名衝突を区別したリーク検査 |
| [reports/implementation_report.md](data/hierarchical_api_selection/group_5fold_v1/reports/implementation_report.md) | 実装結果、件数、少数データの状態、制約 |
| [checks/validation_results.json](data/hierarchical_api_selection/group_5fold_v1/checks/validation_results.json) | 自動検証結果、警告、生成物のハッシュ一覧 |

独立スクリプトは [scripts/hierarchical_api_selection/](scripts/hierarchical_api_selection/) に7ファイルを作成した。

- `prepare_training_dataset.py`：Phase 1の再構成。
- `create_group_folds.py`：共通Foldと内部割当の作成。
- `summarize_data_usage.py`：件数表・README・報告書の作成。
- `validate_dataset.py`：整合性・再現性・既存ファイル保護の検証。
- `pipeline.py`：共通処理。
- `tests/test_pipeline.py`：保護・分割・正規化等のテスト。
- `README.md`：入力・出力・実行コマンド。

新規出力が存在する場合は停止する仕様とし、無条件上書きのオプションは設けていない。再生成時は別の未使用出力先を指定する。

## 6. 検証結果と既存環境の保護

実装時に次を確認した。

- **自動検証116項目が通過**。元データからの教師・本文・復元対象の再照合、部分集合関係、固定ラベル空間、全階層のカテゴリ網羅を検証した。
- 固定評価との実体・正規化本文の重複は**0件**。
- 外側Train／Validation、内部Train／Early Stopping間のgroup重複は**0件**。
- seed42の再計算と、入力順序を反転した再計算で、同一Fold割当を確認した。
- **テスト12件が通過**。同名異実体、本文一致・異ラベル、出力上書き拒否、空本文、group分離等を確認した。
- 作業開始前の**既存13,256ファイル**について、サイズ・更新日時・モード・リンク情報の変更は検出されなかった。宣言した入力・新規コードはSHA-256でも照合した。
- 生成済み44成果物のハッシュを照合した。自分自身を除く検証報告内の一覧であり、検証報告を含むデータ側のファイル総数は45。

既存モデル全体の内容ハッシュは取得していない。既存ファイルのメタデータ照合と、入力・コードのハッシュ照合が保護検証の範囲である。

既存データ、学習・推論コード、モデル、ログ、評価結果、固定100／236件、過去の監査・設計レポートは変更していない。既存の未追跡ファイルも保持した。

## 7. 今回実施していないこと・Phase 3に残した事項

今回の実施範囲はPhase 1・2に限定した。モデル学習・再学習・推論評価、既存checkpointの置換、階層の再構築、負例データの本生成は実施していない。

Phase 3では、以下を決める必要がある。

- Train空・内部検証不能の分類器、正例0件のラベルに対する学習規則。
- 正例が全行で1になる出力で、既存の `pos_weight=neg/pos` が0になる問題への対処。
- 既知Mashupを使う負例の採否・比率と、追加後の件数集計。
- Early Stopping、設定選択、全開発集合による最終再学習の具体的方法。
- 固定Validation100件とTest236件の評価、およびEとのプロット。「両評価の平均」の集計単位も未確定。

固定評価の `trackthis` と `rosetta` には、元Mashupの使用APIから導くカテゴリと固定カテゴリ正解の不一致があることを記録した。最終API正解は元Mashupと一致しており、今回固定評価は修正していない。

今回の処理は実体・正規化本文の一致による混入を防止するもので、近似本文や未確認の別名を網羅した保証ではない。また、過去に固定336件を比較評価へ使用した履歴を解消するものではない。
