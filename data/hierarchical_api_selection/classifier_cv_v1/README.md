# Classifier-local CV/Holdout v1

Phase 2の分類器単位への変更。カテゴリ653、API152、合計805分類器。
全層で共通のFold割当は使用しない。Phase 1の本文group IDだけを保持し、分類器ごとの教師から独立して分割する。

## 固定条件と入力

Phase 1: `data/hierarchical_api_selection/group_5fold_v1`。カテゴリ21,907サービス、API正例4,855サービス、61 Mashupの復元と評価除外をそのまま保持。候補909 API、152カテゴリ、既存16階層は変更しない。
固定Validation100件・Test236件は教師、CV/Holdout、学習条件選択・Early Stoppingに含めない。既存Group 5-Foldのファイルも保持し、その割当は新しい分割に利用しない。

## 分割条件

- サービスn<20かつ独立group G>=2：k=min(5,G)のCV。出力1種類でも試行する。
- n>=20かつG>=2：groupで約80/20 Holdout。Validation group数はmax(1,min(G-1,floor(.2G+.5)))。サービス比率のずれも記録。
- G<2またはn=0：独立分割不能。Foldを捏造せず理由を保存。
- seed42を分類器ごとに独立して適用。希少ラベルからのgroup層化近似で、単一出力は両クラスがあればbinary支持数も考慮。完全なMultilabelStratifiedKFold互換ではない。
- 同じ分類器内でgroupを分けない。層化よりgroup制約を優先し、分割後にValidationからTrainへ移動しない。
- 全正例/全負例、定数教師、Train正例0・負例0、pos_weight=0の可能性をFold別に記録。分割成立を正常なCV学習成功とはみなさない。

API分類分割は**正例のみの暫定版**。Phase 3で既知Mashup負例を追加する場合は、別出力先に新しい教師・分割を作る。今回は負例データも学習fallbackも生成しない。

## ファイルとスキーマ

- manifests/source_manifest.json：Phase 1全成果物・元入力・参照コードの相対パスとSHA-256、Git commit、新実装hash、条件・UTC作成日時。
- manifests/existing_file_snapshot.json：全既存ファイルのsize/mtime/mode/link。モデル内容自体のhashは取得しない。
- manifests/classifier_manifest.json：全分類器ID、層、親、所属構成、固定出力、件数・支持数、保存先、分割方式、共有候補の等価性。
- manifests/split_manifest.json：閾値とgroup分割規則、条件選択/ES/全データ再学習の既存参照条件。学習そのものは行わない。
- splits/category_classification/<hierarchy>/layer_<nn>_parent_<path>.json、splits/api_classification/category_<name>.json：1分類器1ファイル。
- 各ファイルのmembers：entity_id、group_id、target_labels、source_type。descriptionはsource_record_fileのPhase 1データをentity_idで参照し、カテゴリ用・API用教師はここに保存した固定target_labelsを使う。
- partition：method、n_splits、partition_status、full_profile、training_distribution_status、例外理由、folds。
- folds：ローカルfold_id、train_entity_ids、validation_entity_ids、各roleのprofile、警告。fold_idはそのclassifier_uidの中だけで有効。他分類器のfold_01と結合しない。
- profile：services、groups、由来件数、各出力のpositive/negative支持数、正例0/負例0/可変ラベル、定数教師・単一出力の状態。ラベルごとのnegative支持数はmulti-hotの0の数で、追加の負例サービス生成ではない。
- reports/classifier_split_summary.csv、category_classifier_counts.csv、api_classifier_counts.csv：方式・分類器件数。
- reports/fold_sample_counts.csv、label_supports.csv：Fold/role別の実数と全ラベル支持数。
- reports/rare_classifier_summary.csv、single_output_classifiers.csv、unavailable_classifiers.csv：例外・単一出力・分割不能の記録。
- reports/existing_implementation_comparison.md、implementation_report.md、summary_counts.json：既存との差分・集計・Phase 3の判断事項。
- checks/validation_results.json：自動検証結果と生成物ハッシュ。

複数構成のAPI分類器はデータ・教師・条件が一致する152個を参照。カテゴリ分類器は等価群を記録するだけで共有を適用していない。元データ・ラベルを変更せず、全構成・分類器を追跡できる形で保存した。

## 作成・検証コマンド

リポジトリ直下、Python>=3.10、標準ライブラリのみ。元の学習モジュールはimportしない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/create_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/summarize_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1 --write-report
```

既存出力には停止する。再生成は別の未使用--output-rootを指定し、既存を削除・上書きしない。--phase-one-rootで同一スキーマの将来の入力版を指定できるが、今回の固定入力件数・除外検証の条件を満たす必要がある。負例入りのPhase 3入力は正例専用ローダー・検証を別途拡張する必要があり、そのまま通す仕様ではない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/classifier_cv/tests -v
```

通常のvalidateは読み取り専用。作成日時は実行時刻となるが、教師と割当は同じ入力・seedで再現される。分類器単位のValidationをESと条件選択へ使う既存仕様を記録しているが、モデル全体の独立OOF性能の分割ではない。CV後の全データ再学習、損失・少数例外・最良epochの学習実装は次のPhase 3。
