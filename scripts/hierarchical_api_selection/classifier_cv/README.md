# 分類器単位のCV/Holdout：Phase 2の独立実装

既存のPhase 1（カテゴリ21,907、API正例4,855）と全層共通Group 5-Foldを変更せず、分類器別の教師と分割を別ディレクトリへ保存する。モデルの学習・推論、負例生成、階層再構築は行わない。

対象はカテゴリ分類器653個と、全16構成で共用するAPI分類器152個の計805個。旧実装の`n<20 and num_labels>1`とは異なり、今回は単一出力でもn<20ならCVを試みる。独立group数G<2は分割不能、n<20かつG>=2はmin(5,G)-fold、n>=20かつG>=2はgroup単位で約80/20 Holdout。常にseed42、group制約を優先する。

## 入力・出力

| コマンド | 入力 | 新規出力 |
| --- | --- | --- |
| create_classifier_splits.py | Phase 1のrecords、group ID、ラベル・固定階層定義、元データの保護情報 | manifests、805個の分類器別教師/Foldファイル |
| summarize_classifier_splits.py | 新しい分類器manifestと分割 | README、件数・例外CSV、既存実装との比較、実装報告 |
| validate_classifier_splits.py | 全成果物と保護した入力 | 通常は読み取りのみ。--write-report時はchecks/validation_results.jsonを新規保存 |

デフォルト入力は`data/hierarchical_api_selection/group_5fold_v1/`、新規出力は`data/hierarchical_api_selection/classifier_cv_v1/`。本文はPhase 1のentity_idで参照し、分類器固有のtarget_labelsを保存する。旧全層共通Fold割当は使用しない。分類器ごとに1 JSONへ全FoldのTrain/Validation ID、group、支持数をまとめ、管理ファイル数を抑える。

## 実行

リポジトリ直下からPython 3.10以降、標準ライブラリのみで実行する。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/create_classifier_splits.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/summarize_classifier_splits.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --write-report
```

各コマンドは`--output-root`で別出力先を指定できる。既存出力は上書きせず停止し、再生成には未使用の出力先を指定する。`--phase-one-root`は入力版を指定するが、現在のローダーは正例専用・固定研究条件で検証する。Phase 3の負例入りデータを使う際には、入力教師と検証の拡張を伴う新バージョンが必要。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/classifier_cv/tests -v
```

## 状態の解釈

CV/Holdoutの分割成立と、正常な学習・評価の成立を区別する。単一出力全正例、定数教師、Train正例0・負例0、pos_weight=0の可能性は警告として保存し、モデル成功とは扱わない。負例サービスは生成していない。支持数のnegative列は固定multi-hot出力の0の件数であり、追加した負例レコード数ではない。

分割後のサンプル移動・ラベル統合・学習fallbackは実装しない。分類器のCV/ Holdout ValidationをESと条件選択へ使う既存の学習条件を記録するが、今回学習は行わない。固定Validation100/Test236は一切使用しない。

API分類分割は正例のみの暫定版。負例追加時には新しい入力版から再分割する。カテゴリ分類器の共有候補は教師・意味・条件の一致を記録するだけで、今回の分割は個々の分類器に保持する。選択後の学習率・epochなども一致しなければ、将来のモデル共有を確定してはならない。

`classifier_pipeline.py`は既存Phase 1スクリプトの副作用のない射影・group分割・保護検証helperだけを読み取り利用する。既存の学習モジュールはimportしない。
