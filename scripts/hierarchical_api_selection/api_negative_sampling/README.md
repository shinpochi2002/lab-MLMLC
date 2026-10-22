# Phase 3: API負例・分類器別学習条件選択

既存Phase 1/2・既存学習/推論コードは読み取り専用。新規出力は既存パスなら拒否する。学習初期値はローカルキャッシュの `bert-base-uncased` だけを許し、既存分類モデル・Foldモデルを最終学習の初期値にしない。ネットワークダウンロードはしない。

負例はPhase 1学習Mashup3,946件の「使用API∩当該出力API=空集合」。観測上の未使用であり、本質的に不要なAPIである証明ではない。API説明文・候補外API・固定評価336件を負例にしない。元正例4,855サービスは保持する。

API自己説明文と同じgroupをTrainへ固定し、ValidationはMashup限定。まず全負例候補を含むgroupを分割し、その後Trainのみで1:1/1:3/1:5を抽出する。正例を含むgroupの負例は必ず同時採用し、それ以外はseed42と分類器UIDのSHA-256順にgroupの接頭辞を採用。要求数を満たす最後のgroupは分割せず、超過・不足を明示する。3比率でValidation・正例集合は同一。

正例20件未満かつ検証へ回せるMashup正例groupが2以上なら最大5-fold。20以上ならHoldout。API補助groupをTrain固定するため、この条件を満たさない場合は正例を全Trainに残し、lr2e-5/epoch10/負例比率1の事前固定設定を使う。負例のみの診断Validationは保存するが条件選択に使わない。

通常はlr2e-5/3e-5、最大50epoch、最良val_lossの重みを復元、patience3。各比率内で平均val_loss最小のlrを選び、比率間は同一Mashup Validationの平均micro-F1最大で選ぶ。F1同点はRecall、Precision、少ない負例の順。epochは最良epoch中央値を四捨五入して1..50。最終学習は全正例＋全開発母集団から当該比率で選んだ負例を用い、BERTから初期化し直す。

## コマンド

リポジトリ直下で実行する。Pythonのbytecodeを既存ディレクトリへ書かない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/api_negative_sampling/prepare_negative_samples.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/api_negative_sampling/validate_training_data.py --readiness-report
PYTHONDONTWRITEBYTECODE=1 /home/nakagawa/venvs/swallow-py312/bin/python -m unittest discover -s scripts/hierarchical_api_selection/api_negative_sampling/tests -v
PYTHONDONTWRITEBYTECODE=1 /home/nakagawa/venvs/swallow-py312/bin/python scripts/hierarchical_api_selection/api_negative_sampling/run_smoke_tests.py --output results/hierarchical_api_selection/phase3_smoke_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/api_negative_sampling/build_reports.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/api_negative_sampling/validate_training_data.py --write-report
```

`create_training_splits.py`は再分割を含む同seedの再現検証用。生成はprepareの中で分割→抽出の順序を不可分に実施する。既存出力へ再生成しない。`--output`で別の新規データパスを指定できる。

次は将来の本格学習用例で、今回の作業では実行しない。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nakagawa/venvs/swallow-py312/bin/python scripts/hierarchical_api_selection/api_negative_sampling/train_api_classifier.py --classifier api_classification/category_weather --output results/hierarchical_api_selection/new_production_weather
PYTHONDONTWRITEBYTECODE=1 /home/nakagawa/venvs/swallow-py312/bin/python scripts/hierarchical_api_selection/api_negative_sampling/train_category_classifier.py --classifier category_classification/layer3_12_6_2/layer_03_parent_5_3 --output results/hierarchical_api_selection/new_production_category
```

APIの本格選択は3比率×2学習率×全保存Foldを要求する。`--smoke --ratio 1`のみでは完全な選択にならず、成果物にも明記する。カテゴリ分類は既存教師・分割を参照し、比率0（負例抽出なし）、同じ2学習率・最良epoch選択を使う。CUDA利用時は起動前に決定論的演算用の `CUBLAS_WORKSPACE_CONFIG=:4096:8` を設定する。今回の検証はCPU2スレッド。

## 保存形式と限界

モデルは `bert.*`、`classifier.*` の裸のstate_dictと出力順付きラベルJSON。既存推論クラスをASTで副作用なく抽出しstrict loadとlogits一致を確認する。既存推論のモデル探索パスやファイル名は変更しないため、自動的な新モデルへの切替は行われない。推薦閾値はSigmoid>=0.5（既存コード内には>と>=の両方がある）。

Trainの支持数のみでBCE pos_weightを計算。正負両方ありならneg/pos、片方0なら1、[1e-4,1e4]へ制限。正例0の意味上の不足は重み補正では解決しない。同じValidationをESと条件選択に使うため、開発F1を独立した性能推定と呼ばない。API補助データをTrain固定するため、通常の全レコードK-foldではなくMashup開発母集団のgroup分割である。

スモークでは実物BERT全パラメータに勾配を通すが、最大長32・4サンプル・1Fold・2epoch・負例比率1へ限定。2つのlrのうち最初の最良checkpointを保存し、他方はメモリで保存復元を検証する。最終重みも保存・再ロードする。研究用の設定選択結果や推薦精度ではない。

## 正式データ版

正式な検証対象は `api_negative_sampling_v2/`。初回の `v1/` はデータ生成後にスモーク対象の分類を修正したため、コードハッシュを上書きせず保存し、新しいv2へ生成し直した。旧v1を本格学習へ使用しない。修正はweather（出力11）からcontent（出力1）へのスモーク対象変更で、データの負例定義・分割規則は変更していない。
