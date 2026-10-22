# 負例比率比較計画

API自己説明文は補助教師としてTrainに常在。同じgroupのMashupもTrainに固定する。ValidationはMashupだけで構成し、正例groupと負例groupを事前層別分割する。各分類器・Foldで3比率のValidationと正例Trainを完全に固定する。違いはTrainの負例数のみ。同一seed42、同一BERT初期値、lr候補2e-5/3e-5、最大50epoch、patience3、Sigmoid>=0.5を使う。

1. 各比率・lr・FoldでTrainだけからBCE pos_weightを計算する。
2. val_loss最小epochを保存・復元し、その重みのMashup micro Precision/Recall/F1とLossを記録する。
3. 各比率内ではFold平均val_loss最小のlrを選ぶ。同点は平均F1最大、lr小の順。
4. 比率間は選択lrのFold平均micro-F1最大。同点はRecall最大、Precision最大、比率小の順。比率間のval_lossは主選択指標にしない。Foldは等重みで平均し、全件pool micro-F1とは区別する。
5. 選択比率・lrの最良epoch中央値をround-half-upし1..50へ制限する。
6. 全開発母集団から選択比率を再抽出し、全正例を使ってBERTから最終学習し直す。Foldモデルを初期値に使わない。

例外は比率1・lr2e-5・10epochを事前固定し、負例しかない診断Validationから比率・lr・epochを選ばない。選択用ValidationとEarly Stopping用Validationは同一なので、得られるF1は開発指標。固定Validation100/Test236や階層全体の最終推薦F1ではない。

3比率のデータは全152分類器分を作成済み。小規模学習は比率1のみを実施し、比率選択の全Fold本格比較は未実行。比率選択順序・同点規則・中央値処理は合成データの単体テストで検証した。

正負不均衡に応じてpos_weightも変わる。これはTrainから決定される損失規則として全条件で固定されているが、負例数と正例重みは連動する。負例比率の効果を生の件数だけで解釈しない。観測上未使用のAPIを0とすることで、潜在的に利用可能なAPIを負例にする限界がある。
