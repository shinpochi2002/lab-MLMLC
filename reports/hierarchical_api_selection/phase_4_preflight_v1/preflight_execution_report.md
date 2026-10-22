# Phase 4 事前検証 実行報告

2026-10-21 JST。総合判定：**FAIL（GPU事前検証が環境により阻害）。本格GPU学習は開始不可。** データ接続と既存CPU単体テストはPASSであり、GPUでの実学習成功とは区別する。

## 結論

- 既存Phase 3のverify_inputs、load_classifier、safe_weights、choose_config、既存テストを再利用した。学習ループ・教師・分割は変更していない。
- GPUをPCI上で観測したが、nvidia-smiはdriver通信失敗、device nodeなし、PyTorch device_count=0、明示CUDA初期化も失敗した。GPU型番・VRAM総量/空きは確認不能。
- 要求されたAPI4種類・カテゴリ2種類を選び、既存教師・分割をロードして実行条件を保存した。GPUのforward/backward、optimizer更新、1epoch、checkpoint保存復元、実BERT推論一致は全6件で未実施。
- CPUの既存単体テスト8件はすべてPASS。Early Stoppingのモックテストと過去checkpointの読取成功を、GPU検証の代わりにはしていない。
- API152分類器の3比率は読込・選択処理への接続を確認。mappingの1:3/1:5、socialの1:5に不足あり。
- 学習回数はAPI条件選択1,278、カテゴリ条件選択1,322、API最終152、カテゴリ最終653、計3,405。GPU時間見積もりは追加のユーザー指示により対象外。
- FP32の最終805モデルは約352.6 GB。現コードの選択checkpoint全保持も含めると約1.491 TB。現在の空き約3.087 TB。保持抑制案は提示のみで、既存ファイル削除やコード変更は行っていない。

## GPU代表ケース

| 区分 | 分類器UID | 方式 | Train | Val | 出力数 | GPU結果 |
| --- | --- | --- | --- | --- | --- | --- |
| api_cv_multi | api_classification/category_internet_things | kfold | 14 | 1316 | 5 | SKIP_NO_CUDA |
| api_holdout_multi | api_classification/category_mapping | holdout | 2772 | 791 | 52 | SKIP_NO_CUDA |
| api_single_output | api_classification/category_content | holdout | 90 | 788 | 1 | SKIP_NO_CUDA |
| api_one_positive | api_classification/category_booking | fixed_config | 2 | 792 | 1 | SKIP_NO_CUDA |
| category_holdout_multi | category_classification/layer2_4_43/layer_01_parent_root | holdout | 17526 | 4381 | 4 | SKIP_NO_CUDA |
| category_fivefold | category_classification/layer3_12_6_2/layer_03_parent_5_3 | kfold | 9 | 3 | 2 | SKIP_NO_CUDA |

gpu_benchmark.csvの時間・VRAM・新checkpoint容量欄は未測定の空欄、measured_epochs=0、status=SKIP_NO_CUDA。GPU不存在を0秒・0バイトのベンチマーク成功として扱っていない。batch8・長さ512・FP32は既存本格設定を維持した予定値で、実行値ではない。

## 追加ファイルと理由

既存ファイル変更0件。本専用ディレクトリを新規作成し、下記だけを追加した。

| ファイル | 目的 |
| --- | --- |
| environment_report.md | 実環境とCUDA失敗・依存関係 |
| gpu_benchmark.csv | 代表6件の保存データ件数・未実施項目 |
| training_cost_estimate.md | 学習回数・保存容量。時間推定は対象外 |
| implementation_compatibility.md | 既存再利用箇所・静的互換性・未確認事項 |
| preflight_execution_report.md | 本報告 |
| negative_ratio_report.md | 実比率・不足分類器 |
| collect_preflight.py | 既存ローダー・入力検証を呼ぶ新規の読取集計。学習・再生成処理なし |
| build_preflight_reports.py | 確認済みJSON/CSVから本報告群を新規作成 |
| configs/representatives.json | 6分類器の保存全Train/Val ID・label順・予定条件 |
| *_inventory.csv / negative_shortages.csv | 全条件・lr・Fold・件数・不足の証跡 |
| training_cost_counts.json / storage_estimate.json / final_batch_coefficients.json | 再計算の機械可読値 |
| logs/ | コマンド終了コード、CUDA例外、既存テスト、過去checkpoint容量、入力保護確認 |
| configs/existing_file_snapshot.json / initial_context.json | 開始時のGit状態・既存ファイル保護記録 |
| artifact_manifest.json | 新規成果物の容量とSHA-256（自身を除く） |

## 実行と再現

実際に用いたPythonは既存 `/home/nakagawa/venvs/swallow-py312/bin/python`。共通環境変数はPYTHONDONTWRITEBYTECODE=1、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。既存データ出力の上書き・再生成、既存モデルの置換をしていない。集計スクリプト・報告生成はすべて専用出力先で、作成済みファイルへはexclusive-createで停止する。

既存関数の接続集計は `python collect_preflight.py`、報告生成は `python build_preflight_reports.py`。再実行は新規出力先へスクリプトを配置し直す必要があり、現出力へ再実行しない。単体テスト名と実行ログはlogs/selected_existing_tests.json/.log。GPU環境確認の各コマンドはconfigs/initial_context.jsonにcommand/returncode/stdout/stderrを保存した。CUDA例外のtracebackはlogs/python_environment.jsonに保存した。GPUに対する実試行はtorch.cuda.initであり、BERT GPU実行コマンドを成功したようには記録していない。

終了結果：環境コマンドのnvidia-smi=9、存在しないデバイス/driver情報の読取は非0。接続集計=0、選択した既存テスト=0（8/8成功）、checkpoint容量読取=0。CUDA初期化は捕捉した例外としてFAIL記録しており、環境調査プロセスの終了0とGPU成功を混同しない。

## 既存資産・レビュー

HEAD `5b8c7f5251128843b313b88304c3f2d4261e1cf0`・main。開始時にステージ済み/未ステージ差分、未追跡・ignoreを記録した。既存未追跡のAGENTS.md・Phase1〜3・監査成果物等を保持した。既存15,434ファイルのメタデータ比較で変更0件、保護入力/コード1,182ハッシュ一致。Git追跡ファイルの差分は0。モデル全体の内容ハッシュ比較ではなく、既存モデルはmetadata確認と7ファイルの読取容量確認の範囲である。

AGENTS.mdに沿った確認は実施したが、別セッションの独立レビューは未実施。同じセッションの自己点検を独立レビュー完了として扱っていない。理解度確認を1問提示したところ、ユーザーから「時間見積もりは必要ない」と追加指示を受け、見積もり範囲を変更した。この返答を技術的理解確認済み・実験承認済みとは扱っていない。

## 残る問題と最小限の対応

1. **GPU環境**：ホスト/セッションのNVIDIA driverとdevice公開状態を、環境管理者側で既存構成に沿って確認・復旧する必要がある。原因は未確定で、こちらからdriver更新・インストール・module操作・再起動はしていない。
2. **GPU実行未確認**：復旧後に保存した代表6件で既存train_oneを使い、forward/backward、optimizer、有限loss、lr反映、出力次元、最良epoch、保存復元、旧推論互換を実確認する。CPU成功やデータ監査PASSだけではGPU開始可としない。
3. **checkpoint保持**：現コードには保持抑制引数がない。全条件を無条件に永続保持する新設計は行わず、将来の新規学習について小さいopt-in保持設定の案を提示した。今回は未実装であり、自動削除はしない。
4. **負例不足**：上記2 UIDの3条件は実比率を記録して報告する。足りない分を再生成・複製して埋めていない。

分割方式はユーザーの今回の研究方針として確定しており、作り直しは不要。現時点の本格学習開始を妨げる主因はGPU利用不能とGPU検証未完了である。既存実装の全面改修・学習方式再設計は不要と判断したが、GPU復旧前にGPU学習の成功を保証することはできない。
