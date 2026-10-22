"""Write new Phase 4 reports only; never write existing code/data/results."""
import pathlib,json,csv,collections,os,hashlib,subprocess,ast
P=pathlib.Path(__file__).resolve().parent;R=P.parents[2]
read=lambda p:json.loads(pathlib.Path(p).read_text())
rows=lambda p:list(csv.DictReader(open(P/p)))
env=read(P/'logs/python_environment.json');c=read(P/'training_cost_counts.json');st=read(P/'storage_estimate.json');tests=read(P/'logs/selected_existing_tests.json');bench=rows('gpu_benchmark.csv');ratios=rows('negative_ratio_inventory.csv');ctx=read(P/'configs/initial_context.json')
def put(name,s):
    with (P/name).open('x') as f:f.write(s.strip()+'\n')
def table(headers,data):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)
def gb(n):return f'{n/10**9:,.3f} GB（{n/2**30:,.3f} GiB）'
modtable=table(['モジュール','import','実バージョン'],[(k,'PASS' if v['available'] else 'FAIL',v.get('version',v.get('error'))) for k,v in env['modules'].items()])
reptable=table(['区分','分類器UID','方式','Train','Val','出力数','GPU結果'],[(b['case'],b['classifier_id'],b['method'],b['train_count'],b['validation_count'],b['output_labels'],b['status']) for b in bench])
smallratio=table(['分類器（api_classification/ 以下）','要求比率','Train正','Train負','実比率','Val'],[(r['classifier_uid'].split('/',1)[1],r['ratio'],r['positive'],r['negative'],f"1:{float(r['actual_ratio']):.4f}",r['validation']) for r in ratios if r['fold_id']=='fold_01' and r['classifier_uid'] in {b['classifier_id'] for b in bench if b['task']=='api'}])
shortage=table(['分類器','比率','用途','正例','実負例','要求負例','実比率'],[(r['classifier_uid'],r['ratio'],r['fold_id'],r['positive'],r['negative'],r['requested_negative'],f"1:{float(r['actual_ratio']):.4f}") for r in rows('negative_shortages.csv')])
put('environment_report.md',f'''
# Phase 4 環境確認

監査日：2026-10-21 JST。対象HEAD `{ctx['commands']['git_head']['stdout'].strip()}`、branch `{ctx['commands']['git_branch']['stdout'].strip()}`。実行環境はLinux aarch64、kernel `7.0.0-1019-nvidia`。RTX 4090/A6000が利用できるという前提は置いていない。

**FAIL：このセッションから利用可能なCUDA GPUは0枚。GPU実測は未実施。**

| 確認項目 | 実測した状態 |
| --- | --- |
| nvidia-smi | 終了コード9、NVIDIA driverと通信できない |
| NVIDIAデバイスノード | `/dev/nvidia*` は存在しない |
| NVIDIA kernel driver公開情報 | `/proc/driver/nvidia/version` は存在しない |
| kernel module | lsmodでnvidia/nouveauの行なし（検索終了コード1） |
| PCI上の表示装置 | `000f:01:00.0`、NVIDIA、PCI ID `10de:2e12`、VGA controller 1件を観測 |
| GPUの商品名・実枚数・VRAM総量/空き | ドライバ経由で確認不能。PCI表示1件をCUDA GPU1枚として数えていない |
| PyTorch CUDAビルド | torch {env['modules']['torch']['version']}、CUDA {env['cuda']['compiled_version']}、cuDNN {env['cuda']['cudnn_version']} |
| PyTorch認識 | is_available=False、device_count=0 |
| CUDA明示初期化 | `torch.cuda.init()` → RuntimeError: No CUDA GPUs are available |
| CPU | 20 logical CPU、ARM Cortex-X925 / Cortex-A725（lscpu出力を保存） |
| RAM | 128,517,992,448 byte（約119.7 GiB）、MemAvailable 123,860,340,736 byte（約115.4 GiB） |
| Swap | 約16 GiB、観測時使用0 |
| 保存先 | `/home/nakagawa` と `/tmp` は同じ `/dev/nvme0n1p2` |
| 空き容量 | 取得時約{gb(st['free_disk_bytes_at_check'])} |
| Python | `{env['executable']}`、Python 3.12.3 |

CUDA対応PyTorchがimportできても、CUDA deviceが利用可能とは限らない。ドライバ未稼働・未公開状態の原因の特定や、ホスト側の復旧は本監査の実施範囲に含めていない。ドライバ導入・moduleロード・再起動・依存パッケージ更新は行っていない。

## 既存ライブラリ

{modtable}
requirements.txtはバージョンを固定していないため、今回の実importバージョンを再現情報として保存した。追加インストールはしていない。システムPythonへの切替や別GPU環境の推測もしていない。

## BERTキャッシュ

`{env['bert_cache']['path']}` にconfig、vocab、tokenizer、model.safetensors（440,449,768 byte）が存在する。`snapshot_download(..., local_files_only=True)` のローカル解決だけを実行した。今回BERTをGPUにロードできたという意味ではない。HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、PYTHONDONTWRITEBYTECODE=1を使用し、モデルダウンロードは行っていない。

## 既存GPU設定・出力先

Phase 3 `training.main` の `--device` の既定値はcpuで、GPU指定は `--device cuda:0`。実学習はbatch_size=8、max_length=512、FP32、autocast/GradScalerなし。`torch.use_deterministic_algorithms(True)` を使用し、CUDA起動時は `CUBLAS_WORKSPACE_CONFIG=:4096:8` が前提。旧Lightning学習コードはaccelerator=gpu・devices=1、batch8で、明示的なmixed precision設定はない。matmul_precision(high)はmixed precisionの有効化とは区別した。

Phase 3保存先は必須の--output引数で指定し、既存ディレクトリなら停止する。checkpointは `best_ratio_<r>_<fold>_lr_<lr>.pth`、最終重みはmodel_weights.pth、ラベルJSON、run_manifest.json、training_results.jsonを保存する。永続的なコンソールログ収集は外側で行う必要がある。今回の専用出力先は本ディレクトリのみで、学習checkpointは新規作成していない。

生ログ：logs/nvidia_smi.log、logs/nvidia_devices.log、logs/kernel_driver.log、logs/modules.log、logs/gpu_pci.log、logs/python_environment.json、logs/cpu.log、logs/memory.log、logs/disk.log。
''')
put('implementation_compatibility.md',f'''
# 既存実装の再利用・互換性

**データ接続はPASS、GPU実行の互換性は未確認。既存コード変更は0件。**

対象実装：`scripts/hierarchical_api_selection/api_negative_sampling/training.py`、同pipeline.py、train_api_classifier.py、train_category_classifier.py。正式入力はPhase 1 group_5fold_v1、Phase 2 classifier_cv_v1、Phase 3 api_negative_sampling_v2。旧v1 API負例や旧共通Foldへ切り替えていない。既存コードを実際に読み、既存ローダーで6代表をロードし、既存verify_inputsで入力・コード・生成物の保存ハッシュを照合した。再生成関数は呼んでいない。

## 再利用できたもの

| 処理 | 実装と今回の確認 |
| --- | --- |
| 入力保護 | training.verify_inputs(正式v2)を読み取り実行し成功。manifestハッシュの更新なし |
| データローダー | training.load_classifierを6代表に実行。本文はPhase 1 unified_servicesをentity_idで参照、教師は分類器members.target_labels。すべて解決 |
| 既存分割 | APIはv2の比率別JSON、カテゴリはPhase 2保存folds。旧コードの内部再分割を呼ばない |
| 比率接続 | API152分類器×3条件を読み、各FoldのValidation同一・正例不変・負例集合の包含と件数を確認 |
| 重み | 既存safe_weightsへ保存Trainだけを渡し、全API条件で保存重み一致。正例/負例0の保護あり |
| 条件選択 | choose_configが全保存Fold×2lrを要求し、val_lossでlr、APIの比率はF1→Recall→Precision→少ない負例、epochは最良epoch中央値 |
| Early Stopping | train_oneの最大50epoch・patience3・最良重み復元を静的確認。既存CPUモックテストでbest_epoch=2、停止epoch=5、復元を実確認 |
| 最終再学習 | 選択条件でtrain_oneを再初期化、early_stop=False。カテゴリは全members、APIは選択比率の保存final_trainingを使用 |
| 保存・推論形式 | bert.* / classifier.* state_dict＋固定順ラベルJSON。既存legacy_inference_classは旧推論モジュール全体を起動せずクラスASTだけ抽出 |
| 既存checkpoint読込 | 既存7個のpthをtorch.load(weights_only=True,mmap=True,map_location=cpu)で読取。1/5/52出力のLinear形状とFP32容量を確認 |

40,099件の接続チェックはlogs/data_connection_checks.jsonに記録。これはGPU上でモデルを動かした検証数ではない。

## 旧学習コードをそのまま起動しない理由

旧 `mashup_clu_mlc_nosp_nostop_layer1〜4.py` はファイル順80/20を内部生成し、観測ラベルから出力を作り、全study由来pos_weightを使う箇所がある。旧API category_kfold.pyも内部CV/Holdout、ValidationからTrainへの救済移動、固定50epoch最終学習を含む。これらを起動すると、今回維持するPhase 2/3の分割・固定出力・選択方針をそのまま使えない。既存のPhase 3エンジンを優先する方が変更が少なく、現在の研究方針と整合する。

ユーザーがカテゴリの現CV/Holdout維持を今回明示したため、前回監査の全層K-fold方針に関する保留を今回の阻害要因へ持ち越していない。APIの固定設定55件も既存方針として維持した。固定Validation100は階層選定、Test236は最終評価という役割を維持し、今回いずれも学習・条件選択・性能評価に使用していない。

## 代表6件

{reptable}
Train/Valは保存fold_01の全件数。APIは負例比率1、カテゴリは負例抽出なし。configs/representatives.jsonには全ID、固定出力順、lr、batch8、max_length512、FP32、予定epoch1を保存した。**GPU初期化失敗によりこの設定ではまだ実行していない。** bookingのValidationは負例のみの診断用で、条件選択には接続しない。分類器数を条件選択可能な件数と混同しない。

## 既存テストの今回の実行

既存tests/test_phase3.pyの8テストを選択実行し、8 PASS・FAIL0・ERROR0・SKIP0。対象は重み、空Train拒否、lr/比率の選択と同点処理、不完全/非有限結果の拒否、Early Stopping復元、単一出力lossの勾配。splitter/generatorを実行するテストは選択していない。

Early StoppingテストはCPUの小型モデル・tokenizer・Validation値をモックする既存テストで、実BERTやGPUの検証ではない。単一出力の勾配テストもCPUのlossテストである。GPUのforward/backward・optimizer更新・lr実適用・最良epoch保存・GPU checkpoint round-trip・旧推論とのlogits一致は、すべて未実施と区別した。過去Phase 3のBERTスモーク報告を今回のGPU成功として引用していない。

## 必要最小限の追加・残る制約

今回は新規の読取集計・報告生成スクリプトだけを追加し、学習ループは新設していない。GPU復旧後の1epoch確認には既存training.train_oneへepochs=1、batch_size=8、max_length=512、device=cuda:0を実行引数として渡せる。CLIには汎用--epochs/--batch-size引数がなく、--smokeはmax_length32・batch2・Train/Val各最大4件・1Fold・2epochへ制限する。このスモーク結果を通常512トークンの1epoch測定と呼ばないこと。

GPUが使える状態で、既存train_oneを呼ぶ外側の薄い実行・メモリ計測ラッパーを加えればGPU動作確認できる見込みだが、今は作成・実行していない。1epochだけではpatience3の停止発火までは確認できないため、停止発火のGPU検証は別に必要。推論の出力形式は静的互換だが、CUDA実行・保存復元・実BERT logitsの一致は保証しない。

ストレージ節に記したように、本格runは条件ごとの最良checkpointをすべて永続保存する。checkpoint保持方針を抑制する実行引数はない。保持数を制御する最小変更は将来案として提示し、今回はコードを自動修正していない。
''')
costtable=table(['工程','計算根拠','学習回数'],[('API条件選択','(166 CV Fold + 47 Holdout) × 3比率 × 2lr',1278),('カテゴリ条件選択','(2×5 CV Fold + 651 Holdout) × 2lr',1322),('API最終再学習','97選択対象 + 55固定設定',152),('カテゴリ最終再学習','全653分類器',653),('合計','APIは16構成で共用し16倍しない',3405)])
storage_table=table(['保存対象','算出容量'],[('API最終152モデル',gb(st['final_bytes']['api'])),('カテゴリ最終653モデル',gb(st['final_bytes']['category'])),('最終モデル計805',gb(st['final_only_bytes'])),('API条件選択checkpoint 1,278個',gb(st['selection_checkpoint_bytes']['api_selection'])),('カテゴリ条件選択checkpoint 1,322個',gb(st['selection_checkpoint_bytes']['category_selection'])),('現実装ですべて保持した計3,405個',gb(st['all_retained_bytes']))])
put('training_cost_estimate.md',f'''
# 学習回数・保存容量の算出

**ユーザーの追加指示「時間見積もりは必要ない」に従い、GPU時間の見積もりは実施しない。** GPUが利用できないため、1epoch秒数・VRAMの新規実測値も存在しない。空欄を0秒・0GBとして集計したり、CPUテストや過去の小規模スモークをGPU時間へ換算したりしていない。

## 学習回数の再計算

{costtable}
API CV50分類器のFold内訳は2-fold20、3-fold10、4-fold4、5-fold16。合計40+30+16+80=166 Fold。Holdout47を足して213組。固定設定55は保存診断Foldがあっても条件選択0回である。最終再学習は選ばれた比率1つだけで、152×3としていない。固定設定55は既存の比率1・lr2e-5・10epochを使う。

カテゴリは5-fold2個、Holdout651個、計661組。新実装はHoldoutでも2lrを比較する。lr/epochを選ぶための条件別学習2,600回と、新規初期化からの最終805回を分けた結果、3,405回という元報告の値を再現した。最大50epochを掛けてGPU所要時間を断定する方法は用いていない。

全条件のUID・比率・Fold・lr・batch数はselection_run_inventory.csv、集計はtraining_cost_counts.json。必要なら確認できるデータ量の係数も記録した。全選択runを1epochずつ処理する場合、APIはTrain16,896batch／Val176,322batch、カテゴリはTrain192,424batch／Val48,560batch（batch8）となる。APIは小さいTrainでもValidationが大きく、Train件数だけを計算負荷の指標にすることはできない。これは計算時間の推定ではない。

## 保存容量

過去Phase 3の実ファイル7個を今回読み取り、1モデルは438,010,669〜438,167,533 byteだった。これは過去checkpointの容量実測であり、新GPU実行のcheckpoint容量ではない。既存FP32 state_dictのBERT部分437,928,960 byte、出力1個あたりLinear重み・bias3,076 byte、既存torch serialization overhead最大78,633 byteから、各分類器の固定出力次元に応じて算出した。バージョン等による保存形式差はあり得る。

{storage_table}
観測時の空きは約{gb(st['free_disk_bytes_at_check'])}。最終805モデルの約352.6 GBに対しては余裕がある。現実装で全選択checkpointを残す場合は約1.491 TBで、残りは概算1.60 TBとなる。ログ・既存の並行使用分・安全余裕は別枠で確保する必要があり、ストレージが足りてもGPUが利用可能になるわけではない。

run_manifestやラベルJSON、training_resultsは通常のテキストで、全epochの重みは含まない。ただしtraining_resultsはhistoryと全runのTrain/Val ID、支持数、hashを保持するため、合計容量は実行内容に依存する。今回生成した報告・CSV・設定は数MB規模で、最終実測合計をartifact_manifest.jsonへ記録する。将来の全ログ容量の上限は確認していない。

GPU用のactivation/gradient/Adam状態はVRAMまたはRAMであり、checkpointのディスク容量と同一ではない。既存runはAdam optimizer状態を保存せず、resume完全checkpointではない。GPU offload用ディスクキャッシュや専用一時領域をこのエンジンは定義していない。GPU一時VRAM容量は未実測。

## checkpointの保持方針

現コードは各epochを保存する方式ではなく、各条件の最良stateをメモリに保持して、run終了後に条件ごと1個保存する。最大epoch50をcheckpoint数へ掛けない。ただし本格実行では2,600条件分をすべて保存するため、そのままでは大量保持になる。今回はGPU不在のため新規学習checkpointを一切保存していない。

提案は、将来の新規本格出力でのみ、既存runへopt-inのcheckpoint保持引数を小さく追加すること。既定動作を維持しつつ、final-onlyまたはselected-onlyを選べるようにし、全条件の設定・seed・入力/コードhash・history・best_epoch・最良state hash・選択理由は保持する。最終モデルはFoldモデルを使い回さずBERTから再初期化する既存仕様を維持する。全epochの保存や新しい損失/選択方式は導入しない。

最終のみ保持する場合のモデル容量は約352.6 GB。選択checkpointを1分類器単位で逐次処理する案では、既存全条件を一時保持した場合の最大working set概算は約{gb(st['largest_serial_classifier_checkpoint_working_set_bytes'])}だが、既存runには自動抑制機能がない。既存ファイルの削除は提案の実施対象にせず、今回は何も削除・圧縮・移動していない。保持方針の変更は未実装案として扱う。
''')
put('negative_ratio_report.md',f'''
# API負例3比率の接続確認

**PASS：152分類器の1:1・1:3・1:5保存データを読み込み、既存学習処理に接続できる。WARNING：比率不足が3条件で存在する。GPU学習による比率比較は未実施。**

分類器ごとに保存全FoldのValidation ID順・集合が3条件で同一、正例Train集合が不変、負例が包含関係を満たすことを確認した。比率別のTrain件数、実比率、負例不足、group丸め超過はnegative_ratio_inventory.csvへ全件記録した。データ生成・再抽出・再分割を行っていない。API説明文のTrain固定方針も保持した。

## 代表のfold_01

{smallratio}
bookingは正例1件のため3条件が保存されていても比率・lrを比較しない。固定設定としてratio1・lr2e-5・10epochを適用する既存経路である。ほかの97選択対象は3条件×2lr×全保存Foldのchoose_configへ接続する。3条件すべての本格学習は今回行っていない。

## 負例不足の全件

{shortage}
不足は1:1が0分類器、1:3がmappingの1分類器、1:5がmappingとsocialの2分類器。各条件で選択用fold_01とfinal_refitの両方に不足がある。利用可能負例を上限としており、要求比率を満たすための複製・追加生成はしていない。

mappingではratio3とratio5のTrain集合が同じ上限に達しており、2条件を比較しても「異なる実比率の効果」の比較にはならない。一方、現実装の条件選択はこの不足を記録した条件を候補として扱える。ratio名称だけで実比率が3/5に達した実験と報告しないこと。修正ではなく、実比率・不足とともに結果を記述する必要がある。
''')
before=read(P/'configs/existing_file_snapshot.json');changed=[]
for rel,v in before.items():
    p=R/rel
    if not p.exists() and not p.is_symlink():changed.append({'path':rel,'reason':'missing'});continue
    s=p.lstat();now=[s.st_size,s.st_mtime_ns,s.st_mode,os.readlink(p) if p.is_symlink() else None]
    if now!=v:changed.append({'path':rel,'reason':'metadata_changed'})
cksrc=R/'scripts/hierarchical_api_selection/api_negative_sampling/training.py'
ast.parse(cksrc.read_text())
protected=[]
for root in ['group_5fold_v1','classifier_cv_v1','api_negative_sampling_v2']:
    base=R/'data/hierarchical_api_selection'/root;src=read(base/'manifests/source_manifest.json')
    for k in ['source_files','input_sha256','implementation_sha256']:
        for rel,h in src.get(k,{}).items():
            ok=hashlib.sha256((R/rel).read_bytes()).hexdigest()==h;protected.append({'path':rel,'matches':ok})
assert not changed,changed
assert all(x['matches'] for x in protected)
with (P/'logs/final_protection_check.json').open('x') as f:json.dump({'existing_snapshot_count':len(before),'existing_changes':changed,'protected_hash_checks':len(protected),'protected_hash_failures':[],'new_output_scope':str(P),'existing_tracked_diff':subprocess.run(['git','diff','--stat'],cwd=R,capture_output=True,text=True).stdout,'staged_diff':subprocess.run(['git','diff','--cached','--stat'],cwd=R,capture_output=True,text=True).stdout},f,ensure_ascii=False,indent=2)
put('preflight_execution_report.md',f'''
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

{reptable}
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

HEAD `{ctx['commands']['git_head']['stdout'].strip()}`・main。開始時にステージ済み/未ステージ差分、未追跡・ignoreを記録した。既存未追跡のAGENTS.md・Phase1〜3・監査成果物等を保持した。既存{len(before):,}ファイルのメタデータ比較で変更0件、保護入力/コード{len(protected):,}ハッシュ一致。Git追跡ファイルの差分は0。モデル全体の内容ハッシュ比較ではなく、既存モデルはmetadata確認と7ファイルの読取容量確認の範囲である。

AGENTS.mdに沿った確認は実施したが、別セッションの独立レビューは未実施。同じセッションの自己点検を独立レビュー完了として扱っていない。理解度確認を1問提示したところ、ユーザーから「時間見積もりは必要ない」と追加指示を受け、見積もり範囲を変更した。この返答を技術的理解確認済み・実験承認済みとは扱っていない。

## 残る問題と最小限の対応

1. **GPU環境**：ホスト/セッションのNVIDIA driverとdevice公開状態を、環境管理者側で既存構成に沿って確認・復旧する必要がある。原因は未確定で、こちらからdriver更新・インストール・module操作・再起動はしていない。
2. **GPU実行未確認**：復旧後に保存した代表6件で既存train_oneを使い、forward/backward、optimizer、有限loss、lr反映、出力次元、最良epoch、保存復元、旧推論互換を実確認する。CPU成功やデータ監査PASSだけではGPU開始可としない。
3. **checkpoint保持**：現コードには保持抑制引数がない。全条件を無条件に永続保持する新設計は行わず、将来の新規学習について小さいopt-in保持設定の案を提示した。今回は未実装であり、自動削除はしない。
4. **負例不足**：上記2 UIDの3条件は実比率を記録して報告する。足りない分を再生成・複製して埋めていない。

分割方式はユーザーの今回の研究方針として確定しており、作り直しは不要。現時点の本格学習開始を妨げる主因はGPU利用不能とGPU検証未完了である。既存実装の全面改修・学習方式再設計は不要と判断したが、GPU復旧前にGPU学習の成功を保証することはできない。
''')
# Check report integrity and write a hash inventory for all newly created artifacts.
required=['environment_report.md','gpu_benchmark.csv','training_cost_estimate.md','implementation_compatibility.md','preflight_execution_report.md']
assert all((P/x).is_file() for x in required)
assert len(bench)==6 and all(b['status']=='SKIP_NO_CUDA' and b['measured_epochs']=='0' for b in bench)
assert c['total_runs']==3405 and tests['successful']
inv={str(p.relative_to(P)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in P.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
with (P/'artifact_manifest.json').open('x') as f:json.dump({'scope':'new Phase4 artifacts only','files':inv,'total_bytes_excluding_self':sum(x['bytes'] for x in inv.values()),'gpu_benchmark_status':'SKIP_NO_CUDA','overall_status':'FAIL_GPU_UNAVAILABLE','time_estimate':'not requested after user clarification'},f,ensure_ascii=False,indent=2)
print(json.dumps({'reports':required,'files':len(inv)+1,'bytes_excluding_manifest':sum(x['bytes'] for x in inv.values()),'existing_changes':len(changed),'protected_hash_checks':len(protected),'status':'FAIL_GPU_UNAVAILABLE'},ensure_ascii=False,indent=2))
