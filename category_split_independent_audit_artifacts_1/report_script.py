import pathlib,json,csv,collections,hashlib
R=pathlib.Path('/home/nakagawa/lab-MLMLC');A=R/'category_split_independent_audit_artifacts_1'
read=lambda p:json.loads(p.read_text())
s=read(A/'summary.json'); assert not s['errors']
rows=lambda n:list(csv.DictReader(open(A/(n+'.csv'))))
cl=rows('classifiers');sp=rows('splits');ls=rows('label_supports');counts=rows('layer_counts')
# Directly check EVERY fixed-evaluation saved variant, not only manifest membership.
M=R/'data/splits/model_selection';fm=read(M/'manifest.json');fixed_checks=[]
for rel,entry in fm['datasets'].items():
    for role in ['validation','test']:
        p=M/role/rel;d=read(p)
        ok=set(d)==set(fm['splits'][role]['keys']) and len(d)==fm['splits'][role]['count'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'][role]
        assert ok,str(p)
        fixed_checks.append({'path':str(p.relative_to(R)),'role':role,'count':len(d),'keys_and_hash_match':ok})
with (A/'fixed_variant_checks.json').open('x') as f:json.dump(fixed_checks,f,ensure_ascii=False,indent=2)
warnings=[r for r in ls if int(r['train_positive'])<=2 or int(r['train_negative'])<=2 or int(r['validation_positive'])<=1 or int(r['validation_negative'])<=1]
with (A/'sparse_support_warnings.csv').open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(ls[0]));w.writeheader();w.writerows(warnings)
def table(headers,data):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)
by=collections.defaultdict(dict)
for r in counts:by[r['pattern']][int(r['layer'])]=int(r['actual'])
counttable=table(['構成','第1層','第2層','第3層','第4層','合計'],[(p,*[n.get(i,'—') for i in range(1,5)],sum(n.values())) for p,n in by.items()]+[('合計',*[sum(n.get(i,0) for n in by.values()) for i in range(1,5)],653)])
cvtable=table(['分類器UID（category_classification/ 以下）','Fold','Train','Validation','Validation比率'],[(r['classifier_uid'].split('/',1)[1],r['fold_id'],r['train'],r['validation'],f"{float(r['validation_ratio']):.2%}") for r in sp if r['method']=='kfold'])
range_table=table(['支持数','最小','最大','2以下の出力数','5以下の出力数'],[(k,*s['support_ranges'][k],s['small_support'][k]['2'],s['small_support'][k]['5']) for k in ['train_positive','train_negative','validation_positive','validation_negative']])
smalltable=table(['分類器UID（category_classification/ 以下）','Fold','ラベル','Train正','Train負','Val正','Val負'],[(r['classifier_uid'].split('/',1)[1],r['fold_id'],r['label'],r['train_positive'],r['train_negative'],r['validation_positive'],r['validation_negative']) for r in ls if int(r['train_positive'])<=2 or int(r['train_negative'])<=2])
nametable=table(['名前','保持API entity_id','除外された元Mashup entity_id','正規化本文一致'],[(r['name'],r['entity_id'],r['evaluation_entity_id'],'なし') for r in s['same_name_distinct_apis']])
totalchecks=sum(s['checks'].values())
report=f'''# カテゴリ分類器653個の教師データ・Train/Validation分割の独立監査

監査日：2026年10月21日（JST）。対象：`/home/nakagawa/lab-MLMLC`。

## 1. 総合判定

**WARNING。データ・構造・分割の整合性はPASS。研究方針上の全層CVへの適合と、希少ラベルの評価安定性に注意が必要。**

653分類器すべての教師が元データと固定階層からの独立再計算と一致した。661組すべてで非空、entity分離、正規化本文group分離、母集合の完全被覆、保存支持数との一致を確認した。固定Validation100/Test236の元Mashup実体・正規化本文の混入は0件。Train正例0・Train負例0・Validation正例0の出力はすべて0件だった。

一方、現在の5-foldは2分類器だけで、651分類器はHoldoutである。「カテゴリ分類とAPI分類の全部の層でクロスバリデーション」を全分類器のK-foldという要件で読む場合、現実装は満たさない。またValidation正例1件の出力が延べ88件あり、0件でないことだけで評価が十分とは判断できない。

**本格学習への判定：現在のCV/Holdout方針を採用する技術上のデータ準備は可。ただし、教授の全層CV指示への適合を前提とする研究本番の一括学習は、方式を確定するまで保留を推奨する。** 本監査では仕様変更・修正・学習を行っていない。監査PASSは推薦性能、収束、GPUでの学習成功を保証しない。

## 2. 対象・独立性・保護

正式な監査対象は `data/hierarchical_api_selection/classifier_cv_v1/splits/category_classification/` の653 JSONと、そのmanifestである。`group_5fold_v1` はPhase 1の元サービスと固定階層の参照元であり、そこに残る全層共通Fold・内部ES分割は今回の661組に数えない。`api_negative_sampling_v2` はPhase 3のコード接続・履歴ハッシュの比較対象であり、そのAPI負例分割をカテゴリの分割として検査していない。

参照報告書は `phase_1_2_implementation_summary.md`、`classifier_cv_execution_report.md`、`phase_3_execution_report.md`。既存の「passed」記録だけを根拠とせず、Python標準ライブラリによる独立スクリプトで照合した。既存のreconstruct、project_classifiers、classifier_definitions、stratify、make_partition、validate等をimport・呼び出し・コピーして検証する方法は用いていない。既存コードは規約と静的な制御経路の確認に読んだ。

手順は、元Mashup/APIから実体と主カテゴリを対応付け、元階層のcategory_routesから木を構成し、各サービスのカテゴリ経路を親ノードごとの集合へ集約する方法である。本文は独立に `html.unescape → Unicode NFKC → split/join` で正規化し、大文字小文字は保持した。モデル入力本文そのものは変更していない。分割JSONのID列を集合として照合し、支持数は再計算した教師集合から数えた。分割の再生成や割当器の再実行はしていない。

確定集計の失敗は0件、検査条件は延べ{totalchecks:,}件（独立したテストケース数ではない）。保存時点の入力・コード・成果物ハッシュ3,288照合も全一致。監査中の既存15,420ファイルのsize/mtime/mode/link比較で変更0件。モデル全体の内容ハッシュは取得しておらず、全モデルのビット単位不変を主張するものではない。

## 3. 16構成別・層別の分類器構造：PASS

以下は元階層から期待する数と、manifest・実ファイルで観測した数が一致した値。層はカテゴリ分類層だけを数える。

{counttable}
全16元定義のcategory_routes、category_paths、Phase 1保存category_pathsが一致し、各構成152カテゴリを被覆した。全653親経路に対応する分類器を一対一で確認し、未定義親・未定義出力、欠落、余分なJSON、UID重複は0件。分類器ID単体は構成間で同じ値を取り得るため、構成を含むclassifier_uidで識別した。

出力は中間層では子までの完全経路文字列、最終カテゴリ層ではカテゴリ名と照合した。出力順は保存スキーマと同じ文字列の辞書順であり、数値順へ並べ替えていない。653件でmanifestとJSON双方の順序が一致した。出力次元は2〜51、単一出力分類器は0件。42組の構成×層の個数照合もすべて一致した。

## 4. 教師データ：PASS

元データは `data/raw/mashup_data/mashup_data_new.json`、`data/processed/softfill_mashup_data_new_api/nocategory/softfill_mashup_new_data.json`、`data/raw/new_api_data/api_data_new.json`、`data/raw/new_api_data/new_api_data_to_4category.json` を直接読み取った。APIの主カテゴリは元APIレコードの先頭カテゴリ、Mashupのカテゴリは使用APIそれぞれの主カテゴリの集合として再計算した。元カテゴリ列に含まれる他のタグを、分類対象カテゴリへ勝手に追加していない。

固定評価元実体・本文一致・空本文を独立に除外すると、保存済みカテゴリサービス集合と完全一致した。

| 集計 | 独立再計算値 |
| --- | ---: |
| カテゴリ用ユニークentity | 21,907 |
| Mashup | 3,946 |
| API説明文 | 17,961 |
| 正規化本文group | 21,884 |
| 元候補からの除外実体 | 338 |
| 分類器への延べ割当 | 959,057 |
| 2個以上のtargetを持つ延べ割当 | 20,854 |
| 教師完全一致分類器 | 653 / 653 |
| 分類器内entity重複 | 0 |
| 親に属さないtarget・重複target | 0 |

21,907件すべての本文・元実体・教師カテゴリを元データと照合し、カテゴリJSONとunified_servicesも完全一致した。各構成のrootに全件が属するため、一部サービスを未使用にしていない。複数カテゴリから同じ子ノードへ達する場合はその子を1回だけ立て、別の子に達する場合は両方を立てる集合演算で、保存target_labelsを分類器単位で照合した。異なる分類器での同一サービス再利用は正常として扱った。

出力はデータ内に観測されたラベルから定義せず、元階層の全ての子から確定したものと比較した。固定出力の欠落・縮小は0件。今回のカテゴリ群は全出力に実際の支持があり、API分類器で報告された単一出力・全正例の問題を転記していない。

Phase 3のsource_manifestに保存されたPhase 1/2の入力ハッシュと現在値が一致し、再計算教師も一致した。Phase 3のカテゴリ参照はPhase 2 JSONへの参照で、API負例の抽出対象ではない。カテゴリ教師がAPI負例処理によって変更された証拠は検出されなかった。

## 5. Train/Validation全661組：PASS

| 検査 | 対象・結果 |
| --- | --- |
| Train・Validationとも非空 | 661 / 661 |
| 同一entityの両側混入 | 0 / 661 |
| 同一group・正規化本文の両側混入 | 0 / 661 |
| Train∪Validation＝全教師集合 | 661 / 661 |
| 各側内のID重複 | 0 / 661 |
| JSON保存正負支持数との一致 | 全1,322 role profile |
| CSV保存正負支持数との一致 | 全6,170 role×出力 |
| CVの各groupのValidation出現 | 2分類器とも正確に1回 |
| Holdoutの規定Validation group数 | 651 / 651 |

### 5.1 5-fold対象2分類器の全Fold

`layer2_51_3/layer_02_parent_39` は18件・18group、`layer3_12_6_2/layer_03_parent_5_3` は12件・12groupである。各Foldの教師・支持数も個別照合した。

{cvtable}
### 5.2 Holdout651分類器

Validation件数比率は18.5185%〜22.4490%、group比率は18.5185%〜20.9302%。Trainは20〜17,527件、Validationは5〜4,384件。`floor(0.2×G+0.5)` を1〜G−1へ制限したValidation group数に全件一致した。サービス数の20%厳密一致は仕様ではない。

件数比率最小は `category_classification/layer3_12_6_2/layer_03_parent_10_2`（Train22／Val5、全27件）、最大は `category_classification/layer3_12_6_2/layer_03_parent_11_4`（Train38／Val11、全49件、Val10group）。**全651件の実際の件数・group比率は付録splits.csvに記録**している。

### 5.3 分割後の移動の有無

Phase 2のmake_partitionはgroup割当から相補的なID列を直接作り、不成立なら停止する。旧APIコードのsplit_with_all_labelsにはValidation→Train移動があるが、このカテゴリ分割経路から呼ばれていない。Phase 3のカテゴリloaderも保存IDをそのまま使い、本格処理では部分抽出・移動しない。保存manifestのpost_partition_moves=0、当時の成果物ハッシュ一致、現在のgroup容量と全被覆の一致をあわせ、**移動の形跡なし**と判断した。

ただし、分割前後のイベントログは保存されていないため、任意の過去の操作が一切なかったことを履歴から証明するものではない。希少ラベルを優先する割当そのものは層化方針であり、分割完了後に検証データを救済移動することとは区別した。最終再学習で全開発データを使うことも、条件選択中の移動とは異なる。

## 6. 固定評価との分離：PASS

固定Validation100/Test236のkeysを元Mashupへ対応付けて元source_idを取得した。**固定評価manifestのidsは評価データ内の再採番IDであり、全336件で元Mashup IDと異なる。idsへ単にmashup:を付ける検査は誤りになる。** 元データとの対応付け結果をfixed_evaluation_id_map.csvに保存した。独立監査の初回集計でもこのID名前空間の相違を検出し、元実体への対応に基づく確定集計へ訂正した。既存データの不整合ではない。末尾に`_1`のない証跡ディレクトリはその初回の中間集計で、判定根拠は本書がリンクする`_1`側である。

| 対象 | 元Mashup実体の学習混入 | 正規化本文一致の学習混入 | 661分割への混入 |
| --- | ---: | ---: | ---: |
| 固定Validation100 | 0 | 0 | 0 |
| 固定Test236 | 0 | 0 | 0 |

評価本文は元Mashup本文とも正規化一致した。さらに全{len(fm['datasets'])}種類×2用途＝{len(fixed_checks)}個の保存済み固定評価JSONについて、keys集合・100/236件・manifestの内容ハッシュを直接照合し、全一致した。マスク済み評価文で元本文照合を代替していない。

同名でも別API実体である以下4件は正しく保持されていた。名前だけで重複除外する方法は用いていない。

{nametable}
別名本文一致のvacationicは学習集合に含まれず、touropiaの本文流出はなかった。この監査は指定正規化後の完全一致を扱い、意味的類似文や未確認別名を網羅するものではない。

## 7. ラベル支持数と学習可能性

653分類器・661組に含まれる固定出力を、**分類器×Fold×ラベルの延べ3,085出力**として数えた。ゼロ支持は次のとおり。

| 状態 | 出力数 | 該当分割数 | 該当分類器数 |
| --- | ---: | ---: | ---: |
| Train正例0 | 0 | 0 | 0 |
| Train負例0 | 0 | 0 | 0 |
| Validation正例0 | 0 | 0 | 0 |
| Validation負例0 | 0 | 0 | 0 |
| 全出力が定数であるTrain / Validation | 0 | 0 | 0 |

{range_table}
Trainだけから得るneg/posは0.0086767896〜2,738.0で、全3,085出力が有限・正値。Phase 3の[1e-4,1e4]制限内に収まり、今回のカテゴリ群では0支持補正・clampは必要ない。最終全開発集合も各Trainの上位集合であるため、正例・負例の双方を保持できる。

**WARNING：0支持がなくても、十分な学習情報・安定した評価があるとは限らない。** Train正例2件の出力が16、Train負例2件が3、Validation正例1件が88（73分類器）、Validation負例1件が17（9分類器）。Validation正例5件以下は694出力。CVの2分類器はValidation自体が2〜4件なので、最良lr/epochの選択が少数サンプルに敏感となる。

学習側支持2件の全出力を以下に記載する。Validation支持1件などの全UID・Fold・ラベルはsparse_support_warnings.csvへ保存した。これらは教師誤りではなく、データの希少性による警告である。

{smalltable}
影響は、希少出力のRecall等が1例の予測で大きく変わること、micro-F1が多数例に支配され得ること、BCEの重みが大きい出力の最適化が不安定になり得ることである。修正案は「教師や分割を自動で変える」ことではなく、支持数付きの結果報告、事前に定めた条件選択・希少ラベル評価方針の明文化、必要に応じて全層CV方針を研究責任者と確定することである。ValidationからTrainへの救済移動は提案しない。

## 8. Phase 3カテゴリ学習コードとの互換性

対象は `scripts/hierarchical_api_selection/api_negative_sampling/train_category_classifier.py` と同ディレクトリのtraining.py・pipeline.py。学習関数は実行せず、コード・ASTと実データのキー整合だけを確認した。

| 確認項目 | 静的確認と根拠 |
| --- | --- |
| Phase 2 JSON読込 | training.py:180 load_classifierのカテゴリ側がPHASE_TWO/splits/UID.jsonを読む。全653 UIDの実ファイルが存在 |
| 正しい本文 | 同関数がPhase 1 unified_servicesのentity_idでlookupを作り、encode:95がdescriptionを使用。全教師ID解決と元本文一致を独立確認 |
| 分類器固有target | load_classifier:191がmembersのtarget_labelsを返す。api_labelsやcategory_labelsを直接そのまま教師化しない |
| 固定順multi-hot | run:213でobj.output_labelsを取得し、encode:99がその順にラベル包含を判定。次元は2〜51 |
| Train由来pos_weight | make_loss:103がtrain_idsのみをsafe_weightsへ渡す。Validationの支持は重みへ使わない。今回の全出力は通常のneg/posで計算可能 |
| CV/Holdoutの選択 | method文字列で再分割せず、保存partition.foldsを列挙する。CVは5 Fold、Holdoutは1 Foldとして同エンジンへ接続 |
| 学習率・epoch選択 | run:230以降が全Fold×lr2e-5/3e-5を実行する設計。choose_config:17が平均最良val_loss最小、同点でmicro-F1最大・小さいlrを選択。best_epoch中央値を四捨五入して1..50 |
| 最良重みと停止 | train_one:129で新規BERT初期化。patience3、最大50epoch、val_loss最良stateを復元する設計 |
| 最終全開発データ | load_classifier:190が全membersをfinal_training IDsとする。run:253〜257で選択lr/epoch、新規BERT、Validationなし、early_stop=False。全653件で対象全教師集合を利用できる |
| 保存モデル出力次元 | model_class:57〜60がlen(output_labels)をLinear出力数に使用。run:259以降でstate_dictと同順category_labels.jsonを保存 |
| API負例との分離 | カテゴリはratio=0の単一エントリ。APIの1/3/5比率ファイル・負例集合を読まない |

静的なデータ接続に互換性不良は検出されなかった。カテゴリ側はselection_enabled未指定時のTrueを使用するため、現存653分類器はすべて条件選択へ進める。Holdoutに対しても2学習率を比較する点と、最終epochを選択する点はPhase 2 manifestの旧参照条件と異なる（次節）。

**保証できない事項**：tokenizer/torch/transformersの実環境動作、全件テンソル化時のメモリ、GPU容量・決定論的演算、BERTキャッシュの実ロード、勾配・損失の収束、全モデルの保存再ロードと旧推論の実行時互換性。本コードにround-trip検証があっても、今回は実行していない。保存済みスモーク成功報告を全653本の実学習保証へ拡張していない。GPUのCUDAを使う場合のCUBLAS_WORKSPACE_CONFIG指定もコード側の前提として残る。

## 9. 分割方針と研究上の注意

### 9.1 現在の条件と適用理由

独立group数G<2は分割不能。G>=2かつサービス数n<20ではmin(5,G)-fold、n>=20ではgroup単位の約80/20 Holdout、seed42である。出力数はCVを禁止する条件ではない。全653カテゴリ分類器ではG<2が0個、n<20が上記2個、残る651個がn>=20なのでこの内訳になる。現在の閾値は実装された規約であり、本監査で統計的最適性を実証したものではない。

### 9.2 旧コード・Phase 2記録・Phase 3コードの違い

| 項目 | 旧学習コード／Phase 2参照条件 | Phase 3カテゴリコード |
| --- | --- | --- |
| カテゴリ段の分割 | 旧layer1〜4はファイル順80/20、観測ラベルで出力構成、重みに全studyを使う箇所あり | Phase 2の固定出力・group分割を読み、Trainだけで重み計算 |
| 少数APIの旧CV条件 | API用旧category_kfold.pyはn<20かつnum_labels>1。group非対応 | カテゴリはPhase 2のn/G判定を保存Foldから継承。APIのPhase 3規則は別 |
| 未観測Trainラベル | 旧API split_with_all_labelsにVal→Train移動あり | 移動せず保存IDのまま |
| Holdoutの学習率 | 旧コードおよびPhase 2参照は2e-5固定 | 2e-5と3e-5を比較 |
| CV後の最終epoch | Phase 2参照は全nを50epoch固定 | 選択lrの最良epoch中央値 |
| 停止と重み | 旧コードは停止時重みを保存する箇所あり | 最良val_loss重みを復元、最終全件学習はESなし |

この差はJSON読込の不具合ではなく、学習方針の差である。Phase 2のfuture_learning_referenceをそのまま現在trainerの実動仕様と解釈しないこと。対象は特定UIDだけでなく全カテゴリ（Holdout差は651 UID、最終epoch差はCV2 UIDにも及ぶ）。影響は、論文・実験計画へ記載する条件が参照manifestとずれること。対処案は、実験に採用する方針を明記してから学習を開始すること。本監査ではコードもmanifestも修正していない。

### 9.3 教授の「全部の層でCV」指示との関係

現状でできるのは、全層に存在する各分類器へのローカルな開発評価・条件選択である。しかしK-foldを実施するカテゴリ分類器は2個のみで、651個の各groupを必ず1回Validationにする機能は現在の保存分割にはない。2個のCVも異なる分類器のfold_01を共通Foldと見なせず、階層全体で同一サービスを揃えたOOF推薦性能の推定はできない。別分類器のTrain/Validationに同じサービスが現れる正常な再利用を、共通OOFで分離済みという意味に読み替えてはならない。

Phase 3のAPI分類器は50 CV・47 Holdout・55事前固定設定という別規則であり、こちらも全分類器K-foldではない（保存manifestと実装を参照した背景説明であり、今回API分割の独立監査を行ったという意味ではない）。

「全層の各分類器でK-fold」なのか「全層で開発検証を行う」なのか、教授の意図を本監査で推測しない。前者が要件であれば、現存データは要件未達であり、将来の仕様決定と新バージョンの分割が必要になる。旧共通group_5foldの存在で、Phase 2正式対象が全層CV済みになるわけではない。全層共通OOFまで要求するなら、現在の分類器独立Foldとは別の設計条件である。

## 10. 指摘事項・修正案・本格学習への判断

| 判定 | 対象 | 原因・影響 | 修正／対応案（未実施） |
| --- | --- | --- | --- |
| PASS | 全653 UID・661組 | 構造、教師、分離、支持数一致。データ破損や漏洩を検出せず | データ修正不要 |
| WARNING | classifiers.csvのmethod=holdoutである651 UID | n>=20をHoldoutにする規約。全分類器K-foldの要件には未達 | 実験方針を確定し、必要なら将来の別版で対応。現データは維持 |
| WARNING | 上記支持2件表およびsparse_support_warnings.csvのUID・Fold・label | 希少教師。評価／条件選択の分散が大きい可能性 | 支持数と不確実性を付記。0支持がないことと十分な精度を区別 |
| WARNING | Holdout651 UIDとCV2 UIDの学習設定記述 | Phase 2参照とPhase 3のlr/epoch方針差 | 採用する実験条件を事前明記。自動書換えなし |
| 注意 | 固定336 Mashup（分類器固有の欠陥なし） | 再採番IDを元IDと誤認すると漏洩の見逃し／誤検出になる | fixed_evaluation_id_map.csvの元entity対応を使用 |

技術的には、正負双方の支持、非空分割、本文解決、固定出力、最終全件学習の接続が揃っており、現方式の学習入力として利用可能である。ただし研究要件としての全層CVへの適合は別問題である。**無条件の研究本番開始は推奨せず、分割方式とPhase 3の選択方針が実験仕様として確定したうえで進むことを推奨する。** 本報告は承認要求や仕様変更の代行ではない。

実際のBERT学習、スモーク学習、モデル推論、固定評価での性能測定、分割再生成は一切実施していない。既存データ・コード・モデル・学習結果は変更していない。

## 11. 全件証跡

確定版は以下。CSVは省略せず全件を収録している。

- [summary.json](category_split_independent_audit_artifacts_1/summary.json)：検査条件数、失敗0、ゼロ支持集計、比率範囲。
- [layer_counts.csv](category_split_independent_audit_artifacts_1/layer_counts.csv)：42組の期待数・実数。
- [classifiers.csv](category_split_independent_audit_artifacts_1/classifiers.csv)：全653 UID、教師一致、件数、出力数、方式。
- [splits.csv](category_split_independent_audit_artifacts_1/splits.csv)：全661組の件数・group数・実比率・ゼロ支持。
- [label_supports.csv](category_split_independent_audit_artifacts_1/label_supports.csv)：全3,085出力のTrain/Validation正負支持とTrain由来重み。
- [sparse_support_warnings.csv](category_split_independent_audit_artifacts_1/sparse_support_warnings.csv)：低支持出力の全UID・Fold・ラベル。
- [fixed_evaluation_id_map.csv](category_split_independent_audit_artifacts_1/fixed_evaluation_id_map.csv)：固定336件の評価内IDと元Mashup entity_id。
- [fixed_variant_checks.json](category_split_independent_audit_artifacts_1/fixed_variant_checks.json)：全固定評価JSONのkeys・件数・ハッシュ照合。
- [hash_checks.csv](category_split_independent_audit_artifacts_1/hash_checks.csv)：履歴アンカー3,288件の内容照合。
- [audit_script.py](category_split_independent_audit_artifacts_1/audit_script.py)：今回新規作成した独立集計スクリプト。既存関数不使用。
- [report_script.py](category_split_independent_audit_artifacts_1/report_script.py)：追加の固定評価実ファイル照合と本報告書作成処理。

初回中間集計も上書きせず保持した。既存監査報告書が存在すれば新しい末尾番号を採用する方法で本書を新規作成した。
'''
report_path=R/'category_split_independent_audit.md';i=1
while report_path.exists():report_path=R/f'category_split_independent_audit_{i}.md';i+=1
with report_path.open('x') as f:f.write(report)
with (A/'report_script.py').open('x') as f:f.write(pathlib.Path(__file__).read_text())
print(json.dumps({'report':str(report_path),'fixed_variant_files':len(fixed_checks),'warning_rows':len(warnings),'checks':totalchecks,'report_bytes':report_path.stat().st_size},ensure_ascii=False))
