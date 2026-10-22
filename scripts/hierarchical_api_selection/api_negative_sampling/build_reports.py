"""Produce exclusive detailed reports from saved data and completed smoke evidence."""
from __future__ import annotations
import ast
import collections
import csv
import io
import json
import math
from pathlib import Path
from pipeline import ROOT, OUTPUT, PHASE_TWO, POLICY, read, write, sha, old, require


def table(headers, rows):
    def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,row))+' |' for row in rows])+'\n'


def code_references():
    lines=[]
    for name in ['pipeline.py','training.py','run_smoke_tests.py','tests/test_phase3.py']:
        p=Path(__file__).parent/name
        for node in ast.parse(p.read_text()).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                lines.append(f'- `{p.relative_to(ROOT)}:{node.lineno}` — `{node.name}`')
    return '\n'.join(lines)


def main():
    out=OUTPUT
    index=read(out/'manifests/classifier_manifest.json')['classifiers']
    summary=read(out/'reports/summary_counts.json')
    rows=list(csv.DictReader((out/'reports/negative_sample_counts.csv').open()))
    weights=list(csv.DictReader((out/'reports/train_label_weights.csv').open()))
    readiness=read(out/'checks/data_validation_results.json')
    smoke_root=ROOT/'results/hierarchical_api_selection/phase3_smoke_v1'
    smoke=read(smoke_root/'smoke_results.json');require(smoke['passed'],'Smoke not passed')
    source=read(out/'manifests/source_manifest.json')
    oldchanges=old.check_inventory(ROOT,source['original_files']);require(not oldchanges,'Protected files changed')
    tests=read(out/'checks/unit_test_results.json');require(tests['passed'],'Unit tests failed')
    ratio_table=[]
    for ratio in [1,3,5]:
        rr=[r for r in rows if int(r['ratio'])==ratio and r['fold_id']=='final_refit']
        ratio_table.append([f'1:{ratio}',sum(int(r['positive_mashups']) for r in rr),sum(int(r['positive_apis']) for r in rr),
          sum(int(r['positive_total']) for r in rr),sum(int(r['actual_negatives']) for r in rr),sum(int(r['train_total']) for r in rr),
          sum(r['shortage']=='True' for r in rr),sum(int(r['group_rounding_excess'])>0 for r in rr)])
    methods=collections.Counter((e['method'],e['n_splits']) for e in index)
    singles=[e for e in index if e['single_output']]
    corrections=[r for r in weights if r['correction'] not in ['', 'null', 'None']]
    negative_zero=[r for r in weights if int(r['negative'])==0];positive_zero=[r for r in weights if int(r['positive'])==0]
    # Each row is one (classifier,ratio,fold or final) schedule, not independent source services.
    selection_rows=[r for r in rows if r['fold_id']!='final_refit' and next(e for e in index if e['classifier_uid']==r['classifier_uid'])['selection_enabled']]
    selection_runs=2*len(selection_rows)
    api_max_steps=2*50*sum(math.ceil(int(r['train_total'])/8) for r in selection_rows)
    api_max_val_records=2*50*sum(int(r['validation_total']) for r in selection_rows)
    api_max_train_records=2*50*sum(int(r['train_total']) for r in selection_rows)
    final_upper_steps=0;final_upper_records=0
    for e in index:
        rr=[r for r in rows if r['classifier_uid']==e['classifier_uid'] and r['fold_id']=='final_refit']
        if e['selection_enabled']:
            n=max(int(r['train_total']) for r in rr);epochs=50
        else:
            n=next(int(r['train_total']) for r in rr if int(r['ratio'])==1);epochs=10
        final_upper_steps+=epochs*math.ceil(n/8);final_upper_records+=epochs*n
    category_entries=[e for e in read(PHASE_TWO/'manifests/classifier_manifest.json')['classifiers'] if e['task']=='category_classification']
    catfolds=0;cat_steps=0;cat_records=0;cat_val_records=0;cat_final_steps=0;cat_final_records=0
    cat_corrections=collections.Counter();cat_affected=set();cat_weight_rows=0
    for e in category_entries:
        o=read(PHASE_TWO/e['artifact_path']);labels=o['output_labels'];targets={r['entity_id']:r['target_labels'] for r in o['members']}
        from pipeline import safe_weights
        for f in o['partition']['folds']:
            catfolds+=1;n=len(f['train_entity_ids']);v=len(f['validation_entity_ids'])
            cat_steps+=2*50*math.ceil(n/8);cat_records+=2*50*n;cat_val_records+=2*50*v
            for w in safe_weights([targets[x] for x in f['train_entity_ids']],labels):
                cat_weight_rows+=1
                if w['correction']:cat_corrections[w['correction']]+=1;cat_affected.add(e['classifier_uid'])
        n=len(targets);cat_final_steps+=50*math.ceil(n/8);cat_final_records+=50*n
    cost={'batch_size':8,'api_selection_model_runs':selection_runs,'api_final_model_runs':152,
      'api_total_model_runs':selection_runs+152,'api_selection_max_optimizer_steps':api_max_steps,
      'api_final_max_optimizer_steps':final_upper_steps,'api_selection_max_train_service_epochs':api_max_train_records,
      'api_selection_max_validation_service_epochs':api_max_val_records,'api_final_max_train_service_epochs':final_upper_records,
      'category_selection_fold_pairs':catfolds,'category_selection_model_runs':catfolds*2,'category_final_model_runs':653,
      'category_total_model_runs':catfolds*2+653,'category_selection_max_optimizer_steps':cat_steps,
      'category_final_max_optimizer_steps':cat_final_steps,'category_selection_max_train_service_epochs':cat_records,
      'category_selection_max_validation_service_epochs':cat_val_records,'category_final_max_train_service_epochs':cat_final_records,
      'total_model_runs':selection_runs+152+catfolds*2+653,
      'total_max_optimizer_steps':api_max_steps+final_upper_steps+cat_steps+cat_final_steps,
      'assumptions':'all selectable folds, 3 API ratios x 2 lr, 50 epoch upper bound; final selectable classifier worst-case largest ratio and 50 epoch; fallback ratio1 and10epoch; no category sharing; ES may shorten runs; wall clock not extrapolated from 32-token smoke',
      'category_fold_weight_corrections':dict(cat_corrections),'category_fold_corrected_classifiers':len(cat_affected),
      'category_fold_weight_rows':cat_weight_rows}
    write(out/'reports/training_cost_estimate.json',cost)
    costmd=f'''# 本格学習の計算量見積もり

実行した学習ではなく、保存された実件数と上限epochからの算術見積もり。Batch=8、max_length=512、最大50epoch（例外は固定10epoch）。API152個は全16構成で共用し、16倍しない。カテゴリ653個の共有は適用しない。

{table(['対象','条件選択用学習回数','最終再学習回数','合計'],[['API',selection_runs,152,selection_runs+152],['カテゴリ',catfolds*2,653,catfolds*2+653],['全体',selection_runs+catfolds*2,805,cost['total_model_runs']]])}

API条件選択は3比率×2lr×{summary['selection_fold_pairs']} Fold/Holdoutペア。例外55個には条件選択を行わない。カテゴリは既存{catfolds}ペア×2lr。最終学習は各分類器1回。

{table(['対象','条件選択の最大optimizer step','最終学習の最大optimizer step','条件選択Train service-epoch','条件選択Validation service-epoch'],[['API',api_max_steps,final_upper_steps,api_max_train_records,api_max_val_records],['カテゴリ',cat_steps,cat_final_steps,cat_records,cat_val_records]])}

上限step合計は{cost['total_max_optimizer_steps']:,}。選択後のAPI比率・epochは未確定なので、最終学習は通常分類器に最大のTrain集合と50epoch、例外に比率1・10epochを使用した上限。負例比率不足とgroup単位丸めは実件数へ反映済み。ESによる短縮は見込まない。

スモークはCPU2スレッド・32token・4レコード・2epochなので、512tokenのGPU本格実験の所要時間へ換算できない。GPU種別、実batch、sequence length、検証頻度、I/O、利用可能VRAMに依存するためGPU時間は未確認。全データでの1epoch計測が次段階で必要。

学習重みはBERT約438MB/モデルで805個なら概算約353GB（ヘッド・ファイル形式で変動）。実装は条件選択の最良Fold checkpointも保存するため、条件選択モデル{selection_runs+catfolds*2}個をすべて保持するとさらに概算約{(selection_runs+catfolds*2)*.438:.0f}GBを要する。これにはoptimizer状態・データ・キャッシュを含めない。保存/整理方針は本格実験前に判断する。
'''
    write(out/'reports/training_cost_estimate.md',costmd)
    plan='''# 負例比率比較計画

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
'''
    write(out/'reports/ratio_comparison_plan.md',plan)
    smoke_rows=[]
    for c in smoke['cases']:
        r=read(smoke_root/c['result_path'])
        smoke_rows.append([c['case'],c['classifier_uid'].split('/')[-1],len(r['results']),r['selected']['learning_rate'],r['selected']['epochs'],r['final_training']['actual_epochs'],f"{c['elapsed_seconds']:.2f}",r['legacy_inference_strict_load_and_logits_equal']])
    singlenegs=[]
    for ratio in [1,3,5]:
        rr=[r for r in rows if r['fold_id']=='final_refit' and int(r['ratio'])==ratio and r['classifier_uid'] in {e['classifier_uid'] for e in singles}]
        singlenegs.append([ratio,len(rr),sum(int(r['positive_total']) for r in rr),sum(int(r['actual_negatives']) for r in rr),sum(int(r['actual_negatives'])>0 for r in rr)])
    t=table(['比率','Mashup正例延べ','API自己正例延べ','正例計','負例延べ','Train延べ計','不足分類器数','group超過分類器数'],ratio_table)
    fullrows=[]
    for e in index:
        ns=[]
        for ratio in [1,3,5]:
            rr=next(r for r in rows if r['classifier_uid']==e['classifier_uid'] and int(r['ratio'])==ratio and r['fold_id']=='final_refit')
            ns.append(rr['actual_negatives'])
        fullrows.append([e['classifier_uid'].split('/')[-1],len(e['output_labels']),e['positive_mashups'],e['positive_apis'],e['negative_candidates'],e['method_before'],e['method'],e['n_splits'],*ns,e['reason'] or '—'])
    report=f'''# Phase 3 詳細実装・実行報告

## 1. 結論と今回の範囲

既存Phase 1/2を保持し、152 API分類器の負例候補・3比率のTrain・再分割を新規作成した。学習率/epoch選択、最良重み保存復元、BERTからの最終再学習、安全なclass weight、カテゴリ分類への同エンジン適用を実装した。代表4種類の実物BERTによるCPUスモークを完了した。

**全805分類器の本格学習、全16階層の推薦性能評価、固定Validation100件/Test236件を使う学習条件選択・推論は実施していない。スモークの選択値は研究用の最適設定ではない。**

## 2. 不変条件と既存成果物の保護

カテゴリサービス21,907、Mashup3,946、API正例4,855、候補909、カテゴリ152、階層16を維持。APIカテゴリ別の正例割当は延べ6,514件。元データの本文・教師・groupを変更しない。source_manifestの宣言入力{len(source['input_sha256'])}件と新規コード/README{len(source['implementation_sha256'])}件をSHA-256保護し、生成前に存在した{len(source['original_files']):,}ファイルのsize/mtime_ns/mode/symlinkを再照合した。既存モデル内容全体はハッシュしていないため、その保証範囲を過大に述べない。

Phase 1の除外manifestから固定評価の実体ID・正規化本文との交差がないことを確認。固定336件そのものを負例や条件選択に使用しない。近似重複や未発見の別名を完全に検出する保証、過去の評価データ利用履歴を取り消す保証はない。

既存の `data_usage_redesign_proposal.md` は61 Mashup復元前の提案値21,846/4,794を記載している。今回の基準は復元後のPhase 1保存値21,907/4,855。Phase 2の全正例CV97個はカテゴリ2個を含み、APIは95個だった。以下の比較はAPI152個だけを対象とする。

## 3. 教師・負例定義とサービス数

分類器cのoutput_labels順を保持し、Mashupの候補内使用APIとcのAPI集合の交差をmulti-hot化する。交差が空なら全0負例。負例候補は3,946学習Mashupからだけ作り、909自己説明文・17,052候補外API・固定評価を負例にしない。API自己説明文は自己API正例として保持する。未使用は観測関係に基づく0であり、本質的に不要なAPIである証明ではない。

各分類器の母集合は全3,946 Mashupとそのカテゴリの自己API説明文。サービスは複数カテゴリへ再利用される。**延べ正例6,514件＋延べ負例候補593,278件＝延べ599,792件**であり、独立した599,792サービスではない。独立したAPI学習母集合は従来と同じ4,855サービス。正例と負例に同一entity_idを二重登録しない。

## 4. 分割・抽出policy

policy: `{POLICY['name']}`。seed42。API自己説明文を持つgroupはTrain固定。残るMashup groupを正例を含むgroup/負例だけのgroupに事前層別し、各層内で既存の決定的なgroup allocatorを利用する。分割後のValidation→Train移動は行わない。自己説明文に由来する特異な性能を混ぜず、Mashupに対する指標を条件選択に使うための設計。

- 正例合計20未満、検証可能な独立Mashup正例groupが2以上：min(5,正例group数)-fold。
- 正例合計20以上、同groupが2以上：正負groupそれぞれ概ね80/20 Holdout。少数層でもValidationに正例groupを最低1つ確保する容量を事前決定する。
- それ以外：正例groupをすべてTrain固定し、負例のみの診断Holdoutを保存。通常CV/条件選択は無効。共通設定比率1、lr2e-5、10epochを使う。これは事前の保守的な実装用設定で、データから最適と検証された値ではない。

分割の後、Trainの正例数（Mashup＋自己API）×比率を要求負例数とする。正例と同groupの負例は必ず同時採用。それ以外は分類器UID/seed42のSHA-256順でgroupを追加し、要求数へ到達したら止める。最後のgroupは分割しない。比率1⊆3⊆5。不足を複製で埋めず、超過・実比率・選択entity_idを保存する。Validationと正例Trainは比率間で完全共通。

API補助groupを常時Trainに置くため、全レコードを一巡する通常のK-foldではない。検証へ回せるMashupだけがFoldを一巡する。分類器間のFoldは独立で、階層OOF性能測定用ではない。

## 5. API分類器のCV/Holdout比較

{table(['方式','Phase 2','Phase 3'],[['K-fold',95,sum(e['method']=='kfold' for e in index)],['Holdout',47,sum(e['method']=='holdout' for e in index)],['分割不能/共通固定設定','10（旧分割不能）',sum(e['method']=='fixed_config' for e in index)]])}

{table(['Phase 3方式','Fold数','分類器数'],[[method,k,n] for (method,k),n in sorted(methods.items())])}

共通設定55個は正例1件10個と、正例2件以上でも独立したMashup検証正例group不足の45個。自己API正例しか持たない分類器を、負例を増やしただけで検証可能になったと数えない。例外も負例付き最終学習データを持ち、152個すべてを学習対象として保持する。

## 6. 3比率の実件数

以下は全開発母集団を使う**最終再学習用候補データ**の分類器別延べ件数。3条件は重複集合なので合計しない。Fold学習用件数は `negative_sample_counts.csv` の各fold行へすべて保存し、final_refit行と区別する。55例外は3条件を作成するが、現policyで実際に採用するのは比率1のみ。

{t}

各Foldには負例Train候補全体のID、抽出されたID、要求/実比率、不足、group丸め、Validationの正負内訳を記録。Foldの学習集合を足し合わせると同じサービスが繰り返し現れる。

## 7. 単一出力と損失補正

{table(['負例比率','単一出力分類器数','正例延べ','負例延べ','正負両方を持つ分類器'],singlenegs)}

68個すべてを単一出力Sigmoidとして保持。親カテゴリが予測されたことだけでAPIを推薦しない。閾値未満なら非推薦。既存コードに>0.5と>=0.5が混在するため、新規実装は>=0.5で固定しmanifestに記録した。

全APIのTrain支持数行（各比率・Fold・finalを含む）では正例0行{len(positive_zero)}、負例0行{len(negative_zero)}、補正行{len(corrections)}、補正対象分類器{len({r['classifier_uid'] for r in corrections})}。自己APIをTrainに保持したことにより全API出力の正例支持を維持し、負例追加で全正例分布を解消した。学習精度が向上したことを意味するものではない。

各Trainのみからpos_weight=neg/pos。正例0または負例0なら1.0へ補正、有限値を[1e-4,1e4]に制限。負例0のとき旧重み0で正例損失が消える問題を回避する。重み補正は、未観測正例の意味上の不足を解決しない。カテゴリ分類の既存Foldにも同じ安全計算を適用した場合、{len(cat_affected)}分類器・補正{sum(cat_corrections.values())}出力/Fold行（{dict(cat_corrections)}）が対象になる。これはカテゴリ教師の変更ではなく、新学習エンジンの重み規則の変更。

## 8. 学習条件選択・モデル保存

各比率でlr2e-5/3e-5を比較。最大50epoch、patience3、val_loss最小epochの重みをCPUへ保存し復元する。本格実行では最良checkpointを別名でディスクにも保存する。スモークは各caseで最初のlrの最良checkpointを保存し、他方はメモリ保存復元とハッシュ照合。最良重みでPrecision/Recall/micro-F1/val_lossを記録する。

lrは比率内のFold平均Loss最小、同点F1最大、lr小。負例比率は選択lrの平均F1最大、同点Recall/Precision最大、比率小。epochは選択比率/lrの最良epoch中央値を四捨五入し1..50。Fold平均は等重みで、サービス数を重みとするpool集計とは異なる。

最終学習はBERT事前学習初期値へ戻し、全正例と全開発母集団から選択比率で再抽出した負例で学習する。旧Validationのサービスも最終学習候補へ戻すが、全負例候補を無条件に使用するわけではない。最終学習にはESを行わない。裸のstate_dict（bert.*, classifier.*）と出力順ラベルJSONを新規保存する。

カテゴリ分類はPhase 2の653個の教師・Foldをそのまま参照し、負例抽出なし、2lr、最良epoch中央値と全教師再学習を同じエンジンで利用できる。API補助のTrain固定ルールをカテゴリデータへ遡及適用していない。カテゴリの本格学習・実物BERTによるカテゴリ経路のスモークは今回未実行。

## 9. 小規模動作検証

{table(['case','分類器','lr×Fold結果数','選択lr','選択epoch（スモーク）','実final epoch','秒','既存推論互換'],smoke_rows)}

ローカル `bert-base-uncased` の12層/hidden768を使用。CPU2スレッド、max_length32、各Train/Validation最大4件、1Fold、2epoch、比率1のみ。BERTの凍結はせず全パラメータへ勾配を通した。正例1件caseは通常CVを実行せず共通設定を通し、最終動作確認は2epochへ短縮した。固定epoch10の妥当性を検証した実験ではない。

各caseでBERT初期化、有限Loss、最終保存/再ロード、既存推論クラスへのstrict loadとlogits完全一致を確認。選択対象caseでは2lrで同一初期重みハッシュ、最良epoch記録・最良重み復元も確認。閾値以下の非推薦判定は通常のSigmoid処理を使用する。既存推論コードのmodule-level評価ループは実行せず、クラスASTのみ抽出して互換性を確認した。既存推論のモデル探索パスや固定評価データは実行しない。

スモーク結果は `results/hierarchical_api_selection/phase3_smoke_v1/` のcase別JSON/モデル。全比率の実物BERT比較・全Fold比較・50epoch完走・GPU実行・カテゴリ学習経路の実物BERT検証は未確認。3比率選択規則は合成結果の単体テストで確認した。

## 10. 自動検証と既存ファイル保護

データ準備検証：{readiness['checks_passed']:,}条件passed。実体/本文の固定評価非混入、正例教師の保持、全0負例の定義、API由来負例の不在、Train/Validation group非重複、3比率の同一Validation・包含関係、全group採用、Trainのみの重み計算、同seed再生成を確認。最終検証はREADME/本報告等の作成後、`checks/validation_results.json`へ別途保存する（ready検証を上書きしない）。

単体テスト{tests['tests_run']}件passed。比率包含、不足・group超過、正負混在groupの同時採用、API group固定、少数Fold、正例1件、正例group不足、Holdout group分離、有限weight、片側支持0でも非ゼロ勾配、欠落/重複/NaNの選択拒否、同点規則、中央値epoch、既存出力拒否を確認。ES合成テストではLoss列1.0,0.5,0.7,0.8,0.9でepoch5停止・epoch2の重みへ復元することも確認した。

既存保護はsource_manifestのSHA-256と全既存ファイルの保存メタデータを照合。全既存モデル内容のSHA-256をとった保証ではない。作業開始時の13,309件等の過去Phase 2報告の件数と、今回開始時のファイル数は新成果物追加により異なる。

## 11. 本格学習の計算量

API条件選択{selection_runs}回＋最終152回、カテゴリ条件選択{catfolds*2}回＋最終653回、合計{cost['total_model_runs']}回のモデル学習。最大step概算{cost['total_max_optimizer_steps']:,}。詳細は `training_cost_estimate.md/json`。GPU時間は未確認。スモーク秒数から512token本格実験時間を外挿しない。

## 12. 新規コードと成果物

新規コードは `scripts/hierarchical_api_selection/api_negative_sampling/` のみ。データ成果物は `data/hierarchical_api_selection/api_negative_sampling_v2/`、動作確認重み/記録は `results/hierarchical_api_selection/phase3_smoke_v1/`。既存出力が存在すると無条件上書きを拒否する。既存ファイルを移動・改名しない。各入力・コード・データ・checkpointのハッシュ/由来/条件をmanifestへ記録する。

- `manifests/`: 入力/実装/既存ファイル保護、負例抽出方針と各条件のパス・ハッシュ、学習policy、152分類器一覧。
- `splits/api_classification/`: 152分類器の教師、group、正負由来、抽出前分割。
- `negative_samples/negative_ratio_1|3|5/`: 各152個、Foldと最終再学習用ID・件数・重み。
- `reports/negative_sample_counts.csv`: 全Fold/全比率/finalの実件数。
- `reports/classifier_training_feasibility.csv`: 分割前後比較と例外理由。
- `reports/train_label_weights.csv`: 各出力の正負支持と補正。
- `reports/training_cost_estimate.*`: 回数・service-epoch・step上限。
- `checks/`: 単体テスト、ready検証、最終検証と成果物ハッシュ。

初回生成の `api_negative_sampling_v1/` は保存したまま、正式検証対象を `api_negative_sampling_v2/` とした。初回生成後にスモークの単一出力対象weather（実際は出力11）をcontent（出力1）へ修正したため、入力/コードhashを上書きせず新しいパスへ生成した。教師・負例・分割policyは変更していない。旧v1は学習に使わない。

## 13. 未解決事項と次段階

1. 共通設定55個のlr/epoch/比率は最適性未確認。固定評価を使わず、十分な開発データの他分類器から設定を移す研究設計に変更する選択肢がある。現状は事前固定設定を採用したことを明記する。
2. API補助group固定により各APIのTrain正例を保証するが、API説明文からの汎化性能そのものはこのValidationで測らない。API正例だけのカテゴリはMashup再現率を測れない。
3. 正負不均衡とpos_weightが連動する。全0負例の真の関連性は観測データから保証できない。負例数増加が最終階層推薦F1改善に直結するかは未確認。
4. 同じ開発ValidationをES・lr・比率選択に使うため選択バイアスがある。固定100/236は学習条件選択に使わず、後続の階層選定/最終評価でのみ使用する。過去に利用済みのTestの独立性はこの作業だけでは回復しない。
5. 本格前にGPUのVRAM/1epoch時間/保存容量を確認し、checkpoint保持方針を決める。モデル自動配置や既存推論の切替、Eプロット、階層選定、最終評価は今回の範囲外。
6. CPUスモークは2epochで、実験結果を推薦性能として報告できない。全3比率×2lr×全Foldで条件選択し、BERTから最終再学習する実験は次段階。

## 付録A. 全152分類器の件数と分割

負例数は各比率のfinal_refit候補。Fold別の詳細はCSV/JSONを参照。単一出力も削除しない。

{table(['分類器','出力数','Mashup正例','API正例','負例候補','旧方式','新方式','Fold数','負例1','負例3','負例5','例外理由'],fullrows)}

## 付録B. 新規実装の根拠（関数・行番号）

{code_references()}

既存参考：`scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py:124` load_categoryは空used_apisを除外するため、今回の全0負例loaderには流用しない。同:159の分割後のラベル保持移動はgroup分離を壊し得るため再利用しない。同:272 train_kfoldの2lr/50epochを維持するが、旧実装の最終epoch50固定・最良重みcheckpoint不保存を、今回の中央値epoch/最良重み復元に変更した。`scripts/inference/mashup_dis_nosp_category_api.py:17` のクラス構造・forwardと保存state_dictの互換性を確認した。論文本体の参照は不要という最新依頼に従い、論文との直接一致は主張しない。
'''
    # Count is computed rather than copied from historical paper/reference.
    negative_candidates=sum(e['negative_candidates'] for e in index);assignments=sum(e['positive_total'] for e in index)
    report=report.replace('延べ負例候補593,278件＝延べ599,792件',f'延べ負例候補{negative_candidates:,}件＝延べ{negative_candidates+assignments:,}件')
    write(out/'reports/implementation_report.md',report)
    write(out/'README.md',f'''# Phase 3: API負例と分類器別条件選択

実装/実行詳細は[implementation_report.md](reports/implementation_report.md)。本格学習・固定評価は未実行。実物BERT CPUスモーク4例を完了。

{t}

独立サービス数は4,855（Mashup3,946+自己API909）。表は分類器別の延べ件数。各条件を合計しない。CV50/Holdout47/共通固定設定55。正例1件10個・単一出力68個を保持。

[コードと再現手順](../../../scripts/hierarchical_api_selection/api_negative_sampling/README.md)、[比較計画](reports/ratio_comparison_plan.md)、[計算量](reports/training_cost_estimate.md)、[学習可能性CSV](reports/classifier_training_feasibility.csv)、[Fold件数CSV](reports/negative_sample_counts.csv)。

負例は観測上未使用のAPIであり、本質的に不要なAPIである証明ではない。固定Validation100/Test236を負例・CV・ES・条件選択に使用しない。既存Phase 1/2・コード・モデルは変更しない。既存出力へ上書きしない。
''')
    print(json.dumps({'reports_created':True,'training_cost':cost,'api_weight_corrections':len(corrections)},indent=2))
if __name__=='__main__':main()
