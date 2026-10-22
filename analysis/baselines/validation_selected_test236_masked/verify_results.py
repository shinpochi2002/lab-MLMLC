"""Read-only audit of source predictions; writes audit outputs beside this script."""
import csv, hashlib, json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from common import evaluation_split
from common.project_paths import legacy_path
sys.path.insert(0, str(ROOT/'scripts/analysis'))
from select_hierarchy_from_validation import select_hierarchy
OUT = Path(__file__).resolve().parent
H = ROOT/'results/hierarchy/hierarchy_existing_models_api_name_masked'
B = ROOT/'results/baselines/test236_masked'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f, escapechar='\\'))
def decode(v): return json.loads(v) if isinstance(v,str) else v
def unpack(e):
    return (e['description'], e['apis']) if isinstance(e,dict) else (e[1],e[2])
def metrics(rs):
    sums=[0.,0.,0.];tp=fp=fn=pr=empty=0
    for r in rs:
        t=set(decode(r['true_apis']));p=set(decode(next(r[k] for k in ['pred_apis','predicted_apis','recommended_apis'] if k in r)))
        a=len(t&p);b=len(p-t);c=len(t-p)
        vals=[a/len(p) if p else 0,a/len(t) if t else 0,2*a/(len(t)+len(p)) if t or p else 0]
        for j,k in enumerate(['precision','recall','f1']):
            assert math.isclose(vals[j],float(r[k]),abs_tol=1e-12),(k,r['idx']);sums[j]+=vals[j]
        tp+=a;fp+=b;fn+=c;pr+=len(p);empty+=not p
    return dict(n=len(rs),precision=sums[0]/len(rs),recall=sums[1]/len(rs),f1=sums[2]/len(rs),micro_precision=tp/(tp+fp),micro_recall=tp/(tp+fn),micro_f1=2*tp/(2*tp+fp+fn),mean_predicted_apis=pr/len(rs),empty_predictions=empty,tp=tp,fp=fp,fn=fn)
m=evaluation_split.load_manifest();hashes={}
for relative in m['datasets']:
    for split in ['validation','test']:
        p=evaluation_split.resolve_split_path(evaluation_split.SOURCE_ROOT/relative,split);hashes[str(p.relative_to(ROOT))]=sha(p)
train=ROOT/m['training_data']['path'];assert sha(train)==m['training_data']['sha256']
assert not set(read(train)) & (set(m['splits']['validation']['keys'])|set(m['splits']['test']['keys']))
canonical={s:read(evaluation_split.DEFAULT_SPLIT_DIR/s/m['source'].replace('.json','_api_name_masked.json')) for s in ['validation','test']}
ranking=[];models=set();log_checks={};validated_rows=0;execution_issues=[];affected=[]
for split,n in [('validation',100),('test',236)]:
    tasks=read(H/split/'run_summary.json');assert len(tasks)==16
    for task in tasks:
        assert task['status']=='ok' and task['returncode']==0
        full_log=Path(task['log_path']).read_text()
        errors=re.findall(r'ERROR:root:ファイルが見つかりません: (.*)',full_log)
        if errors: execution_issues.append(dict(split=split,pattern=task['name'],skip_count=len(errors),missing_files=sorted(set(errors))))
        rs=rows(task['out_csv_path']);data=read(task['evaluation_data_path']);assert len(rs)==len(data)==n
        for idx,((key,entry),r) in enumerate(zip(data.items(),rs)):
            text,true=unpack(entry);ct,ca=unpack(canonical[split][key])
            assert int(r['idx'])==idx and r['text']==text==ct
            assert set(decode(r['true_apis']))==set(true)==set(ca)
        calculated=metrics(rs);validated_rows+=n
        if split=='validation':ranking.append((calculated['f1'],task['name']))
        if task['name']=='layer3_12_4_3':
            log=Path(task['log_path']).read_text();assert 'Traceback' not in log
            assert [int(i) for i in re.findall(r'Processing text (\d+)/'+str(n),log)]==list(range(1,n+1))
            log_checks[split]={'processed':n,'missing_branch_skips':len(errors),'no_traceback':True}
            for r in rs:
                process=decode(r['inference_process']);l1={str(a) for batch in process['layer1'] for a,s in batch}
                assert l1==set(decode(r['pred_layer1']))
                assert {str(x['l1']) for x in process['layer2']}==l1
                l2={str(a) for x in process['layer2'] for a,s in x['results']}
                assert l2==set(decode(r['pred_layer2']))
                assert {(str(x['l1']),str(a)) for x in process['layer2'] for a,s in x['results']}=={(str(x['l1']),str(x['l2'])) for x in process['layer3']}
                l3={str(a) for x in process['layer3'] for a,s in x['results']}
                assert l3==set(decode(r['pred_layer3']))
                leaf_categories={str(x['cat']) for x in process['leaf']}
                assert leaf_categories <= l3
                if l3-leaf_categories: affected.append(dict(split=split,idx=int(r['idx']),sample=list(data)[int(r['idx'])],missing_categories=sorted(l3-leaf_categories),true_apis=decode(r['true_apis']),f1=float(r['f1'])))
                predicted=[str(a) for x in process['leaf'] for a,s in x['api_scores']]
                assert predicted==decode(r['pred_apis'])
                for stage in ['layer1','layer2','layer3','leaf']:
                    batches=process[stage] if stage=='layer1' else [x['api_scores' if stage=='leaf' else 'results'] for x in process[stage]]
                    for batch in batches:
                        assert all(math.isfinite(s) and .5<=s<=1 for a,s in batch)
                paths=[f'/home/nakagawa/lab-MLMLC/layer3_12_4_3/mashup_softfill_api_all_layer3_12_model_weights.pth']
                paths += [f'/home/nakagawa/lab-MLMLC/layer3_12_4_3/mashup_softfill_api_all_layer3_12_{x}_model_weights.pth' for x in l1]
                paths += [f'/home/nakagawa/lab-MLMLC/layer3_12_4_3/mashup_softfill_api_all_layer3_12_{x["l1"]}_{x["l2"].replace("-","_")}_model_weights.pth' for x in process['layer3']]
                paths += [f'/home/nakagawa/lab-MLMLC/study_category_inc_nomashup_model_new_kfold/bert_multilabel_{c}_model_weights.pth' for c in leaf_categories]
                for p in paths:
                    resolved=Path(legacy_path(p));assert resolved.is_file(),resolved;models.add(str(resolved.relative_to(ROOT)))
ranking.sort(key=lambda r:(-r[0],r[1]));assert ranking[0][1]=='layer3_12_4_3'
selection=select_hierarchy(H/'validation');saved=read(H/'validation/selection.json');assert selection==saved
results={};pairwise=[];hr=rows(H/'test/layer3_12_4_3_results.csv');rs_by_name={'hierarchy_layer3_12_4_3':hr}
for name in ['tfidf_ovr_logistic','sbert_ovr_logistic']:
    config=read(B/name/'config.json');assert config['threshold']==.5 and not config['tuned_threshold'] and config['valid_path'] is None
    assert Path(config['test_path']).resolve()==(evaluation_split.DEFAULT_SPLIT_DIR/'test'/m['source'].replace('.json','_api_name_masked.json')).resolve()
    assert config['split']['num_train_records']==3651 and config['split']['num_test_records']==236
    rs_by_name[name]=read(B/name/'predictions.json')
    for r in rs_by_name[name]:assert set(r['predicted_apis'])=={a for a,s in r['scores'].items() if s>=.5}
ce=read(B/'cross_encoder_all909_threshold05.json');meta=ce['metadata'];assert meta['threshold']==.5 and meta['candidate_method']=='all' and meta['num_api_candidates']==909
assert Path(meta['eval_path']).resolve()==Path(config['test_path']).resolve()
assert (Path(meta['model_dir'])/'model.safetensors').is_file()
rs_by_name['cross_encoder_threshold05']=ce['records']
for name,rs in rs_by_name.items():
    assert len(rs)==236
    for i,(r,h,e) in enumerate(zip(rs,hr,canonical['test'].values())):
        tx,tr=unpack(e);assert int(r['idx'])==i and r['text'].strip()==tx.strip()==h['text'].strip()
        assert set(decode(r['true_apis']))==set(tr)
    results[name]=metrics(rs)
    if name in ['tfidf_ovr_logistic','sbert_ovr_logistic']:
        existing=read(B/name/'metrics.json')
        for k in ['precision','recall','f1','micro_precision','micro_recall','micro_f1']:assert math.isclose(results[name][k],existing[k],abs_tol=1e-12)
    if name.startswith('cross'):
        for k in ['precision','recall','f1']:assert math.isclose(results[name][k],ce['average_metrics'][k],abs_tol=1e-12)
    if rs is not hr:
        ds=[float(h['f1'])-float(r['f1']) for h,r in zip(hr,rs)]
        pairwise.append(dict(baseline=name,hierarchy_higher_f1=sum(d>1e-12 for d in ds),baseline_higher_f1=sum(d< -1e-12 for d in ds),equal_f1=sum(abs(d)<=1e-12 for d in ds),hierarchy_minus_baseline_f1=math.fsum(ds)/236))
previous=read(OUT/'summary.json')
for r in previous['results']:
    for k,v in results[r['method']].items():assert math.isclose(v,r[k],abs_tol=1e-12),(r['method'],k)
for old,new in zip(previous['pairwise_f1'],pairwise):
    for k in new:assert old[k]==new[k] if isinstance(new[k],str) else math.isclose(old[k],new[k],abs_tol=1e-12)
report=dict(all_numerical_checks_passed=True,execution_fully_valid=False,execution_issues=execution_issues,selected_pattern_affected_records=affected,all_44_dataset_variants_hash_verified=True,validation_test_disjoint=True,training_overlap=0,validated_hierarchy_rows=validated_rows,validation_ranking=ranking,selection_matches_saved=True,selected_hierarchy_log_checks=log_checks,hierarchy_prediction_trace_matches=True,used_checkpoint_files_exist=len(models),results=results,pairwise_f1=pairwise,dataset_hashes=hashes,source_prediction_hashes={str((H/'test/layer3_12_4_3_results.csv').relative_to(ROOT)):sha(H/'test/layer3_12_4_3_results.csv')},limits=['Historical checkpoint hashes and complete launch arguments are not recorded; identical historical weights cannot be established from logs alone.','All 16 structures were evaluated on test; current comparison selects strictly by validation, but the historical order of selection is not established.','Cross-encoder training provenance is not fully recorded in its saved model directory.'])
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['dataset_hashes','validation_ranking','source_prediction_hashes','pairwise_f1']},ensure_ascii=False,indent=2))
