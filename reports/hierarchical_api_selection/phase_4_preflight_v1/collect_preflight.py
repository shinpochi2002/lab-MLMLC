"""Read-only Phase 4 checks. No training, splitter, or data generation is called."""
import sys, pathlib, json, csv, collections, hashlib, math, os, traceback, ast
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/hierarchical_api_selection/api_negative_sampling'))
import training
from pipeline import PHASE_ONE,PHASE_TWO,OUTPUT,POLICY
read=lambda p:json.loads(pathlib.Path(p).read_text())
def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def csvsave(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks=[]
def ck(ok,name,where=''):
    checks.append({'check':name,'where':where,'passed':bool(ok)})
    if not ok:raise RuntimeError(name+': '+where)
training.verify_inputs(OUTPUT)
ck(True,'existing_verify_inputs_no_training')
api=read(OUTPUT/'manifests/classifier_manifest.json')['classifiers']
category=[e for e in read(PHASE_TWO/'manifests/classifier_manifest.json')['classifiers'] if e['task']=='category_classification']
ratio_rows=[]; run_rows=[];rep_configs=[]; cls_rows=[]
api_objs={};ratio_objs={};category_objs={}
for entry in api:
    uid=entry['classifier_uid'];obj=read(OUTPUT/entry['path']);api_objs[uid]=obj
    members={x['entity_id']:x for x in obj['members']};labels=obj['output_labels'];part=obj['partition'];enabled=part['selection_enabled']
    ck(obj['output_labels']==entry['output_labels'],'api_output_order',uid)
    prior_pos={};prior_neg={};reference_val={}
    for ratio in [1,3,5]:
        rs=read(OUTPUT/entry['ratios'][str(ratio)]['path']);ratio_objs[uid,ratio]=rs
        ck(rs['ratio']==ratio and rs['classifier_uid']==uid,'ratio_identity',uid)
        ck([f['fold_id'] for f in rs['folds']]==[f['fold_id'] for f in part['folds']],'ratio_fold_ids',uid)
        for f in rs['folds']+[dict(rs['final_training'],fold_id='final_refit')]:
            fid=f['fold_id'];ids=f['train_entity_ids'];ss=f['negative_sampling'];positive={e for e in ids if members[e]['target_labels']};negative=set(ids)-positive
            ck(len(set(ids))==len(ids),'unique_ids',uid+'/'+fid)
            ck(len(negative)==ss['actual_negatives'] and len(positive)*ratio==ss['requested_negatives'],'ratio_counts',uid+'/'+fid)
            ck(set(ss['entity_ids'])==negative,'negative_saved_ids',uid+'/'+fid)
            ck(ss['shortage']==(ss['available_negatives']<ss['requested_negatives']),'shortage_flag',uid+'/'+fid)
            if ratio==1:prior_pos[fid]=positive;prior_neg[fid]=negative
            else:
                ck(positive==prior_pos[fid] and prior_neg[fid]<=negative,'positive_equal_negative_nested',uid+'/'+fid);prior_neg[fid]=negative
            va=f.get('validation_entity_ids',[])
            if fid!='final_refit':
                orig=next(x for x in part['folds'] if x['fold_id']==fid)
                ck(va==orig['validation_entity_ids'],'saved_validation_same',uid+'/'+fid)
                if ratio==1:reference_val[fid]=va
                else:ck(va==reference_val[fid],'validation_equal_across_ratios',uid+'/'+fid)
                ck(not set(ids)&set(va),'no_entity_leakage',uid+'/'+fid)
                ck(not {members[e]['group_id'] for e in ids}&{members[e]['group_id'] for e in va},'no_group_leakage',uid+'/'+fid)
            else:ck(positive=={e for e,m in members.items() if m['target_labels']},'all_final_positives',uid)
            # Recompute weights with existing safe helper; no fitting occurs.
            w=training.safe_weights([members[e]['target_labels'] for e in ids],labels)
            ck(w==f['train_label_weights'],'existing_train_weight_loader',uid+'/'+fid)
            row={'classifier_uid':uid,'method':part['method'],'selection_enabled':enabled,'ratio':ratio,'fold_id':fid,'train':len(ids),'positive':len(positive),'negative':len(negative),'requested_negative':ss['requested_negatives'],'available_negative':ss['available_negatives'],'actual_ratio':len(negative)/len(positive),'shortage':ss['shortage'],'group_excess':ss['group_rounding_excess'],'validation':len(va),'outputs':len(labels)}
            ratio_rows.append(row)
            if fid!='final_refit' and enabled:
                for lr in POLICY['learning_rates']:run_rows.append(dict(row,stage='api_selection',learning_rate=lr,train_batches=math.ceil(len(ids)/8),validation_batches=math.ceil(len(va)/8)))
    cls_rows.append({'classifier_uid':uid,'task':'api','method':part['method'],'selection_enabled':enabled,'folds':len(part['folds']) if enabled else 0,'outputs':len(labels)})
for e in category:
    uid=e['classifier_uid'];obj=read(PHASE_TWO/e['artifact_path']);category_objs[uid]=obj
    for f in obj['partition']['folds']:
        for lr in POLICY['learning_rates']:
            row={'classifier_uid':uid,'method':obj['partition']['method'],'selection_enabled':True,'ratio':0,'fold_id':f['fold_id'],'train':len(f['train_entity_ids']),'positive':'','negative':'','requested_negative':'','available_negative':'','actual_ratio':'','shortage':False,'group_excess':0,'validation':len(f['validation_entity_ids']),'outputs':len(obj['output_labels']),'stage':'category_selection','learning_rate':lr,'train_batches':math.ceil(len(f['train_entity_ids'])/8),'validation_batches':math.ceil(len(f['validation_entity_ids'])/8)}
            run_rows.append(row)
    cls_rows.append({'classifier_uid':uid,'task':'category','method':obj['partition']['method'],'selection_enabled':True,'folds':len(obj['partition']['folds']),'outputs':len(obj['output_labels'])})
# The caller's required representative cases; all use saved fold_01 without resplitting.
cases=[('api_cv_multi','api_classification/category_internet_things'),('api_holdout_multi','api_classification/category_mapping'),('api_single_output','api_classification/category_content'),('api_one_positive','api_classification/category_booking'),('category_holdout_multi','category_classification/layer2_4_43/layer_01_parent_root'),('category_fivefold','category_classification/layer3_12_6_2/layer_03_parent_5_3')]
env=read(OUT/'logs/python_environment.json');bench=[]
for case,uid in cases:
    obj,rs,lookup,targets=training.load_classifier(uid,OUTPUT);task='api' if uid.startswith('api_') else 'category';ratio=1 if task=='api' else 0;f=rs[ratio]['folds'][0]
    for e in f['train_entity_ids']+f['validation_entity_ids']:ck(e in lookup and e in targets,'representative_loader_ids',uid)
    ck(all(set(v)<=set(obj['output_labels']) for v in targets.values()),'representative_target_space',uid)
    config={'case':case,'classifier_uid':uid,'task':task,'ratio':ratio,'fold_id':f['fold_id'],'method':obj['partition']['method'],'selection_enabled':obj['partition'].get('selection_enabled',True),'train_entity_ids':f['train_entity_ids'],'validation_entity_ids':f['validation_entity_ids'],'output_labels':obj['output_labels'],'learning_rates':POLICY['learning_rates'],'production_batch_size':8,'production_max_length':512,'precision':'float32; no autocast/GradScaler','planned_measurement_epochs':1,'device':'cuda:0','CUBLAS_WORKSPACE_CONFIG':':4096:8','execution':'SKIP_NO_CUDA','existing_entry':'training.train_one','no_new_training_loop':True}
    rep_configs.append(config)
    bench.append({'case':case,'classifier_id':uid,'task':task,'method':obj['partition']['method'],'selection_enabled':config['selection_enabled'],'ratio':ratio,'fold_id':f['fold_id'],'train_count':len(f['train_entity_ids']),'validation_count':len(f['validation_entity_ids']),'output_labels':len(obj['output_labels']),'gpu_model':'UNAVAILABLE','batch_size':8,'max_length':512,'precision':'float32','epoch_seconds':'','vram_peak_allocated_bytes':'','vram_peak_reserved_bytes':'','checkpoint_bytes':'','measured_epochs':0,'status':'SKIP_NO_CUDA','error':'CUDA init: No CUDA GPUs are available','forward_backward':'NOT_RUN','optimizer_update':'NOT_RUN','early_stopping':'NOT_RUN_ON_GPU','checkpoint_roundtrip':'NOT_RUN_ON_GPU','legacy_inference':'NOT_RUN_ON_GPU'})
save('configs/representatives.json',rep_configs);csvsave('gpu_benchmark.csv',bench);csvsave('negative_ratio_inventory.csv',ratio_rows);csvsave('selection_run_inventory.csv',run_rows);csvsave('classifier_inventory.csv',cls_rows)
csvsave('negative_shortages.csv',[r for r in ratio_rows if r['shortage']])
save('logs/data_connection_checks.json',{'passed':True,'checks':checks})
cost={'api_classifiers':len(api),'category_classifiers':len(category),'api_methods':dict(collections.Counter(e['method'] for e in api)),'api_cv_fold_histogram':dict(collections.Counter(e['n_splits'] for e in api if e['method']=='kfold')),'api_selection_fold_pairs':sum(c['folds'] for c in cls_rows if c['task']=='api'),'category_selection_fold_pairs':sum(c['folds'] for c in cls_rows if c['task']=='category'),'runs':dict(collections.Counter(r['stage'] for r in run_rows)),'api_final_runs':len(api),'category_final_runs':len(category),'learning_rates':POLICY['learning_rates'],'ratios':[1,3,5],'selection_epochs_unknown':True,'gpu_hours':None,'api_fixed_final_epochs':POLICY['fallback_config']['epochs'],'ratio_shortages':{str(r):{'fold_conditions':sum(x['shortage'] for x in ratio_rows if x['ratio']==r and x['fold_id']!='final_refit'),'final_conditions':sum(x['shortage'] for x in ratio_rows if x['ratio']==r and x['fold_id']=='final_refit'),'classifiers':sorted({x['classifier_uid'] for x in ratio_rows if x['ratio']==r and x['shortage']})} for r in [1,3,5]},'selection_single_epoch_batches':{stage:{key:sum(r[key] for r in run_rows if r['stage']==stage) for key in ['train_batches','validation_batches']} for stage in ['api_selection','category_selection']}}
cost['total_runs']=sum(cost['runs'].values())+cost['api_final_runs']+cost['category_final_runs']
save('training_cost_counts.json',cost)
print(json.dumps({'counts':cost,'representatives':[{k:v for k,v in b.items() if k in ['case','classifier_id','method','train_count','validation_count','output_labels','status']} for b in bench],'read_only_checks':len(checks)},ensure_ascii=False,indent=2))
