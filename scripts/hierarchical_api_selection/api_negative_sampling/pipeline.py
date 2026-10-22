"""Phase 3 data preparation; standard library only, exclusive new outputs."""
from __future__ import annotations
import collections
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('phase_two_readonly', ROOT/'scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py')
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)
old = p2.legacy
read, write, sha, require = old.read, old.write, old.sha, old.require
PHASE_ONE = ROOT/'data/hierarchical_api_selection/group_5fold_v1'
PHASE_TWO = ROOT/'data/hierarchical_api_selection/classifier_cv_v1'
OUTPUT = ROOT/'data/hierarchical_api_selection/api_negative_sampling_v2'
POLICY = {
 'name':'api_negative_group_anchor_positive_threshold_v1', 'seed':42,
 'ratios':[1,3,5], 'learning_rates':[2e-5,3e-5], 'maximum_epochs':50,
 'early_stopping_patience':3, 'threshold':0.5, 'threshold_comparison':'>=',
 'positive_cv_threshold_exclusive':20, 'maximum_folds':5,
 'auxiliary_api_policy':'all own-category API positives and their complete groups are always Train; Validation is Mashup-only',
 'split_policy':'allocate positive-bearing and negative-only group strata separately, with >=1 validation positive group for selectable holdout; split all nonanchored Mashup groups before negative sampling; n_positive<20 and >=2 independent positive validation groups -> min(5,positive_groups)-fold; otherwise 80/20 group holdout when >=2 positive validation groups',
 'fallback_policy':'fewer than two nonanchored Mashup-positive groups: retain all positive groups in Train; negative-only diagnostic holdout; no per-classifier metric-based config selection',
 'fallback_config':{'learning_rate':2e-5,'epochs':10,'negative_ratio':1,
   'origin':'a priori conservative engineering default, not empirically optimal; not selected using any validation/test data'},
 'group_sampling':'retain all negatives sharing positive Train groups first; then seed42 classifier-specific SHA256 group ordering, take prefix until requested count reached; last whole group may overshoot; stop at availability; nested ratios',
 'learning_rate_selection':['minimum mean best-epoch validation loss','maximum mean Mashup micro-F1','smaller learning rate'],
 'ratio_selection':['maximum mean Mashup micro-F1','maximum mean Mashup recall','maximum mean Mashup precision','smaller negative ratio'],
 'epoch_selection':'median best epochs of selected ratio/lr, round half up, clamp 1..50',
 'loss_weight':'Train-only neg/pos when both >0; otherwise 1.0; finite float32 clamp [1e-4,1e4]',
 'selection_validation':'same local validation used for early stopping and tuning; scores are development metrics, not unbiased performance estimates',
 'final_training':'reinitialize from local pretrained BERT; all positives plus selected-ratio negatives sampled from full development pool including former validation; no early stopping',
 'negative_interpretation':'unobserved in recorded used-API relation, not proof of intrinsic irrelevance',
 'fixed_evaluation_used':False,
}


def sampling_order(members, uid):
    grouped=collections.defaultdict(list)
    for r in members: grouped[r['group_id']].append(r['entity_id'])
    return [(g, sorted(grouped[g])) for g in sorted(grouped,key=lambda g:old.digest(f'42|{uid}|{g}'))]


def sample_negatives(members, positive_count, ratio, uid, positive_groups=()):
    requested=positive_count*ratio
    selected=[]; groups=[]
    ordered=sampling_order(members,uid)
    mandatory=set(positive_groups)
    for gid, ids in ordered:
        if gid in mandatory: selected.extend(ids); groups.append(gid)
    mandatory_count=len(selected)
    for gid, ids in ordered:
        if gid in mandatory: continue
        if len(selected)>=requested: break
        selected.extend(ids); groups.append(gid)
    return {'entity_ids':sorted(selected),'group_ids':groups,'requested_negatives':requested,
      'available_negatives':len(members),'actual_negatives':len(selected),'mandatory_group_negatives':mandatory_count,
      'actual_ratio':len(selected)/positive_count if positive_count else None,
      'shortage':len(members)<requested,'group_rounding_excess':max(0,len(selected)-requested)}


def safe_weights(targets, labels):
    """Exact training supports, no validation labels. No tensor/library dependency."""
    require(bool(targets),'Empty Train')
    n=len(targets); c=collections.Counter(z for t in targets for z in t)
    out=[]
    for label in labels:
        pos=c[label];neg=n-pos
        raw=neg/pos if pos and neg else None
        weight=1.0 if raw is None else min(1e4,max(1e-4,raw))
        out.append({'label':label,'positive':pos,'negative':neg,'pos_weight':weight,
          'correction':'zero_positive_use_one' if not pos else 'zero_negative_use_one' if not neg else 'clamped' if raw!=weight else None})
    return out


def build_split(members, labels, uid):
    pool={r['entity_id']:r for r in members}
    bygroup=collections.defaultdict(list)
    for r in members: bygroup[r['group_id']].append(r)
    anchored={r['group_id'] for r in members if r['source_type']=='api'}
    positives=[r for r in members if r['target_labels']]
    available={g:rs for g,rs in bygroup.items() if g not in anchored}
    positive_groups={g for g,rs in available.items() if any(r['target_labels'] for r in rs)}
    selection=len(positives)>1 and len(positive_groups)>=2
    reason=None
    if not selection:
        anchored |= positive_groups
        available={g:rs for g,rs in bygroup.items() if g not in anchored}
        method='fixed_config'
        reason='one_positive_record' if len(positives)==1 else 'fewer_than_two_nonanchored_mashup_positive_groups'
        fractions=[.8,.2]; k=1
    elif len(positives)<20:
        method='kfold';k=min(5,len(positive_groups));fractions=[1/k]*k
    else:
        method='holdout';k=1;fractions=[.8,.2]
    features={g:collections.Counter({'positive_record':sum(bool(r['target_labels']) for r in rs),
      'negative_record':sum(not r['target_labels'] for r in rs),
      **{'api:'+z:sum(z in r['target_labels'] for r in rs) for z in labels if any(z in r['target_labels'] for r in rs)}})
      for g,rs in available.items()}
    # Zero-valued feature keys must not enter rare-label indices.
    features={g:collections.Counter({z:n for z,n in f.items() if n}) for g,f in features.items()}
    require(len(features)>=2,'Cannot create even diagnostic validation groups')
    if method!='kfold':
        ng=len(features);nv=max(1,min(ng-1,int(math.floor(.2*ng+.5))));fractions=[(ng-nv)/ng,nv/ng]
    # Allocate positive-bearing and negative-only group strata independently.
    # Positive-stratum holdout capacity is at least one, determined BEFORE allocation.
    alloc={}
    for has_positive in [True,False]:
        subset={g:f for g,f in features.items() if any(r['target_labels'] for r in available[g])==has_positive}
        if not subset:continue
        if method=='kfold':
            subfractions=fractions
        elif len(subset)>=2:
            nv=max(1,min(len(subset)-1,int(math.floor(.2*len(subset)+.5))))
            subfractions=[(len(subset)-nv)/len(subset),nv/len(subset)]
        else:
            alloc.update({g:0 for g in subset});continue
        alloc.update(old.stratify(subset,subfractions,42))
    folds=[]
    for j in range(k):
        vg={g for g,idx in alloc.items() if idx==(j if method=='kfold' else 1)}
        train=sorted((r for r in members if r['group_id'] not in vg),key=lambda r:r['entity_id'])
        val=sorted((r for r in members if r['group_id'] in vg),key=lambda r:r['entity_id'])
        require(all(r['source_type']=='mashup' for r in val),'API descriptions in validation')
        positive=[r for r in train if r['target_labels']];negative=[r for r in train if not r['target_labels']]
        require(positive and val,'Empty positive Train or Validation')
        # If the soft stratifier cannot place validation positives, refuse to call it successful tuning.
        feasible=selection and any(r['target_labels'] for r in val)
        folds.append({'fold_id':f'fold_{j+1:02d}',
          'positive_train_entity_ids':[r['entity_id'] for r in positive],
          'negative_train_candidate_entity_ids':[r['entity_id'] for r in negative],
          'validation_entity_ids':[r['entity_id'] for r in val],
          'train_positive_mashups':sum(r['source_type']=='mashup' for r in positive),
          'train_positive_apis':sum(r['source_type']=='api' for r in positive),
          'validation_positive':sum(bool(r['target_labels']) for r in val),
          'validation_negative':sum(not r['target_labels'] for r in val),
          'eligible_for_selection':feasible})
    if selection and not all(f['eligible_for_selection'] for f in folds):
        raise ValueError(f'Stratification failed to retain validation positives: {uid}; stop, do not repair after splitting')
    return {'method':method,'n_splits':k if method=='kfold' else 0,
      'selection_enabled':selection,'reason':reason,'anchored_train_group_ids':sorted(anchored),
      'available_positive_validation_groups':len(positive_groups),'folds':folds}


def build_classifier(definition, positive, mashups):
    labels=definition['output_labels'];labelset=set(labels)
    members=[]
    for r in mashups:
        targets=sorted(set(r['api_labels'])&labelset)
        members.append({'entity_id':r['entity_id'],'group_id':r['group_id'],'source_type':'mashup',
          'role':'mashup_positive' if targets else 'mashup_negative','target_labels':targets})
    for r in positive:
        if r['source_type']=='api' and set(r['api_labels'])&labelset:
            members.append({'entity_id':r['entity_id'],'group_id':r['group_id'],'source_type':'api',
              'role':'api_self_positive','target_labels':sorted(set(r['api_labels'])&labelset)})
    members.sort(key=lambda r:r['entity_id'])
    expected={r['entity_id']:sorted(set(r['api_labels'])&labelset) for r in positive if set(r['api_labels'])&labelset}
    require({r['entity_id']:r['target_labels'] for r in members if r['target_labels']}==expected,'Positive records or labels changed')
    return {**definition,'policy':POLICY['name'],'members':members,
      'partition':build_split(members,labels,definition['classifier_uid'])}


def create(output=OUTPUT):
    output=Path(output).resolve();require(not output.exists(),f'Refuse existing output: {output}')
    records,positive,schema,_=p2.load_phase_one(ROOT,PHASE_ONE)
    p2.verify_source_manifest(ROOT,PHASE_TWO)
    initial=old.inventory(ROOT,output)
    mashups=[r for r in records if r['source_type']=='mashup'];require(len(mashups)==3946,'Mashup count')
    definitions=[d for d,m in p2.classifiers(records,positive,schema) if d['task']=='api_classification']
    sources={}
    for base in (PHASE_ONE,PHASE_TWO):
        for p in sorted(base.rglob('*')):
            if p.is_file():sources[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ['data_usage_audit.md','data_usage_redesign_proposal.md','phase_1_2_implementation_summary.md','classifier_cv_execution_report.md',
      'scripts/hierarchical_api_selection/pipeline.py','scripts/hierarchical_api_selection/classifier_cv/classifier_pipeline.py',
      'scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py',
      'scripts/inference/mashup_dis_nosp_category_api.py']:
        sources[name]=sha(ROOT/name)
    own={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(Path(__file__).parent.rglob('*')) if p.is_file()}
    output.mkdir(parents=True,exist_ok=False)
    write(output/'manifests/source_manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'input_sha256':sources,'implementation_sha256':own,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
      'original_files':initial,'phase_one':str(PHASE_ONE.relative_to(ROOT)), 'phase_two':str(PHASE_TWO.relative_to(ROOT))})
    write(output/'manifests/training_policy.json',POLICY)
    index=[];sampling=[];counts=[];feas=[];supportrows=[]
    for de in definitions:
        obj=build_classifier(de,positive,mashups);uid=de['classifier_uid'];labels=de['output_labels'];lookup={r['entity_id']:r for r in obj['members']}
        path='splits/'+uid+'.json';write(output/path,old.canonical(obj)+'\n')
        pos=[r for r in obj['members'] if r['target_labels']];neg=[r for r in obj['members'] if not r['target_labels']]
        ratios={}
        for ratio in POLICY['ratios']:
            fs=[]
            for f in obj['partition']['folds']:
                ps=f['positive_train_entity_ids'];nc=[lookup[e] for e in f['negative_train_candidate_entity_ids']]
                sampled=sample_negatives(nc,len(ps),ratio,uid,{lookup[e]['group_id'] for e in ps})
                train=sorted(ps+sampled['entity_ids'])
                weights=safe_weights([lookup[e]['target_labels'] for e in train],labels)
                row={'classifier_uid':uid,'ratio':ratio,'fold_id':f['fold_id'],'positive_mashups':f['train_positive_mashups'],
                  'positive_apis':f['train_positive_apis'],'positive_total':len(ps),'negative_candidates':len(nc),
                  **{k:sampled[k] for k in ['requested_negatives','actual_negatives','actual_ratio','shortage','group_rounding_excess']},
                  'train_total':len(train),'validation_total':len(f['validation_entity_ids']),
                  'validation_positive':f['validation_positive'],'validation_negative':f['validation_negative']}
                counts.append(row)
                fs.append({'fold_id':f['fold_id'],'train_entity_ids':train,'negative_sampling':sampled,
                  'validation_entity_ids':f['validation_entity_ids'],'train_label_weights':weights})
                for w in weights:supportrows.append({'classifier_uid':uid,'ratio':ratio,'fold_id':f['fold_id'],**w})
            finalsample=sample_negatives(neg,len(pos),ratio,uid,{r['group_id'] for r in pos})
            finalids=sorted([r['entity_id'] for r in pos]+finalsample['entity_ids'])
            finalweights=safe_weights([lookup[e]['target_labels'] for e in finalids],labels)
            ratioobj={'classifier_uid':uid,'ratio':ratio,'folds':fs,'final_training':{
              'train_entity_ids':finalids,'negative_sampling':finalsample,'train_label_weights':finalweights,
              'all_positive_records_included':True,'former_validation_eligible':True}}
            rp=f'negative_samples/negative_ratio_{ratio}/{uid.split("/")[-1]}.json'
            write(output/rp,old.canonical(ratioobj)+'\n');ratios[str(ratio)]={'path':rp,'sha256':sha(output/rp)}
            counts.append({'classifier_uid':uid,'ratio':ratio,'fold_id':'final_refit',
              'positive_mashups':sum(r['source_type']=='mashup' for r in pos),'positive_apis':sum(r['source_type']=='api' for r in pos),
              'positive_total':len(pos),'negative_candidates':len(neg),
              **{k:finalsample[k] for k in ['requested_negatives','actual_negatives','actual_ratio','shortage','group_rounding_excess']},
              'train_total':len(finalids),'validation_total':0,'validation_positive':0,'validation_negative':0})
            for w in finalweights:supportrows.append({'classifier_uid':uid,'ratio':ratio,'fold_id':'final_refit',**w})
        before=read(PHASE_TWO/('splits/'+uid+'.json'))['partition']
        entry={'classifier_uid':uid,'path':path,'sha256':sha(output/path),'output_labels':labels,'ratios':ratios,
          'positive_total':len(pos),'positive_mashups':sum(r['source_type']=='mashup' for r in pos),
          'positive_apis':sum(r['source_type']=='api' for r in pos),'negative_candidates':len(neg),
          'single_output':len(labels)==1,'method_before':before['method'],'folds_before':before['n_splits'],
          'method':obj['partition']['method'],'n_splits':obj['partition']['n_splits'],
          'selection_enabled':obj['partition']['selection_enabled'],'reason':obj['partition']['reason']}
        index.append(entry);feas.append({k:v for k,v in entry.items() if k not in ['path','sha256','output_labels','ratios']})
        sampling.append({'classifier_uid':uid,'ratios':ratios})
    write(output/'manifests/classifier_manifest.json',{'classifiers':index,'category_classifiers':{
      'source':str(PHASE_TWO.relative_to(ROOT)),'unchanged_count':653,'negative_sampling':False,
      'selection_policy':'reuse existing local CV/Holdout; lr loss-first, median epoch; same Train-only weight safety'}})
    write(output/'manifests/negative_sampling_manifest.json',{'policy':POLICY,'classifiers':sampling,
      'negative_source_mashups':3946,'api_positive_unique_services':4855,'positive_assignments':sum(e['positive_total'] for e in index),
      'category_services_unchanged':21907,'validation_api_description_records':0})
    old.write_csv(output/'reports/negative_sample_counts.csv',counts)
    old.write_csv(output/'reports/classifier_training_feasibility.csv',feas)
    old.write_csv(output/'reports/train_label_weights.csv',supportrows)
    summary={'method_counts':dict(collections.Counter(e['method'] for e in index)),
      'before_method_counts':dict(collections.Counter(e['method_before'] for e in index)),
      'single_output':sum(e['single_output'] for e in index),'one_positive':sum(e['positive_total']==1 for e in index),
      'selection_fold_pairs':sum(e['n_splits'] if e['method']=='kfold' else 1 if e['method']=='holdout' else 0 for e in index),
      'weight_correction_rows':sum(r['correction'] is not None for r in supportrows),
      'weight_corrected_classifiers':len({r['classifier_uid'] for r in supportrows if r['correction']}),
      'final_training_by_ratio':{str(r):{k:sum(c[k] for c in counts if c['ratio']==r and c['fold_id']=='final_refit') for k in ['positive_total','actual_negatives','train_total']} for r in POLICY['ratios']}}
    write(output/'reports/summary_counts.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def validate(output=OUTPUT, save=False):
    output=Path(output);source=read(output/'manifests/source_manifest.json');checks=0
    def check(ok,message):
        nonlocal checks
        require(ok,message);checks+=1
    for p,h in {**source['input_sha256'],**source['implementation_sha256']}.items():check(sha(ROOT/p)==h,'Protected hash changed: '+p)
    check(not old.check_inventory(ROOT,source['original_files']),'Existing files changed')
    records,positive,schema,_=p2.load_phase_one(ROOT,PHASE_ONE)
    excluded=read(PHASE_ONE/'manifests/exclusion_manifest.json');entities={r['entity_id']:r for r in records}
    idx=read(output/'manifests/classifier_manifest.json')['classifiers']
    check(len(idx)==152,'Classifier coverage')
    definitions={d['classifier_uid']:d for d,m in p2.classifiers(records,positive,schema) if d['task']=='api_classification'}
    mashups=[r for r in records if r['source_type']=='mashup']
    for e in idx:
        uid=e['classifier_uid'];obj=read(output/e['path']);lookup={r['entity_id']:r for r in obj['members']};labels=e['output_labels']
        check(sha(output/e['path'])==e['sha256'],'Classifier artifact hash')
        check(obj==build_classifier(definitions[uid],positive,mashups),'Same-seed reproduction differs')
        check(len(lookup)==len(obj['members']),'Duplicate entity')
        check(not set(lookup)&set(excluded['fixed_evaluation_entity_ids']),'Fixed evaluation entity')
        check(not {entities[x]['normalized_text_sha256'] for x in lookup}&set(excluded['fixed_evaluation_text_sha256']),'Fixed evaluation text')
        for r in obj['members']:
            expected=sorted(set(entities[r['entity_id']]['api_labels'])&set(labels))
            check(r['target_labels']==expected,'Incorrect observed API intersection')
            check(bool(expected)==(r['role']!='mashup_negative'),'Positive/negative contradiction')
            check(r['role']!='mashup_negative' or r['source_type']=='mashup','Negative not Mashup')
        group_entities=collections.defaultdict(set)
        for x,r in lookup.items():group_entities[r['group_id']].add(x)
        prev={};prevfinal=set()
        for ratio in [1,3,5]:
            rr=e['ratios'][str(ratio)];rs=read(output/rr['path']);check(sha(output/rr['path'])==rr['sha256'],'Ratio artifact hash')
            for f,rf in zip(obj['partition']['folds'],rs['folds']):
                tr=rf['train_entity_ids'];va=rf['validation_entity_ids'];ts=set(tr);vs=set(va)
                check(len(ts)==len(tr) and not ts&vs,'Duplicate or shared records')
                check(va==f['validation_entity_ids'],'Validation changed across ratios')
                check(all(lookup[x]['source_type']=='mashup' for x in va),'NonMashup validation')
                check(not {lookup[x]['group_id'] for x in tr}&{lookup[x]['group_id'] for x in va},'Group leakage')
                check(set(f['positive_train_entity_ids'])<=ts,'Positive loss')
                expected=sample_negatives([lookup[x] for x in f['negative_train_candidate_entity_ids']],len(f['positive_train_entity_ids']),ratio,uid,{lookup[x]['group_id'] for x in f['positive_train_entity_ids']})
                check(expected==rf['negative_sampling'],'Nonreproducible sampling')
                check(ts==set(f['positive_train_entity_ids'])|set(expected['entity_ids']),'Train membership inconsistent')
                negset=set(expected['entity_ids']);check(prev.get(f['fold_id'],set())<=negset,'Ratios not nested');prev[f['fold_id']]=negset
                check(rf['train_label_weights']==safe_weights([lookup[x]['target_labels'] for x in tr],labels),'Train-only weights inconsistent')
                check(all(math.isfinite(w['pos_weight']) and w['pos_weight']>0 for w in rf['train_label_weights']),'Invalid weights')
                for gid in {lookup[x]['group_id'] for x in tr}:
                    # Only records from the classifier universe; whole groups including mixed-role records.
                    check(group_entities[gid]<=ts,'Partial group sampling')
            fin=rs['final_training'];fs=set(fin['train_entity_ids']);ps={x for x,r in lookup.items() if r['target_labels']}
            check(ps<=fs,'Final refit lost positives')
            check(prevfinal<=set(fin['negative_sampling']['entity_ids']),'Final ratios not nested');prevfinal=set(fin['negative_sampling']['entity_ids'])
            expected=sample_negatives([r for r in lookup.values() if not r['target_labels']],len(ps),ratio,uid,{lookup[x]['group_id'] for x in ps})
            check(expected==fin['negative_sampling'],'Final sampling reproducibility')
            check(fs==ps|set(expected['entity_ids']),'Final Train membership')
            check(fin['train_label_weights']==safe_weights([lookup[x]['target_labels'] for x in fin['train_entity_ids']],labels),'Final weights')
        if obj['partition']['method']=='kfold':
            vals=[x for f in obj['partition']['folds'] for x in f['validation_entity_ids']]
            expected={x for x,r in lookup.items() if r['group_id'] not in set(obj['partition']['anchored_train_group_ids'])}
            check(len(set(vals))==len(vals) and set(vals)==expected,'CV coverage of nonanchored population')
    result={'passed':True,'checks_passed':checks,'fixed_evaluation_used':False,'group_overlaps':0,
      'same_validation_across_ratios':True,'same_seed_reproduced':True,
      'generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'generated_artifact_sha256':{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p!=output/'checks/validation_results.json'}}
    if save:write(output/'checks/validation_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='generated_artifact_sha256'},indent=2))
    return result
