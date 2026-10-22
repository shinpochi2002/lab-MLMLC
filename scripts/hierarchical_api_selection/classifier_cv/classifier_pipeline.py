"""Classifier-local CV/Holdout artifacts. No training imports and no existing-file writes."""
from __future__ import annotations
import argparse
import collections
import csv
import importlib.util
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('phase_one_readonly_helpers',ROOT/'scripts/hierarchical_api_selection/pipeline.py')
legacy=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
require=legacy.require
read=legacy.read
canonical=legacy.canonical
sha=legacy.sha
write=legacy.write
write_csv=legacy.write_csv
DEFAULT_PHASE_ONE=Path('data/hierarchical_api_selection/group_5fold_v1')
DEFAULT_OUTPUT=Path('data/hierarchical_api_selection/classifier_cv_v1')
SETTINGS={
 'policy':'classifier_local_threshold_20_v1','cv_threshold_exclusive':20,'max_folds':5,'seed':42,
 'cv_learning_rate_candidates':[2e-5,3e-5],
 'cv_selection_order':['minimum_mean_validation_loss','maximum_mean_micro_f1_on_loss_tie'],
 'holdout_learning_rate':2e-5,'early_stopping_monitor':'val_loss','early_stopping_patience':3,
 'maximum_epochs':50,'final_cv_refit':'all classifier members, selected learning rate; legacy fixed 50 epochs without validation',
 'single_output_cv':'attempt when n<20 and at least two independent groups; report constant-target/loss failures separately',
 'validation_purpose':'local hyperparameter selection and early stopping as in existing implementation, not full-hierarchy OOF evaluation',
 'fixed_evaluation_use':'neither fixed validation100 nor test236 may fit, select settings, or stop these classifiers',
 'negative_service_generation':False,'learning_fallback_implemented':False,
 'group_priority':'strict isolation before approximate multilabel balance; no post-split sample moves',
}


def metadata(source=None):
    result={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'seed':42,'conditions':SETTINGS,'source_manifest':'manifests/source_manifest.json'}
    if source:
        result.update(git_commit=source['git_commit'],implementation_sha256=source['implementation_sha256'])
    return result


def relpath(path,root):
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def verify_source_manifest(root,output):
    source=read(output/'manifests/source_manifest.json')
    for path,value in {**source['input_sha256'],**source['implementation_sha256']}.items():
        require(sha(root/path)==value,f'Protected input/code hash changed: {path}')
    changes=legacy.check_inventory(root,read(output/'manifests/existing_file_snapshot.json')['files'])
    require(not changes,f'Existing files changed: {changes[:10]}')
    return source


def load_phase_one(root,phase_one):
    require(phase_one.is_dir(),'Phase 1 directory missing')
    old_check=read(phase_one/'checks/validation_results.json')
    require(old_check['passed'] is True,'Phase 1 validation did not pass')
    for path,value in old_check['generated_artifact_sha256'].items():
        require(sha(phase_one/path)==value,f'Phase 1 artifact changed: {path}')
    source=legacy.check_provenance(root,phase_one)
    records=read(phase_one/'source_records/category_classification.json')
    positive=read(phase_one/'source_records/api_classification_positive.json')
    unified=read(phase_one/'source_records/unified_services.json')
    schema=read(phase_one/'manifests/label_schema.json')
    groups=read(phase_one/'source_records/groups.json')
    exclusion=read(phase_one/'manifests/exclusion_manifest.json')
    require(len(records)==21907 and len(positive)==4855 and records==unified,'Phase 1 count/content mismatch')
    require(positive==[r for r in records if r['api_classification_eligible']],'API subset relationship broken')
    require(len({r['entity_id'] for r in records})==len(records),'Duplicate entity identity')
    require(len(schema['api_candidates'])==909 and len(schema['categories'])==152 and len(schema['patterns'])==16,'Fixed spaces changed')
    require(all(r['api_labels'] is None or set(r['api_labels'])<=set(schema['api_candidates']) for r in records),'API candidate violation')
    require(all(set(r['category_labels'])<=set(schema['categories']) for r in records),'Category violation')
    require(not {r['entity_id'] for r in records}&set(exclusion['fixed_evaluation_entity_ids']),'Fixed evaluation entity leaked')
    require(not {legacy.digest(legacy.normalize(r['description'])) for r in records}&set(exclusion['fixed_evaluation_text_sha256']),'Fixed evaluation text leaked')
    expected_groups=legacy.make_groups([dict(r) for r in records])
    require(expected_groups==groups,'Phase 1 group identity mismatch')
    fixed=read(root/legacy.SOURCES['fixed_split'])
    require(len(fixed['splits']['validation']['keys'])==100 and len(fixed['splits']['test']['keys'])==236,'Fixed evaluation sizes changed')
    for pattern in schema['patterns']:
        h=read(root/pattern['definition_path'])
        require(h['category_paths']==pattern['category_paths'] and set(h['category_paths'])==set(schema['categories']),f'Hierarchy differs: {pattern["pattern"]}')
    return records,positive,schema,source


def training_fingerprint(de,members,semantic_labels):
    # Potential sharing only. No shared assignment or model alias is applied.
    return legacy.digest(canonical({'task':de['task'],'output_labels':de['output_labels'],
      'members':[(r['entity_id'],r['target_labels']) for r in members],
      'semantic_label_map':semantic_labels,'training_conditions':SETTINGS}))


def classifiers(records,positive,schema):
    definitions=legacy.classifier_definitions(schema)
    require(len([d for d in definitions if d['task']=='api_classification'])==152,'API classifier coverage mismatch')
    for pattern in schema['patterns']:
        pools=collections.defaultdict(list)
        for record in records:
            for key,labels in legacy.project_classifiers(record,pattern).items():
                pools[key].append({'entity_id':record['entity_id'],'group_id':record['group_id'],
                  'target_labels':sorted(labels),'source_type':record['source_type']})
        for de in definitions:
            if de['pattern']!=pattern['pattern']:continue
            prefix=tuple(de['parent_path']);layer=de['layer']
            members=sorted(pools[(layer,prefix)],key=lambda r:r['entity_id'])
            stem=f'layer_{layer:02d}_parent_'+('_'.join(map(str,prefix)) or 'root')
            uid=f'category_classification/{de["pattern"]}/{stem}'
            semantic={cat:(cat if layer==pattern['category_layers'] else '-'.join(map(str,path[:layer])))
              for cat,path in pattern['category_paths'].items() if tuple(path[:layer-1])==prefix}
            spec={**de,'classifier_uid':uid,'hierarchies':[de['pattern']],
              'source_records':'source_records/category_classification.json',
              'sharing_applied':False,'semantic_label_map':semantic}
            yield spec,members
    pools=collections.defaultdict(list)
    for record in positive:
        targets=collections.defaultdict(set)
        for api in record['api_labels']:targets[schema['api_categories'][api]].add(api)
        for cat,labels in targets.items():pools[cat].append({'entity_id':record['entity_id'],'group_id':record['group_id'],
          'target_labels':sorted(labels),'source_type':record['source_type']})
    slugs=set()
    for de in definitions:
        if de['task']!='api_classification':continue
        cat=de['parent_path'][0];slug=re.sub('[^a-z0-9]+','_',cat.lower()).strip('_')
        require(slug and slug not in slugs,f'API folder slug collision: {cat}');slugs.add(slug)
        uid='api_classification/category_'+slug
        spec={**de,'classifier_uid':uid,'hierarchies':[p['pattern'] for p in schema['patterns']],
          'layer_by_hierarchy':{p['pattern']:p['category_layers']+1 for p in schema['patterns']},
          'source_records':'source_records/api_classification_positive.json','sharing_applied':True,
          'sharing_reason':'same records, API-category targets, fixed output schema and prescribed settings in all 16 structures',
          'semantic_label_map':{a:cat for a in de['output_labels']}}
        yield spec,sorted(pools[cat],key=lambda r:r['entity_id'])


def target_profile(members,labels):
    positive=collections.Counter(z for r in members for z in r['target_labels']);n=len(members)
    supports={z:{'positive':positive[z],'negative':n-positive[z]} for z in labels}
    zero_pos=[z for z in labels if positive[z]==0]
    zero_neg=[z for z in labels if n>0 and positive[z]==n]
    mixed=[z for z in labels if 0<positive[z]<n]
    return {'services':n,'groups':len({r['group_id'] for r in members}),
      'mashup_services':sum(r['source_type']=='mashup' for r in members),
      'api_services':sum(r['source_type']=='api' for r in members),
      'label_supports':supports,'zero_positive_labels':zero_pos,'zero_negative_labels':zero_neg,
      'variable_labels':mixed,'distinct_target_vectors':len({tuple(r['target_labels']) for r in members}),
      'positive_group_supports':{z:len({r['group_id'] for r in members if z in r['target_labels']}) for z in labels},
      'negative_group_supports':{z:len({r['group_id'] for r in members if z not in r['target_labels']}) for z in labels},
      'single_output':len(labels)==1,'single_output_has_positive_and_negative':len(labels)==1 and bool(mixed),
      'all_positive_outputs':zero_neg,'all_negative_outputs':zero_pos,
      'all_targets_constant':bool(n) and not mixed}


def local_features(members,labels):
    result=collections.defaultdict(collections.Counter)
    for r in members:
        f=result[r['group_id']]
        f.update('positive:'+z for z in r['target_labels'])
        if len(labels)==1 and labels[0] not in r['target_labels']:f['binary_negative:'+labels[0]]+=1
        f['source:'+r['source_type']]+=1
    return dict(result)


def make_partition(members,labels,seed=42):
    members=sorted(members,key=lambda r:r['entity_id'])
    full=target_profile(members,labels);g=full['groups'];n=full['services']
    if n==0 or g<2:
        return {'method':'unavailable','n_splits':0,'partition_status':'no_independent_train_validation',
          'reason':'no_records' if n==0 else 'one_independent_group','folds':[],
          'full_profile':full,'training_distribution_status':'unavailable','requires_phase_3_decision':True}
    features=local_features(members,labels)
    if n<20:
        method='kfold';k=min(5,g);alloc=legacy.stratify(features,[1/k]*k,seed)
        status='cv_5fold_partition_valid' if k==5 else 'cv_reduced_fold_partition_valid'
        val_groups=[{gid for gid,i in alloc.items() if i==j} for j in range(k)]
    else:
        method='holdout';k=1
        # A two-group population still needs nonempty Train and Validation.
        nv=max(1,min(g-1,int(math.floor(.2*g+.5))))
        alloc=legacy.stratify(features,[(g-nv)/g,nv/g],seed)
        status='holdout_partition_valid';val_groups=[{gid for gid,i in alloc.items() if i==1}]
    folds=[]
    for j,vg in enumerate(val_groups,1):
        tr=[r for r in members if r['group_id'] not in vg];va=[r for r in members if r['group_id'] in vg]
        tp=target_profile(tr,labels);vp=target_profile(va,labels)
        partition_valid=bool(tr and va) and not {r['group_id'] for r in tr}&{r['group_id'] for r in va}
        warnings=[]
        if not partition_valid:warnings.append('empty_or_overlapping_partition')
        if tp['zero_positive_labels']:warnings.append('train_zero_positive_labels')
        if tp['zero_negative_labels']:warnings.append('train_zero_negative_labels_legacy_pos_weight_zero')
        if tp['all_targets_constant']:warnings.append('train_all_targets_constant')
        if vp['all_targets_constant']:warnings.append('validation_all_targets_constant')
        if not vp['variable_labels']:warnings.append('validation_no_within_label_discrimination')
        if vp['zero_positive_labels']:warnings.append('validation_zero_positive_labels')
        if abs(len(va)/n-(.2 if method=='holdout' else 1/k))>.1:warnings.append('service_ratio_deviation_over_0_10_due_to_group_constraints')
        folds.append({'fold_id':f'fold_{j:02d}',
          'train_entity_ids':[r['entity_id'] for r in tr],'validation_entity_ids':[r['entity_id'] for r in va],
          'train_profile':tp,'validation_profile':vp,'partition_valid':partition_valid,
          'train_legacy_pos_weight_zero_labels':tp['zero_negative_labels'],
          'legacy_positive_loss_can_all_be_zero':bool(tr) and not tp['variable_labels'],
          'training_performed':False,'training_success':None,'warnings':warnings,
          'requires_phase_3_decision':bool(warnings)})
    require(all(f['partition_valid'] for f in folds),'Local partition construction failed; no repair attempted')
    constant=full['all_targets_constant']
    return {'method':method,'n_splits':k,'partition_status':status,'reason':None,'folds':folds,'full_profile':full,
      'training_distribution_status':'constant_targets_not_normal_cv_success' if constant else 'nonconstant_targets_with_fold_support_checks',
      'requires_phase_3_decision':constant or any(f['requires_phase_3_decision'] for f in folds),
      'labels_with_fewer_positive_groups_than_cv_folds':[z for z in labels if method=='kfold' and full['positive_group_supports'][z]<k],
      'exact_stratification_guaranteed':False}


def create(root,phase_one,output):
    require(not output.exists(),f'Output already exists; refusing overwrite: {output}')
    records,positive,schema,old_source=load_phase_one(root,phase_one)
    before=legacy.inventory(root,output)
    own=Path(__file__).parent
    code={relpath(p,root):sha(p) for p in sorted(own.rglob('*.py'))}
    inputs={relpath(p,root):sha(p) for p in sorted(phase_one.rglob('*')) if p.is_file()}
    inputs.update(old_source['source_files'])
    inputs[relpath(ROOT/'scripts/hierarchical_api_selection/pipeline.py',root)]=sha(ROOT/'scripts/hierarchical_api_selection/pipeline.py')
    docs=['README.md','data_usage_audit.md','data_usage_redesign_proposal.md','phase_1_2_implementation_summary.md',
      'scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py',
      'scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category.py']
    docs += [f'scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_layer{i}.py' for i in range(1,5)]
    for doc in docs:inputs[doc]=sha(root/doc)
    source={**metadata(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
      'implementation_sha256':code,'input_sha256':dict(sorted(inputs.items())),
      'phase_one_root':relpath(phase_one,root),'source_counts':{'category_services':len(records),'api_positive_services':len(positive),
        'categories':len(schema['categories']),'api_candidates':len(schema['api_candidates']),'hierarchies':len(schema['patterns'])},
      'existing_files_checked':len(before),'all_layer_fold_assignments_used':False,
      'protection_scope':'all pre-existing file metadata plus declared-source/code SHA-256; model weights not content-hashed',
      'group_allocator_dependency':'scripts/hierarchical_api_selection/pipeline.py:stratify (pure helper, invoked separately per classifier)',
      'paper_access':'Latest request specifies existing code as reference; no original-paper equivalence claim'}
    meta=metadata(source);output.mkdir(parents=True,exist_ok=False)
    write(output/'manifests/source_manifest.json',source)
    write(output/'manifests/existing_file_snapshot.json',{**meta,'files':before})
    index=[];equivalent=collections.defaultdict(list)
    for de,members in classifiers(records,positive,schema):
        labels=de['output_labels'];partition=make_partition(members,labels)
        signature=training_fingerprint(de,members,de['semantic_label_map'])
        path='splits/'+de['classifier_uid']+'.json'
        record={**de,'phase_one_root':relpath(phase_one,root),'source_record_file':relpath(phase_one/de['source_records'],root),
          'source_record_sha256':inputs[relpath(phase_one/de['source_records'],root)],
          'conditions':SETTINGS,'seed':42,'group_assignments':'classifier-local only; Phase 1 group IDs preserved, no old Fold IDs imported',
          'members':members,'partition':partition,'training_equivalence_signature':signature,
          'provisional_api_positive_only':de['task']=='api_classification',
          'split_schema_version':1,'source_manifest':'manifests/source_manifest.json'}
        # Compact one-file-per-classifier records avoid duplicate descriptions and thousands of fold files.
        write(output/path,canonical(record)+'\n')
        entry={k:v for k,v in de.items()}
        entry.update(artifact_path=path,artifact_sha256=sha(output/path),
          services=len(members),groups=partition['full_profile']['groups'],output_label_count=len(labels),
          output_labels=labels,positive_supports={z:s['positive'] for z,s in partition['full_profile']['label_supports'].items()},
          method=partition['method'],n_splits=partition['n_splits'],partition_status=partition['partition_status'],
          training_distribution_status=partition['training_distribution_status'],
          requires_phase_3_decision=partition['requires_phase_3_decision'],training_equivalence_signature=signature)
        index.append(entry)
        if de['task']=='category_classification':equivalent[signature].append(de['classifier_uid'])
    index.sort(key=lambda d:d['classifier_uid'])
    sharing=[{'signature':s,'classifier_uids':uids,'sharing_applied':False,
      'verified_conditions':'same source entities, exact target labels, output order, semantic category mapping and prescribed training settings',
      'pending_condition':'Actual selected hyperparameters and stopping rules must also agree before any future model reuse'} for s,uids in sorted(equivalent.items()) if len(uids)>1]
    write(output/'manifests/classifier_manifest.json',{**meta,'classifiers':index,'potential_category_sharing_groups':sharing,
      'api_classifier_sharing':'152 classifiers used by all 16 structures; no 16-fold multiplication of independent data',
      'category_sharing_applied':False})
    tally=dict(sorted(collections.Counter(c['method'] for c in index).items()))
    write(output/'manifests/split_manifest.json',{**meta,'classifier_count':len(index),'method_counts':tally,
      'scope':'independent partitions for each classifier; matching local fold_01 names have no cross-classifier alignment',
      'algorithm':'classifier_local_rarest_remaining_label_group_deficit_v1',
      'holdout_group_rule':'validation groups=max(1,min(G-1,floor(0.2*G+0.5))); strict group isolation; service ratio may deviate',
      'kfold_rule':'n<20 and G>=2 -> min(5,G); output label cardinality does not disable attempts',
      'single_output_stratification':'positive and negative binary supports when available, otherwise partition only with constant-target warnings',
      'post_partition_moves':0,'old_global_fold_mapping_used':False,
      'schema_file':relpath(phase_one/'manifests/label_schema.json',root),
      'future_learning_reference':SETTINGS})
    print(canonical({'created':str(output),'classifiers':len(index),'methods':tally,'potential_category_sharing_groups':len(sharing)}),flush=True)


def row_summary(record):
    de=record;part=de['partition'];p=part['full_profile'];folds=part['folds']
    return {'classifier_uid':de['classifier_uid'],'task':de['task'],'hierarchies':de['hierarchies'],'layer':de['layer'],
      'parent_path':de['parent_path'],'services':p['services'],'groups':p['groups'],'output_label_count':len(de['output_labels']),
      'method':part['method'],'n_splits':part['n_splits'],'partition_status':part['partition_status'],
      'training_distribution_status':part['training_distribution_status'],
      'independent_partition_unavailable_reason':part['reason'] or '',
      'single_output':int(p['single_output']),'single_output_has_both_classes':int(p['single_output_has_positive_and_negative']),
      'constant_target_vectors':int(p['all_targets_constant']),
      'full_zero_positive_labels':len(p['zero_positive_labels']),'full_zero_negative_labels':len(p['zero_negative_labels']),
      'train_zero_positive_label_max':max((len(f['train_profile']['zero_positive_labels']) for f in folds),default=0),
      'train_zero_negative_label_max':max((len(f['train_profile']['zero_negative_labels']) for f in folds),default=0),
      'validation_zero_positive_label_max':max((len(f['validation_profile']['zero_positive_labels']) for f in folds),default=0),
      'folds_with_all_zero_legacy_positive_loss':sum(f['legacy_positive_loss_can_all_be_zero'] for f in folds),
      'folds_needing_phase_3_decision':sum(f['requires_phase_3_decision'] for f in folds),
      'requires_phase_3_decision':int(part['requires_phase_3_decision']),
      'provisional_api_positive_only':int(de['provisional_api_positive_only'])}


def summarize(root,output):
    require(not (output/'reports').exists() and not (output/'README.md').exists(),'Summary outputs already exist; stop')
    source=verify_source_manifest(root,output);manifest=read(output/'manifests/classifier_manifest.json')
    summaries=[];folds=[];supports=[]
    for entry in manifest['classifiers']:
        record=read(output/entry['artifact_path']);summaries.append(row_summary(record));part=record['partition']
        for role,profile,fid in [('full_development',part['full_profile'],'all')]+[(role,f[role+'_profile'],f['fold_id']) for f in part['folds'] for role in ['train','validation']]:
            if role!='full_development':
                folds.append({'classifier_uid':record['classifier_uid'],'task':record['task'],'method':part['method'],'fold_id':fid,'role':role,
                  'services':profile['services'],'groups':profile['groups'],'mashup_services':profile['mashup_services'],'api_services':profile['api_services'],
                  'zero_positive_labels':len(profile['zero_positive_labels']),'zero_negative_labels':len(profile['zero_negative_labels']),
                  'all_targets_constant':int(profile['all_targets_constant'])})
            for label,sp in profile['label_supports'].items():supports.append({'classifier_uid':record['classifier_uid'],
              'task':record['task'],'fold_id':fid,'role':role,'label':label,'positive_services':sp['positive'],'negative_services':sp['negative']})
    summaries.sort(key=lambda r:r['classifier_uid'])
    folds.sort(key=lambda r:(r['classifier_uid'],r['fold_id'],r['role']))
    supports.sort(key=lambda r:(r['classifier_uid'],r['fold_id'],r['role'],r['label']))
    reports=output/'reports';reports.mkdir(exist_ok=False)
    for name,rows in [('classifier_split_summary',summaries),('category_classifier_counts',[r for r in summaries if r['task']=='category_classification']),
      ('api_classifier_counts',[r for r in summaries if r['task']=='api_classification']),
      ('rare_classifier_summary',[r for r in summaries if r['services']<20 or r['requires_phase_3_decision']]),
      ('unavailable_classifiers',[r for r in summaries if r['method']=='unavailable']),
      ('single_output_classifiers',[r for r in summaries if r['single_output']]),('fold_sample_counts',folds),('label_supports',supports)]:
        write_csv(reports/(name+'.csv'),rows,list(summaries[0]) if name in ['unavailable_classifiers','single_output_classifiers','rare_classifier_summary'] else None)
    totals={'classifier_count':len(summaries),'category_classifier_count':sum(r['task']=='category_classification' for r in summaries),
      'api_classifier_count':sum(r['task']=='api_classification' for r in summaries),'kfold_partition_count':sum(r['method']=='kfold' for r in summaries),
      'fivefold_partition_count':sum(r['method']=='kfold' and r['n_splits']==5 for r in summaries),
      'reduced_fold_partition_count':sum(r['method']=='kfold' and r['n_splits']<5 for r in summaries),
      'holdout_partition_count':sum(r['method']=='holdout' for r in summaries),'independent_partition_unavailable_count':sum(r['method']=='unavailable' for r in summaries),
      'single_output_classifier_count':sum(r['single_output'] for r in summaries),
      'single_output_cv_partition_count':sum(r['single_output'] and r['method']=='kfold' for r in summaries),
      'single_output_holdout_partition_count':sum(r['single_output'] and r['method']=='holdout' for r in summaries),
      'single_output_unavailable_count':sum(r['single_output'] and r['method']=='unavailable' for r in summaries),
      'constant_target_classifier_count':sum(r['constant_target_vectors'] for r in summaries),
      'kfold_needing_training_policy_decision':sum(r['method']=='kfold' and r['requires_phase_3_decision'] for r in summaries),
      'classifier_with_unseen_train_labels_count':sum(r['train_zero_positive_label_max']>0 for r in summaries),
      'classifier_with_train_negative_zero_labels_count':sum(r['train_zero_negative_label_max']>0 for r in summaries),
      'training_or_inference_runs':0}
    write(reports/'summary_counts.json',{**metadata(source),'counts':totals})
    write(reports/'existing_implementation_comparison.md',comparison_report())
    write(reports/'implementation_report.md',implementation_report(totals,manifest))
    write(output/'README.md',dataset_readme(totals,source['phase_one_root']))
    print(canonical({'summary':totals}),flush=True)


def comparison_report():
    return '''# 既存実装と分類器単位の分割の比較

根拠は現在の既存ソースコード。論文本体と直接照合したという主張はしない。最新依頼の明示仕様に従った。

| 項目 | 既存コード | 今回の分割 |
| --- | --- | --- |
| APIの少数CV条件 | 有効n<20かつ出力ラベル数>1 | n<20でG>=2なら単一出力でも試行 |
| n>=20 | 80/20 Holdout | 同じ適用閾値、group単位で約80/20 |
| n=1 | 複数ラベルならk=2へ進んで失敗し得る | 独立group1として分割不能を明示 |
| Fold数 | max(2,min(5,n)) | min(5,独立group数)、G<2は分割しない |
| 層化 | MultilabelStratifiedKFold、group非対応 | 分類器固有の正解分布からgroup対応の希少ラベル優先割当。完全な同一割当ではない |
| 単一出力 | 少数でもHoldoutへ送る | 正例/負例のbinary支持数を考慮。全正例は分割成立と教師分布不成立を区別 |
| 未観測Trainラベル | ValidationからTrainへ移す補正あり | 補正移動せず、該当ラベルと分類器を記録 |
| カテゴリ各段 | ファイル内の順序で80/20 | 全653分類器にも同じn/G条件を適用 |
| seed | 分割seed42 | 各分類器に独立して42を適用。旧全層Foldは読み込まない |
| CV学習率 | 2e-5,3e-5 | 次の学習用設定として記録。今回は選択・学習しない |
| 選択規則 | 平均val_loss最小、同値なら平均micro-F1最大 | 同じ規則をmanifestへ記録 |
| Holdout学習率 | 2e-5固定 | 同じ値を記録 |
| Early Stopping | 分類器のValidation loss、patience3、最大50epoch | 同じ利用目的を記録。旧共通Foldの内部ES割当は使用しない |
| CV後 | 選択lrで全nを50epoch、Validationなしで再学習 | 次の学習の参照条件として記録。今回は再学習しない |
| 出力空間 | 当該データに観測されたAPIから構築 | 固定909候補のカテゴリ別部分集合・既存階層の出力一覧を固定 |
| 負例 | 空used_apisは現行autoローダーで除外 | Phase 1の正例のみ。新規負例は作らない |

根拠ファイル：`scripts/create_study_data_and_model/mashup_clu_mlc_nosp_nostop_category_kfold.py` の `load_category` 124–155行、`split_with_all_labels` 159–209行、`train_holdout` 222–270行、`train_kfold_then_final` 272–386行、`train_one_category_auto` 399–430行。単一ラベル条件は416行。
カテゴリ段は同ディレクトリの `mashup_clu_mlc_nosp_nostop_layer{1,2,3,4}.py`。例：layer3の127–146行で教師化・全studyからの重み計算・80/20スライス、163–176行で停止と保存。

既存trainerは最良checkpointの復元を行わず停止時の重みを保存する箇所があり、CV後50epoch固定が最適とは確認していない。損失重み、epoch数、最良重みの扱いはPhase 3の判断事項。

今回はモデル全体のOOF推薦性能を計測する分割ではない。同一サービスが別分類器のTrainとValidationへ現れることは許容される。分類器内では実体・本文groupを厳密に分離する。固定Validation100/Test236はいずれも分類器の条件選択・停止・教師に使わない。
'''


def implementation_report(counts,manifest):
    lines=['# 分類器単位CV/Holdout：実装結果','',
      '既存Phase 1と全層共通Group 5-Foldを保持したまま、新しい独立分割を作成した。モデル学習・推論・負例生成は行っていない。','',
      '## 分類器数','', '| 項目 | 件数 |','| --- | ---: |']
    for key,value in counts.items():lines.append(f'| {key} | {value} |')
    lines+=['','上記CV件数は**分割が作れた分類器数**であり、通常の教師学習・評価に成功した分類器数ではない。単一出力の全正例、Train未観測ラベル、負例0、既存pos_weight=0は別に記録する。',
      'API分類器は152個を全16構成で共用するため16倍しない。カテゴリ653個は構成別IDを維持した。',
      '', '## データと共有候補','',
      'カテゴリ21,907サービス・API正例4,855サービスは既存ファイル参照で保持し、本文を再出力・変更していない。分類器教師は固定経路の親prefixと出力一覧から再現した。',
      f"カテゴリ共有候補の等価群は{len(manifest['potential_category_sharing_groups'])}組。元ID・教師・出力順・意味対応・規定学習条件を比較したが、今回カテゴリ分類器の統合は適用していない。選択後のlr・epoch等も一致した場合にのみ将来のモデル共有を検討する。",
      '', '## 保存形式','',
      '1分類器1JSONに教師のentity/group/target、全FoldのTrain/Validation IDとラベル正負支持数をまとめた。本文はPhase 1を参照する。Foldごとに別ファイルを大量に作る方式を避け、805個の分割ファイルと共通manifestで管理する。',
      '', '## 未解決事項・Phase 3','',
      '- 正例のみのAPI暫定版。負例追加後は件数・CV閾値・ラベル分布が変わるため新しい出力先で再分割する。',
      '- 1groupでは独立分割不能。固定予測・代替モデル・例の追加は実装していない。',
      '- 単一出力全正例では全0負例を捏造せず、pos_weight=0や識別情報なしを明示する。',
      '- Train正例0・負例0の出力の損失・予測規則と、最良epoch/最終再学習条件を決定する必要がある。',
      '- 既存実装同様のValidationによるESと条件選択は独立した汎化推定ではない。固定評価での最終推薦性能とは区別する。',
      '- 固定評価カテゴリGTの既知の2件の不整合はPhase 1の記録を引き継ぎ、固定ファイルを変更しない。',
      '', '## 検証','',
      'checks/validation_results.jsonに元ファイル保護、教師再現、分類器内group分離、CVのValidation一巡、seed再現、全件数・例外検出と成果物ハッシュを保存する。',
      'モデル本体の内容ハッシュは取得せず、既存全ファイルのメタデータと宣言入力・コードのSHA-256を照合する。']
    return '\n'.join(lines)+'\n'


def dataset_readme(counts,phase_one):
    return f'''# Classifier-local CV/Holdout v1

Phase 2の分類器単位への変更。カテゴリ{counts['category_classifier_count']}、API{counts['api_classifier_count']}、合計{counts['classifier_count']}分類器。
全層で共通のFold割当は使用しない。Phase 1の本文group IDだけを保持し、分類器ごとの教師から独立して分割する。

## 固定条件と入力

Phase 1: `{phase_one}`。カテゴリ21,907サービス、API正例4,855サービス、61 Mashupの復元と評価除外をそのまま保持。候補909 API、152カテゴリ、既存16階層は変更しない。
固定Validation100件・Test236件は教師、CV/Holdout、学習条件選択・Early Stoppingに含めない。既存Group 5-Foldのファイルも保持し、その割当は新しい分割に利用しない。

## 分割条件

- サービスn<20かつ独立group G>=2：k=min(5,G)のCV。出力1種類でも試行する。
- n>=20かつG>=2：groupで約80/20 Holdout。Validation group数はmax(1,min(G-1,floor(.2G+.5)))。サービス比率のずれも記録。
- G<2またはn=0：独立分割不能。Foldを捏造せず理由を保存。
- seed42を分類器ごとに独立して適用。希少ラベルからのgroup層化近似で、単一出力は両クラスがあればbinary支持数も考慮。完全なMultilabelStratifiedKFold互換ではない。
- 同じ分類器内でgroupを分けない。層化よりgroup制約を優先し、分割後にValidationからTrainへ移動しない。
- 全正例/全負例、定数教師、Train正例0・負例0、pos_weight=0の可能性をFold別に記録。分割成立を正常なCV学習成功とはみなさない。

API分類分割は**正例のみの暫定版**。Phase 3で既知Mashup負例を追加する場合は、別出力先に新しい教師・分割を作る。今回は負例データも学習fallbackも生成しない。

## ファイルとスキーマ

- manifests/source_manifest.json：Phase 1全成果物・元入力・参照コードの相対パスとSHA-256、Git commit、新実装hash、条件・UTC作成日時。
- manifests/existing_file_snapshot.json：全既存ファイルのsize/mtime/mode/link。モデル内容自体のhashは取得しない。
- manifests/classifier_manifest.json：全分類器ID、層、親、所属構成、固定出力、件数・支持数、保存先、分割方式、共有候補の等価性。
- manifests/split_manifest.json：閾値とgroup分割規則、条件選択/ES/全データ再学習の既存参照条件。学習そのものは行わない。
- splits/category_classification/<hierarchy>/layer_<nn>_parent_<path>.json、splits/api_classification/category_<name>.json：1分類器1ファイル。
- 各ファイルのmembers：entity_id、group_id、target_labels、source_type。descriptionはsource_record_fileのPhase 1データをentity_idで参照し、カテゴリ用・API用教師はここに保存した固定target_labelsを使う。
- partition：method、n_splits、partition_status、full_profile、training_distribution_status、例外理由、folds。
- folds：ローカルfold_id、train_entity_ids、validation_entity_ids、各roleのprofile、警告。fold_idはそのclassifier_uidの中だけで有効。他分類器のfold_01と結合しない。
- profile：services、groups、由来件数、各出力のpositive/negative支持数、正例0/負例0/可変ラベル、定数教師・単一出力の状態。ラベルごとのnegative支持数はmulti-hotの0の数で、追加の負例サービス生成ではない。
- reports/classifier_split_summary.csv、category_classifier_counts.csv、api_classifier_counts.csv：方式・分類器件数。
- reports/fold_sample_counts.csv、label_supports.csv：Fold/role別の実数と全ラベル支持数。
- reports/rare_classifier_summary.csv、single_output_classifiers.csv、unavailable_classifiers.csv：例外・単一出力・分割不能の記録。
- reports/existing_implementation_comparison.md、implementation_report.md、summary_counts.json：既存との差分・集計・Phase 3の判断事項。
- checks/validation_results.json：自動検証結果と生成物ハッシュ。

複数構成のAPI分類器はデータ・教師・条件が一致する152個を参照。カテゴリ分類器は等価群を記録するだけで共有を適用していない。元データ・ラベルを変更せず、全構成・分類器を追跡できる形で保存した。

## 作成・検証コマンド

リポジトリ直下、Python>=3.10、標準ライブラリのみ。元の学習モジュールはimportしない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/create_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/summarize_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1 --write-report
```

既存出力には停止する。再生成は別の未使用--output-rootを指定し、既存を削除・上書きしない。--phase-one-rootで同一スキーマの将来の入力版を指定できるが、今回の固定入力件数・除外検証の条件を満たす必要がある。負例入りのPhase 3入力は正例専用ローダー・検証を別途拡張する必要があり、そのまま通す仕様ではない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/classifier_cv/validate_classifier_splits.py --output-root data/hierarchical_api_selection/classifier_cv_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/classifier_cv/tests -v
```

通常のvalidateは読み取り専用。作成日時は実行時刻となるが、教師と割当は同じ入力・seedで再現される。分類器単位のValidationをESと条件選択へ使う既存仕様を記録しているが、モデル全体の独立OOF性能の分割ではない。CV後の全データ再学習、損失・少数例外・最良epochの学習実装は次のPhase 3。
'''


def independent_profile(members,labels):
    """Validation path does not reuse the profile generation helper."""
    n=len(members);pos={z:0 for z in labels};pg={z:set() for z in labels};ng={z:set() for z in labels}
    for row in members:
        target=set(row['target_labels'])
        for z in labels:
            pos[z]+=int(z in target)
            (pg if z in target else ng)[z].add(row['group_id'])
    return {'services':n,'groups':len({r['group_id'] for r in members}),
      'mashup_services':sum(r['source_type']=='mashup' for r in members),
      'api_services':sum(r['source_type']=='api' for r in members),
      'label_supports':{z:{'positive':pos[z],'negative':n-pos[z]} for z in labels},
      'zero_positive_labels':[z for z in labels if pos[z]==0],
      'zero_negative_labels':[z for z in labels if n and pos[z]==n],
      'variable_labels':[z for z in labels if 0<pos[z]<n],
      'distinct_target_vectors':len({tuple(r['target_labels']) for r in members}),
      'positive_group_supports':{z:len(pg[z]) for z in labels},'negative_group_supports':{z:len(ng[z]) for z in labels},
      'single_output':len(labels)==1,'single_output_has_positive_and_negative':len(labels)==1 and 0<pos[labels[0]]<n,
      'all_positive_outputs':[z for z in labels if n and pos[z]==n],
      'all_negative_outputs':[z for z in labels if pos[z]==0],
      'all_targets_constant':bool(n) and not any(0<pos[z]<n for z in labels)}


def validate(root,phase_one,output,write_report=False):
    if write_report:require(not (output/'checks').exists(),'Validation outputs already exist; stop')
    source=verify_source_manifest(root,output);records,positive,schema,_=load_phase_one(root,phase_one)
    require(source['phase_one_root']==relpath(phase_one,root),'Phase 1 input root mismatch')
    tests=[]
    def check(name,ok,detail=None):
        require(ok,f'Validation failed: {name}: {detail}')
        tests.append({'check':name,'passed':True,'detail':detail})
    check('phase_one_counts_unchanged',len(records)==21907 and len(positive)==4855)
    check('phase_one_artifacts_sources_and_code_sha256_unchanged',True,len(source['input_sha256']))
    check('preexisting_file_metadata_unchanged',True,source['existing_files_checked'])
    check('fixed_spaces_and_fixed_evaluation_exclusion',len(schema['api_candidates'])==909 and len(schema['categories'])==152 and len(schema['patterns'])==16)
    cm=read(output/'manifests/classifier_manifest.json');entries=cm['classifiers'];index={c['classifier_uid']:c for c in entries}
    check('classifier_ids_unique',len(index)==len(entries))
    check('fixed_classifier_coverage',sum(e['task']=='category_classification' for e in entries)==653 and sum(e['task']=='api_classification' for e in entries)==152)
    summaries=[];expected_fold_rows=[];expected_support_rows=[];shares=collections.defaultdict(list)
    expected_uids=[];results={'group_overlaps':0,'fixed_evaluation_overlaps':0,'same_seed_reproducible':True,'global_fold_assignment_used':False}
    for definition,members in classifiers(records,positive,schema):
        uid=definition['classifier_uid'];expected_uids.append(uid);entry=index[uid];record=read(output/entry['artifact_path'])
        labels=definition['output_labels'];lookup={r['entity_id']:r for r in members};fullg={r['group_id'] for r in members}
        check(uid+'/artifact_hash',sha(output/entry['artifact_path'])==entry['artifact_sha256'])
        check(uid+'/definition_and_labels',all(record[k]==v for k,v in definition.items()))
        check(uid+'/phase_one_pointer_and_hash',record['source_record_file']==relpath(phase_one/definition['source_records'],root) and
          record['source_record_sha256']==source['input_sha256'][record['source_record_file']])
        check(uid+'/exact_source_members_targets',record['members']==members and len(lookup)==len(members))
        part=record['partition'];profile=independent_profile(members,labels)
        check(uid+'/full_label_supports',part['full_profile']==profile)
        # Complete deterministic rerun verifies IDs and all warning/status fields, not just counts.
        check(uid+'/seed42_reproduction',part==make_partition(members,labels,42))
        check(uid+'/input_order_independent',part==make_partition(list(reversed(members)),labels,42))
        g=profile['groups'];n=len(members)
        expected_method='unavailable' if g<2 or n==0 else 'kfold' if n<20 else 'holdout'
        expected_k=0 if expected_method=='unavailable' else min(5,g) if expected_method=='kfold' else 1
        check(uid+'/threshold_group_and_single_output_policy',part['method']==expected_method and part['n_splits']==expected_k and len(part['folds'])==expected_k)
        seen=collections.Counter()
        for fold in part['folds']:
            tr=fold['train_entity_ids'];va=fold['validation_entity_ids'];ts=set(tr);vs=set(va)
            check(uid+'/'+fold['fold_id']+'/entity_partition',bool(tr and va) and not ts&vs and ts|vs==set(lookup) and len(ts)==len(tr) and len(vs)==len(va))
            tg={lookup[e]['group_id'] for e in tr};vg={lookup[e]['group_id'] for e in va}
            check(uid+'/'+fold['fold_id']+'/group_isolation',not tg&vg and tg|vg==fullg)
            seen.update(vg)
            for role,ids in [('train',tr),('validation',va)]:
                role_profile=independent_profile([lookup[e] for e in ids],labels)
                check(uid+'/'+fold['fold_id']+'/'+role+'_supports',fold[role+'_profile']==role_profile)
                expected_fold_rows.append({'classifier_uid':uid,'task':definition['task'],'method':part['method'],'fold_id':fold['fold_id'],'role':role,
                  'services':role_profile['services'],'groups':role_profile['groups'],'mashup_services':role_profile['mashup_services'],'api_services':role_profile['api_services'],
                  'zero_positive_labels':len(role_profile['zero_positive_labels']),'zero_negative_labels':len(role_profile['zero_negative_labels']),
                  'all_targets_constant':int(role_profile['all_targets_constant'])})
            check(uid+'/'+fold['fold_id']+'/no_training_or_evaluation_run',fold['training_performed'] is False and fold['training_success'] is None)
        if expected_method=='kfold':
            check(uid+'/validation_groups_one_full_pass',set(seen)==fullg and all(v==1 for v in seen.values()) and 2<=expected_k<=min(5,g))
        if profile['single_output'] and profile['all_targets_constant']:
            check(uid+'/single_output_degenerate_not_claimed_success',part['requires_phase_3_decision'] and
              (part['method']=='unavailable' or part['training_distribution_status']=='constant_targets_not_normal_cv_success'))
        signature=training_fingerprint(definition,members,definition['semantic_label_map'])
        check(uid+'/sharing_evidence',signature==record['training_equivalence_signature']==entry['training_equivalence_signature'])
        if definition['task']=='category_classification':shares[signature].append(uid)
        for fid,role,pr in [('all','full_development',profile)]+[(f['fold_id'],role,f[role+'_profile']) for f in part['folds'] for role in ['train','validation']]:
            for label,support in pr['label_supports'].items():expected_support_rows.append({'classifier_uid':uid,'task':definition['task'],
              'fold_id':fid,'role':role,'label':label,'positive_services':support['positive'],'negative_services':support['negative']})
        summaries.append(row_summary(record))
    check('all_expected_classifier_ids_present',set(index)==set(expected_uids))
    check('category_sharing_candidate_signatures',[(r['signature'],r['classifier_uids']) for r in cm['potential_category_sharing_groups']]==
      [(k,v) for k,v in sorted(shares.items()) if len(v)>1] and cm['category_sharing_applied'] is False)
    summaries.sort(key=lambda r:r['classifier_uid'])
    expected_fold_rows.sort(key=lambda r:(r['classifier_uid'],r['fold_id'],r['role']))
    expected_support_rows.sort(key=lambda r:(r['classifier_uid'],r['fold_id'],r['role'],r['label']))
    def csv_expected(rows):
        return [{k:canonical(v) if isinstance(v,(list,dict)) else str(v) for k,v in row.items()} for row in rows]
    reportsets={
      'classifier_split_summary':summaries,
      'category_classifier_counts':[r for r in summaries if r['task']=='category_classification'],
      'api_classifier_counts':[r for r in summaries if r['task']=='api_classification'],
      'rare_classifier_summary':[r for r in summaries if r['services']<20 or r['requires_phase_3_decision']],
      'unavailable_classifiers':[r for r in summaries if r['method']=='unavailable'],
      'single_output_classifiers':[r for r in summaries if r['single_output']],
      'fold_sample_counts':expected_fold_rows,'label_supports':expected_support_rows}
    # Manifest and generation iteration have the same canonical category/pattern order.
    for name,expected in reportsets.items():
        actual=legacy.load_csv(output/'reports'/f'{name}.csv')
        check(name+'/all_csv_rows_and_supports',actual==csv_expected(expected))
    counts=read(output/'reports/summary_counts.json')['counts']
    check('reported_method_totals',counts['classifier_count']==len(entries) and counts['kfold_partition_count']==sum(e['method']=='kfold' for e in entries) and
      counts['holdout_partition_count']==sum(e['method']=='holdout' for e in entries) and counts['independent_partition_unavailable_count']==sum(e['method']=='unavailable' for e in entries))
    sm=read(output/'manifests/split_manifest.json')
    check('independent_assignment_scope',sm['old_global_fold_mapping_used'] is False and sm['post_partition_moves']==0 and sm['future_learning_reference']==SETTINGS)
    previous=output/'checks/validation_results.json'
    if previous.exists():
        old=read(previous)
        check('saved_generated_artifact_hashes',all(sha(output/p)==h for p,h in old['generated_artifact_sha256'].items()))
    result={**metadata(source),'passed':True,'classifier_counts':counts,'checks':tests,'leakage_results':results,
      'scope':'partition and source integrity only; constant/rare teachers flagged; no model performance claim',
      'existing_protection':'lstat metadata for every pre-existing file; SHA-256 for Phase 1, declared sources, hierarchy definitions and implementation; model content not hashed',
      'generated_artifact_sha256':{relpath(p,output):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and 'checks' not in p.relative_to(output).parts}}
    if write_report:write(output/'checks/validation_results.json',result)
    print(canonical({'validation':'passed','checks':len(tests),'classifiers':len(entries),'counts':counts,'report_written':write_report}),flush=True)
    return result


def main(command):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root',type=Path,default=ROOT)
    parser.add_argument('--phase-one-root',type=Path,default=DEFAULT_PHASE_ONE)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    if command=='validate':parser.add_argument('--write-report',action='store_true')
    args=parser.parse_args();root=args.repository_root.resolve()
    phase=(root/args.phase_one_root).resolve() if not args.phase_one_root.is_absolute() else args.phase_one_root.resolve()
    output=(root/args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    try:
        if command=='create':create(root,phase,output)
        elif command=='summarize':summarize(root,output)
        elif command=='validate':validate(root,phase,output,args.write_report)
    except (ValueError,FileExistsError,FileNotFoundError,KeyError,TypeError) as error:
        parser.exit(1,f'STOP: {error}\nExisting artifacts were not overwritten. Inspect the failure before continuing.\n')
