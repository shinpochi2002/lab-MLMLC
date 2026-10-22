import json, csv, html, unicodedata, hashlib, collections, pathlib, os, math, ast
R=pathlib.Path('/home/nakagawa/lab-MLMLC')
B=R/'data/hierarchical_api_selection'; P1=B/'group_5fold_v1'; P2=B/'classifier_cv_v1'; P3=B/'api_negative_sampling_v2'
def read(p):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate JSON key '+k+' in '+str(p))
            out[k]=v
        return out
    return json.loads(pathlib.Path(p).read_text(),object_pairs_hook=pairs)
def sha(p): return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def norm(s): return ' '.join(unicodedata.normalize('NFKC',html.unescape(s)).split())
def snapshot():
    out={}
    for base,dirs,files in os.walk(R):
        dirs[:]=[d for d in dirs if d!='.git']
        for f in files:
            p=pathlib.Path(base)/f; s=p.lstat();out[str(p.relative_to(R))]=(s.st_size,s.st_mtime_ns,s.st_mode,os.readlink(p) if p.is_symlink() else None)
    return out
before=snapshot(); errors=[]; checks=collections.Counter()
def check(ok,kind,where=''):
    checks[kind]+=1
    if not ok:errors.append({'kind':kind,'where':where})
schema=read(P1/'manifests/label_schema.json');records=read(P1/'source_records/category_classification.json'); lookup={r['entity_id']:r for r in records}
check(len(records)==len(lookup)==21907,'phase1_count')
check(records==read(P1/'source_records/unified_services.json'),'category_unified_equal')
rawm=read(R/'data/raw/mashup_data/mashup_data_new.json'); rawa=read(R/'data/raw/new_api_data/api_data_new.json')
filtered=read(R/'data/processed/softfill_mashup_data_new_api/nocategory/softfill_mashup_new_data.json')
expanded=read(R/'data/raw/new_api_data/new_api_data_to_4category.json')
ev=read(R/'results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json')
fixed=read(R/'data/splits/model_selection/manifest.json')
fixednames={k for s in fixed['splits'].values() for k in s['keys']}
check(set(ev)==fixednames and len(ev)==336,'fixed_keys')
evals={}; same_name=[]
for role,s in fixed['splits'].items():
    ids={'mashup:'+str(rawm[k][-1]) for k in s['keys']}; texts={norm(ev[k][1]) for k in s['keys']}
    check(len(ids)==s['count'],'fixed_count',role)
    check(set(s['ids'])=={rawm[k][-1] for k in s['keys']},'fixed_source_ids',role)
    for k in s['keys']:
        check(norm(ev[k][1])==norm(rawm[k][1]),'fixed_raw_text',k)
    evals[role]={'ids':ids,'texts':texts}
    check(not ids&set(lookup),'fixed_entity_separation',role)
    check(not texts&{norm(r['description']) for r in records},'fixed_text_separation',role)
all_eval_text={norm(v[1]) for v in ev.values()}; all_eval_ids=set.union(*(e['ids'] for e in evals.values()))
# Independent raw population reconstruction: identity, description and primary-category teacher.
expected={};excluded=[]
for typ,src,keys in [('mashup',filtered,filtered),('api',rawa,expanded)]:
    for name in keys:
        v=src[name]; eid=typ+':'+str(rawm[name][-1] if typ=='mashup' else v[-1]);text=norm(v[1])
        if eid in all_eval_ids or text in all_eval_text or not text:
            excluded.append((eid,name));continue
        cats=sorted({rawa[a][0][0] for a in v[2]}) if typ=='mashup' else [v[0][0]]
        expected[eid]={'description':v[1],'categories':cats,'source_key':name,'source_type':typ}
check(set(expected)==set(lookup),'raw_population')
for eid,r in lookup.items():
    e=expected.get(eid,{})
    check(r['description']==e.get('description') and r['category_labels']==e.get('categories') and r['source_key']==e.get('source_key') and r['source_type']==e.get('source_type'),'raw_teacher_text',eid)
    check(r['normalized_text_sha256']==hashlib.sha256(norm(r['description']).encode()).hexdigest(),'text_hash',eid)
    if r['source_key'] in fixednames:
        same_name.append({'name':r['source_key'],'entity_id':eid,'evaluation_entity_id':'mashup:'+str(rawm[r['source_key']][-1]),'equal_text':norm(r['description'])==norm(ev[r['source_key']][1])})
bytext=collections.defaultdict(list)
for eid,r in lookup.items():bytext[norm(r['description'])].append(eid)
textgroup={}; groupmembers={}
for text,ids in bytext.items():
    gid='group:'+hashlib.sha256(json.dumps(sorted(ids),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    groupmembers[gid]=sorted(ids)
    for eid in ids:
        textgroup[eid]=text
        check(lookup[eid]['group_id']==gid,'group_reconstruction',eid)
savedgroups=read(P1/'source_records/groups.json')
check({g['group_id']:g['entity_ids'] for g in savedgroups}==groupmembers,'group_inventory')
entries=[e for e in read(P2/'manifests/classifier_manifest.json')['classifiers'] if e['task']=='category_classification']
index={e['classifier_uid']:e for e in entries};check(len(index)==len(entries)==653,'manifest_unique_count')
expected_uids=set(); classifier_rows=[];split_rows=[];label_rows=[]; count_rows=[]; cv_rows=[]
for pat in schema['patterns']:
    name=pat['pattern']; depth=pat['category_layers']; definition=read(R/pat['definition_path'])
    paths={x['category']:[x['layer'+str(i)] for i in range(1,depth)] for x in definition['category_routes']}
    check(paths==pat['category_paths']==definition['category_paths'],'paths_vs_original',name)
    check(set(paths)==set(schema['categories']) and len(paths)==152,'category_coverage',name)
    # Build a trie of full routes from the ORIGINAL routes, including terminal category names.
    trie={}
    for cat,path in paths.items():
        node=trie
        for step in path:node=node.setdefault(step,{})
        node[cat]=None
    pools=collections.defaultdict(lambda:collections.defaultdict(set))
    for eid,e in expected.items():
        for cat in e['categories']:
            path=paths[cat]
            for level in range(depth):
                parent=tuple(path[:level]); label=cat if level==depth-1 else '-'.join(map(str,path[:level+1]))
                pools[parent][eid].add(label)
    layers=collections.Counter()
    def visit(node,parent=()):
        layer=len(parent)+1;layers[layer]+=1
        uid=f'category_classification/{name}/layer_{layer:02d}_parent_'+('_'.join(map(str,parent)) or 'root');expected_uids.add(uid)
        labels=sorted(str(k) if layer==depth else '-'.join(map(str,parent+(k,))) for k in node)
        # Tree membership gives label set. Saved schema defines sorted string output order.
        if uid not in index:
            check(False,'missing_classifier',uid);return
        entry=index[uid];p=P2/entry['artifact_path'];obj=read(p)
        for container in [entry,obj]:
            check(container['output_labels']==labels,'output_fixed_order',uid)
            check(container['parent_path']==list(parent) and container['layer']==layer and container['pattern']==name,'classifier_structure',uid)
        check(obj['classifier_uid']==uid and obj['hierarchy_definition']==pat['definition_path'],'classifier_identity',uid)
        check(sha(p)==entry['artifact_sha256'],'artifact_manifest_hash',uid)
        actual={r['entity_id']:set(r['target_labels']) for r in obj['members']}; target=pools[parent]
        check(len(actual)==len(obj['members']),'member_unique',uid)
        check(actual==target,'teacher_exact',uid)
        for m in obj['members']:
            eid=m['entity_id'];check(len(m['target_labels'])==len(set(m['target_labels'])) and set(m['target_labels'])<=set(labels),'valid_target',uid+'/'+eid)
            check(m['group_id']==lookup[eid]['group_id'] and m['source_type']==lookup[eid]['source_type'],'member_source',uid+'/'+eid)
        ids=set(target); part=obj['partition'];n=len(ids);g=len({lookup[e]['group_id'] for e in ids})
        method='kfold' if n<20 and g>=2 else 'holdout' if g>=2 else 'unavailable'; nf=min(5,g) if method=='kfold' else 1 if method=='holdout' else 0
        check(part['method']==method and part['n_splits']==nf and len(part['folds'])==nf,'method_threshold',uid)
        fullpos=collections.Counter(z for e in ids for z in target[e])
        check(part['full_profile']['label_supports']=={z:{'positive':fullpos[z],'negative':n-fullpos[z]} for z in labels},'full_support',uid)
        check(entry['services']==n and entry['groups']==g and entry['positive_supports']=={z:fullpos[z] for z in labels},'manifest_support',uid)
        cr={'classifier_uid':uid,'layer':layer,'pattern':name,'services':n,'groups':g,'outputs':len(labels),'multi_hot_members':sum(len(t)>1 for t in target.values()),'method':method,'splits':nf,'teacher_match':actual==target,'single_output':len(labels)==1,'full_constant':all(fullpos[z] in (0,n) for z in labels)};classifier_rows.append(cr)
        seen=collections.Counter()
        for f in part['folds']:
            fid=f['fold_id'];tr=f['train_entity_ids'];va=f['validation_entity_ids'];ts=set(tr);vs=set(va)
            tg={lookup[e]['group_id'] for e in tr};vg={lookup[e]['group_id'] for e in va}
            check(bool(tr) and bool(va),'nonempty',uid+'/'+fid)
            check(len(ts)==len(tr) and len(vs)==len(va) and not ts&vs,'entity_isolation',uid+'/'+fid)
            check(not tg&vg and not {textgroup[e] for e in tr}&{textgroup[e] for e in va},'text_group_isolation',uid+'/'+fid)
            check(ts|vs==ids,'partition_union',uid+'/'+fid)
            seen.update(vg)
            for role,subset in [('train',tr),('validation',va)]:
                pos=collections.Counter(z for e in subset for z in target[e]); profile=f[role+'_profile']
                check(profile['label_supports']=={z:{'positive':pos[z],'negative':len(subset)-pos[z]} for z in labels},'fold_support',uid+'/'+fid+'/'+role)
                check(profile['services']==len(subset) and profile['groups']==len({lookup[e]['group_id'] for e in subset}),'fold_counts',uid+'/'+fid+'/'+role)
                for er,e in evals.items():
                    check(not set(subset)&e['ids'] and not {textgroup[x] for x in subset}&e['texts'],'split_fixed_separation',uid+'/'+fid+'/'+role+'/'+er)
            tp=collections.Counter(z for e in tr for z in target[e]);vp=collections.Counter(z for e in va for z in target[e])
            lr=[]
            for z in labels:
                a=tp[z];b=len(tr)-a;c=vp[z];d=len(va)-c
                lr.append({'classifier_uid':uid,'fold_id':fid,'label':z,'train_positive':a,'train_negative':b,'validation_positive':c,'validation_negative':d,'train_pos_weight':b/a if a and b else 1.,'train_zero_positive':a==0,'train_zero_negative':b==0,'validation_zero_positive':c==0})
            label_rows.extend(lr)
            sr={'classifier_uid':uid,'fold_id':fid,'method':method,'train':len(tr),'validation':len(va),'train_groups':len(tg),'validation_groups':len(vg),'validation_ratio':len(va)/n,'validation_group_ratio':len(vg)/g,'train_zero_positive':sum(r['train_zero_positive'] for r in lr),'train_zero_negative':sum(r['train_zero_negative'] for r in lr),'validation_zero_positive':sum(r['validation_zero_positive'] for r in lr),'train_constant':all(r['train_positive'] in (0,len(tr)) for r in lr),'validation_constant':all(r['validation_positive'] in (0,len(va)) for r in lr)};split_rows.append(sr)
            if method=='holdout':check(len(vg)==max(1,min(g-1,math.floor(g*.2+.5))),'holdout_group_capacity',uid)
        if method=='kfold':
            check(set(seen)=={lookup[e]['group_id'] for e in ids} and set(seen.values())=={1},'cv_validation_once',uid)
            cv_rows.append({'classifier_uid':uid,'services':n,'groups':g,'validation_group_visits':dict(seen)})
        for k,v in node.items():
            if v is not None:visit(v,parent+(k,))
    visit(trie)
    for layer,n in sorted(layers.items()):
        got=sum(e['pattern']==name and e['layer']==layer for e in entries)
        check(got==n,'layer_count',name+'/'+str(layer));count_rows.append({'pattern':name,'layer':layer,'expected':n,'actual':got})
check(set(index)==expected_uids,'complete_classifier_inventory')
files=list((P2/'splits/category_classification').rglob('*.json'));check(len(files)==653 and {str(p.relative_to(P2)) for p in files}=={e['artifact_path'] for e in entries},'file_inventory')
# Independently compare CSV support records, in addition to JSON profiles.
support_index={(r['classifier_uid'],r['fold_id'],r['role'],r['label']):r for r in csv.DictReader(open(P2/'reports/label_supports.csv'))}
for r in label_rows:
    for role in ['train','validation']:
        saved=support_index.get((r['classifier_uid'],r['fold_id'],role,r['label']),{})
        check(saved.get('positive_services')==str(r[role+'_positive']) and saved.get('negative_services')==str(r[role+'_negative']),'csv_support',r['classifier_uid']+'/'+r['fold_id']+'/'+r['label']+'/'+role)
# Saved historical hash anchors: no existing validator is invoked.
hash_rows=[]
for version in [P1,P2,P3]:
    src=read(version/'manifests/source_manifest.json')
    for field in ['source_files','input_sha256','implementation_sha256']:
        for p,h in src.get(field,{}).items():
            path=R/p;ok=path.is_file() and sha(path)==h
            hash_rows.append({'anchor':str(version.relative_to(R))+'/'+field,'path':p,'matches':ok});check(ok,'historical_hash',p)
    for p in (version/'checks').glob('*validation*results.json'):
        obj=read(p)
        for rel,h in obj.get('generated_artifact_sha256',{}).items():
            path=version/rel;ok=path.is_file() and sha(path)==h
            hash_rows.append({'anchor':str(p.relative_to(R)),'path':str(path.relative_to(R)),'matches':ok});check(ok,'saved_artifact_hash',str(path))
# AST only; do not import or execute any training, splitter, or validation module.
for p in (R/'scripts/hierarchical_api_selection/api_negative_sampling').glob('*.py'):ast.parse(p.read_text())
after=snapshot();changed=[p for p,v in before.items() if after.get(p)!=v];check(not changed,'audit_nonmutation')
summary={'audit_date_jst':'2026-10-21','checks':dict(checks),'errors':errors,'existing_files_snapshotted':len(before),'existing_files_changed':changed,'services':len(records),'groups':len(bytext),'source_counts':dict(collections.Counter(r['source_type'] for r in records)),'excluded_raw_population':len(excluded),'same_name_distinct_apis':same_name,'classifiers':len(classifier_rows),'pairs':len(split_rows),'label_fold_outputs':len(label_rows),'methods':dict(collections.Counter(r['method'] for r in classifier_rows)),'multi_hot_members':sum(r['multi_hot_members'] for r in classifier_rows),'total_member_assignments':sum(r['services'] for r in classifier_rows),'cv_classifiers':cv_rows,'hash_checks':len(hash_rows),'zero_summary':{k:{'outputs':sum(r[k] for r in label_rows),'pairs':sum(r[k]>0 for r in split_rows),'classifiers':len({r['classifier_uid'] for r in split_rows if r[k]>0})} for k in ['train_zero_positive','train_zero_negative','validation_zero_positive']},'full_constant_classifiers':sum(r['full_constant'] for r in classifier_rows),'single_output_classifiers':sum(r['single_output'] for r in classifier_rows)}
holds=[r for r in split_rows if r['method']=='holdout'];summary['holdout_range']={k:[min(r[k] for r in holds),max(r[k] for r in holds)] for k in ['validation_ratio','validation_group_ratio','train','validation']}
dest=R/'category_split_independent_audit_artifacts'
i=1
while dest.exists():dest=R/f'category_split_independent_audit_artifacts_{i}';i+=1
dest.mkdir()
for name,rows in [('classifiers',classifier_rows),('splits',split_rows),('label_supports',label_rows),('layer_counts',count_rows),('hash_checks',hash_rows)]:
    with (dest/(name+'.csv')).open('x') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (dest/'summary.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
with (dest/'audit_script.py').open('x') as f:f.write(pathlib.Path(__file__).read_text())
print(json.dumps({'artifacts':str(dest),'summary':{k:v for k,v in summary.items() if k not in ['checks','cv_classifiers']}},ensure_ascii=False,indent=2))
