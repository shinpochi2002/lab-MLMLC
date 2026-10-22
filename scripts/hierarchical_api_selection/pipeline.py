"""Independent, standard-library-only dataset preparation. Never imports training code."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import heapq
import io
import json
import math
import os
import random
import stat
import subprocess
import unicodedata
import html
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('data/hierarchical_api_selection/group_5fold_v1')
SOURCES = {
    'raw_mashups': 'data/raw/mashup_data/mashup_data_new.json',
    'filtered_mashups': 'data/raw/mashup_data/updated_mashup_new_data.json',
    'mashups': 'data/processed/softfill_mashup_data_new_api/nocategory/softfill_mashup_new_data.json',
    'raw_apis': 'data/raw/new_api_data/api_data_new.json',
    'expanded_apis': 'data/raw/new_api_data/new_api_data_to_4category.json',
    'expanded_services': 'data/processed/softfill_mashup_data_new_api_inclu_nomash/nocategory/softfill_mashup_data_new_api_category_inclu_nomash.json',
    'api_candidates': 'data/processed/root_json/api_dataset_like_manifest.json',
    'api_categories': 'data/processed/root_json/api_category_pairs.json',
    'evaluation': 'results/hierarchy/output/layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json',
    'fixed_split': 'data/splits/model_selection/manifest.json',
}
# Existing published identifiers, definition folders, and model folder correspondence.
PATTERNS = [
 ('layer2_4_43',2,'4-43','layer2/layer2_4'),
 ('layer2_6_25',2,'6-25','layer2/layer2_6'),
 ('layer2_8_19',2,'8-19','layer2/layer2_8'),
 ('layer2_12_13',2,'12_13','layer2/layer2_12'),
 ('layer2_15_10',2,'15-10','layer2/layer2_15'),
 ('layer2_24_7',2,'24-7','layer2/layer2_24'),
 ('layer2_33_4',2,'33-4','layer2/layer2_33'),
 ('layer2_51_3',2,'51-3','layer2/layer2_51'),
 ('layer3_4_4_8',3,'4cluster','layer3/layer3_4_4_8'),
 ('layer3_8_4_4',3,'8_4_4','layer3/layer3_8_4_4'),
 ('layer3_12_2_6',3,'12_2_6','layer3/layer3_12_2_6'),
 ('layer3_12_4_3',3,'12_4_3','layer3/layer3_12_4_3'),
 ('layer3_12_6_2',3,'12_6_2','layer3/layer3_12_6_2'),
 ('layer3_16_4_3',3,'16cluster','layer3/layer3_16_4_3'),
 ('layer4_8_3_2_3',4,'8_3_2_3','layer4/layer4_8_3_2_3'),
 ('layer4_12_2_2_3',4,'12_2_3_2','layer4/layer4_12_2_2_3'),
]
# Assertions against research constraints/reference counts, never used to select or pad data.
EXPECTED_SOURCE = dict(raw_mashups=6184, filtered_mashups=4284, mashups=4284,
 raw_apis=21526, expanded_apis=17961, expanded_services=22180,
 api_candidates=909, api_categories=909, evaluation=336)
EXPECTED_RESULT = {'category_services':21907,'api_positive_services':4855,
 'restored_mashups':61,'categories':152,'api_candidates':909,'patterns':16}
NORMALIZATION = 'html.unescape -> Unicode NFKC -> str.split/join; case preserved; v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        require(k not in d, f'Duplicate JSON key: {k!r}')
        d[k]=v
    return d


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_pairs)


def write(path, value):
    """Exclusive creation, including during retries. Existing outputs are never updated."""
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        if isinstance(value,str): f.write(value)
        else: json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2); f.write('\n')


def write_csv(path, rows, columns=None):
    rows=list(rows)
    columns=columns or (list(rows[0]) if rows else [])
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=columns,lineterminator='\n');w.writeheader()
    for row in rows:
        w.writerow({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    write(path,buf.getvalue())


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',html.unescape(text)).split())


def fingerprint(description, apis):
    return (normalize(description),tuple(sorted(set(apis))))


def inventory(root, exclude=None):
    result={}
    for base,dirs,files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d!='.git' and (exclude is None or (Path(base)/d).resolve()!=exclude.resolve()))
        for name in sorted(files):
            p=Path(base)/name;s=p.lstat()
            result[p.relative_to(root).as_posix()]={'size':s.st_size,'mtime_ns':s.st_mtime_ns,
              'mode':s.st_mode,'link':os.readlink(p) if stat.S_ISLNK(s.st_mode) else None}
    return result


def check_inventory(root, before):
    changes=[]
    for rel,old in before.items():
        p=root/rel
        if not p.exists() and not p.is_symlink(): changes.append({'path':rel,'reason':'missing'});continue
        s=p.lstat();now={'size':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':s.st_mode,
          'link':os.readlink(p) if p.is_symlink() else None}
        if now!=old:changes.append({'path':rel,'reason':'metadata_changed'})
    return changes


def provenance(root, source_files):
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    except subprocess.CalledProcessError:commit=None
    code={p.relative_to(root).as_posix():sha(p) for p in sorted((root/'scripts/hierarchical_api_selection').rglob('*.py'))}
    return {'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'git_commit':commit,'implementation_sha256':code,
      'source_files':{p:sha(root/p) for p in sorted(set(source_files))},
      'normalization':NORMALIZATION,'seed':42,'n_folds':5,
      'scope':'Phase 1/2 only; no training, inference, clustering, or negative-record generation'}


def load_inputs(root):
    data={k:read(root/v) for k,v in SOURCES.items()}
    for key,n in EXPECTED_SOURCE.items():
        require(len(data[key])==n,f'Source count mismatch: {key}: {len(data[key])}, reference {n}; stop without correction')
    raw,m,a,pa,x,c,ac,ev=(data[k] for k in ['raw_mashups','mashups','raw_apis','expanded_apis','expanded_services','api_candidates','api_categories','evaluation'])
    require(set(m)==set(data['filtered_mashups']), 'Filtered Mashup key mismatch')
    require(set(m)<=set(raw), 'A Mashup source key is missing from the original raw file')
    for label,src in [('raw Mashup',raw),('raw API',a)]:
        ids=[v[-1] for v in src.values()]
        require(all(isinstance(i,int) and not isinstance(i,bool) for i in ids),f'Noninteger {label} ID')
        require(len(set(ids))==len(ids),f'Duplicate {label} source IDs; identity rule must be reviewed')
    require(set(x)==set(m)|set(pa),'Expanded service union differs from original provenance')
    require(set(c)==set(ac) and set(c)<=set(pa),'Candidate catalog relationship mismatch')
    cats=sorted({cat for v in x.values() for cat in v[0]})
    require(len(cats)==152,'Category space differs from fixed 152 categories')
    require({a[k][0][0] for k in c}==set(cats),'Candidates do not cover fixed category space')
    require(all(ac[k]==a[k][0][0] for k in c),'Candidate API/category mismatch')
    require(all(set(v[2])<=set(c) for v in m.values()),'Mashup uses an API outside fixed candidates')
    for k,v in m.items():
        require(v==data['filtered_mashups'][k],f'Filtered Mashup source differs: {k}')
        require(raw[k][1]==v[1] and set(raw[k][2])==set(v[2]),f'Raw Mashup text/API source mismatch: {k}')
    for k,v in x.items():
        src=a[k] if k in pa else m[k]
        labels={a[k][0][0]} if k in pa else {a[z][0][0] for z in src[2]}
        require(v[1]==src[1] and set(v[0])==labels,f'Expanded provenance/label mismatch: {k}')
    fixed=data['fixed_split'];vk=set(fixed['splits']['validation']['keys']);tk=set(fixed['splits']['test']['keys'])
    require(len(vk)==100 and len(tk)==236 and not vk&tk and vk|tk==set(ev),'Fixed evaluation split identity mismatch')
    for k,v in ev.items():
        require(k in m and fingerprint(v[1],v[2])==fingerprint(m[k][1],m[k][2]),f'Fixed evaluation source mismatch: {k}')
    files=list(SOURCES.values())
    # Validate every fixed dataset, not only the canonical file; keep existing masking and GT.
    for rel,meta in fixed['datasets'].items():
        for split,keys in [('validation',vk),('test',tk)]:
            p=Path('data/splits/model_selection')/split/rel;dd=read(root/p)
            require(set(dd)==keys and sha(root/p)==meta['sha256'][split],f'Fixed artifact mismatch: {p}')
            for k,v in dd.items():
                description,apis,numeric_id=(v['description'],v['apis'],v['id']) if isinstance(v,dict) else (v[1],v[2],v[3])
                require(set(apis)==set(ev[k][2]) and numeric_id==ev[k][3],f'Fixed label/ID mismatch: {p}/{k}')
            files.append(p.as_posix())
    patterns=[]
    for name,layers,folder,model in PATTERNS:
        base=Path('results/hierarchy/output/by_pattern')/folder
        options=[(p,read(p)) for p in sorted((root/base).glob('category_hierarchy*.json'))]
        options=[(p,h) for p,h in options if h['max_depth']==layers-1]
        require(len(options)==1,f'Ambiguous/missing fixed hierarchy definition: {name}')
        p,h=options[0];paths={k:list(v) for k,v in h['category_paths'].items()}
        require(set(paths)==set(cats) and all(len(v)==layers-1 for v in paths.values()),f'Fixed hierarchy categories/depth mismatch: {name}')
        files.append(p.relative_to(root).as_posix())
        patterns.append({'pattern':name,'category_layers':layers,'definition_path':p.relative_to(root).as_posix(),
          'category_paths':paths,'existing_model_directory':'models/checkpoints/hierarchy/'+model,
          'existing_api_model_directory':'models/checkpoints/category_kfold/study_category_inc_nomashup_model_new_kfold',
          'inference_script':f'scripts/inference/mashup_dis_nosp_{layers}layer_api.py'})
    data['patterns']=patterns;data['categories']=cats;data['files']=files
    return data


def make_groups(records):
    """Confirmed entity identity is unique already; identical normalized input links all origins."""
    bytext=collections.defaultdict(list)
    for r in records:
        require(normalize(r['description']),f'Empty normalized text: {r["entity_id"]}')
        bytext[normalize(r['description'])].append(r)
    groups=[]
    for text,rs in bytext.items():
        ids=sorted(r['entity_id'] for r in rs)
        gid='group:'+digest(canonical(ids))
        for r in rs:r['group_id']=gid
        groups.append({'group_id':gid,'entity_ids':ids,'normalized_text_sha256':digest(text),
          'service_count':len(rs),'source_counts':dict(sorted(collections.Counter(r['source_type'] for r in rs).items()))})
    return sorted(groups,key=lambda g:g['group_id'])


def reconstruct(root):
    d=load_inputs(root);m,a,raw,pa,x,c,ac,ev=(d[k] for k in ['mashups','raw_apis','raw_mashups','expanded_apis','expanded_services','api_candidates','api_categories','evaluation'])
    collision=set(m)&set(pa);require(len(collision)==65,'Collision baseline changed')
    efp={fingerprint(v[1],v[2]) for v in ev.values()};etext=collections.defaultdict(list)
    for k,v in ev.items():etext[normalize(v[1])].append(k)
    eid_eval={'mashup:'+str(raw[k][-1]) for k in ev}
    records=[];excluded=[];restored=[];same_name_apis=[]
    for source_type,src,keys,source_path in [
      ('mashup',m,sorted(m),SOURCES['mashups']),('api',a,sorted(pa),SOURCES['raw_apis'])]:
        for key in keys:
            v=src[key];sid=raw[key][-1] if source_type=='mashup' else v[-1];eid=source_type+':'+str(sid)
            used=list(v[2]) if source_type=='mashup' else None
            categories=sorted({a[z][0][0] for z in used}) if source_type=='mashup' else [v[0][0]]
            eligible=source_type=='mashup' or key in c
            api_labels=sorted(set(used)) if source_type=='mashup' else [key] if eligible else None
            r={'entity_id':eid,'source_type':source_type,'source_id':sid,'source_key':key,
               'source_path':SOURCES['raw_mashups'] if source_type=='mashup' else source_path,
               'teacher_source_path':source_path,'source_record_sha256':digest(canonical(raw[key] if source_type=='mashup' else v)),
               'description':v[1],'category_labels':categories,'original_source_categories':list(v[0]),
               'used_apis':used,'api_labels':api_labels,'api_classification_eligible':eligible,
               'api_category_pairs':{z:ac[z] for z in api_labels} if eligible else None,
               'normalized_text_sha256':digest(normalize(v[1])),
               'present_as_entity_in_expanded_data':key in x and (source_type=='api' or key not in pa),
               'restored_mashup':source_type=='mashup' and key in collision,
               'expanded_source_key':key if key in x else None}
            reason=[];matches=[]
            if source_type=='mashup' and (eid in eid_eval or fingerprint(v[1],used) in efp):reason.append('fixed_evaluation_mashup_entity')
            if normalize(v[1]) and normalize(v[1]) in etext:
                matches=sorted(etext[normalize(v[1])]);reason.append('fixed_evaluation_normalized_text')
            if not normalize(v[1]):reason.append('empty_description')
            if reason:
                excluded.append({'entity_id':eid,'source_type':source_type,'source_id':sid,'source_key':key,
                  'source_path':r['source_path'],'teacher_source_path':source_path,'source_record_sha256':r['source_record_sha256'],
                  'present_as_entity_in_expanded_data':r['present_as_entity_in_expanded_data'],
                  'collision_mashup_not_restored':r['restored_mashup'],'reasons':reason,'fixed_evaluation_names':matches,
                  'normalized_text_sha256':r['normalized_text_sha256']})
                continue
            records.append(r)
            if r['restored_mashup']:restored.append(dict(r))
            if source_type=='api' and key in ev:
                same_name_apis.append({'api_entity_id':eid,'source_key':key,'evaluation_mashup_entity_id':'mashup:'+str(raw[key][-1]),
                  'equal_normalized_text':normalize(v[1])==normalize(ev[key][1]),'decision':'retain distinct API entity; do not restore evaluation Mashup'})
    records.sort(key=lambda r:r['entity_id']);groups=make_groups(records)
    lookup={r['entity_id']:r for r in records};restored=[lookup[r['entity_id']] for r in restored]
    positive=[r for r in records if r['api_classification_eligible']]
    counts={'category_services':len(records),'api_positive_services':len(positive),'restored_mashups':len(restored),
      'categories':len(d['categories']),'api_candidates':len(c),'patterns':len(d['patterns']),
      'mashup_services':sum(r['source_type']=='mashup' for r in records),
      'candidate_api_services':sum(r['source_type']=='api' and r['api_classification_eligible'] for r in records),
      'noncandidate_api_services':sum(not r['api_classification_eligible'] for r in records),'groups':len(groups),
      'api_positive_groups':len({r['group_id'] for r in positive}),
      'excluded_evaluation_entities':sum('fixed_evaluation_mashup_entity' in r['reasons'] for r in excluded),
      'excluded_text_only':sum(r['reasons']==['fixed_evaluation_normalized_text'] for r in excluded),
      'excluded_empty_description':sum('empty_description' in r['reasons'] for r in excluded)}
    for k,v in EXPECTED_RESULT.items():require(counts[k]==v,f'Reconstructed count mismatch {k}: {counts[k]} vs reference {v}; stop')
    require(len(lookup)==len(records),'Duplicate entity identity')
    mismatch=[]
    for k,v in ev.items():
        expected=sorted({a[z][0][0] for z in m[k][2]})
        if expected!=sorted(set(v[0])):mismatch.append({'name':k,'fixed_categories':v[0],'mashup_derived_categories':expected})
    schema={'api_candidates':sorted(c),'categories':d['categories'],'api_categories':dict(sorted(ac.items())),
      'patterns':d['patterns'],'api_label_space':'fixed catalog of 909 APIs; branch outputs are fixed subsets, not observed-fold labels',
      'api_ineligible_target':'null, never a 909-dimensional all-zero negative'}
    exclusions={'records':excluded,'same_name_distinct_apis_retained':same_name_apis,'collision_names':sorted(collision),
      'restored_entity_ids':sorted(r['entity_id'] for r in restored),'fixed_category_gt_warnings':mismatch,
      'fixed_evaluation_entity_ids':sorted(eid_eval),'fixed_evaluation_text_sha256':sorted(digest(t) for t in etext),
      'rule':'exclude evaluation entity or identical normalized text, then empty text, before grouping/branch expansion'}
    return d,records,positive,groups,restored,counts,schema,exclusions


def prepare(root, output):
    require(not output.exists(),f'Output already exists; refusing any overwrite: {output}')
    before=inventory(root,output)
    d,records,positive,groups,restored,counts,schema,exclusions=reconstruct(root)
    source=provenance(root,d['files']);meta={k:source[k] for k in ['schema_version','generated_at_utc','git_commit','implementation_sha256','seed','n_folds']}
    meta['source_manifest']='manifests/source_manifest.json'
    source['observed_source_counts']={k:len(d[k]) for k in EXPECTED_SOURCE}
    source['source_id_rule']='raw Mashup/raw API integer IDs, unique within each immutable source; entity_id=source_type:source_id'
    source['existing_files_verification']='SHA-256 for declared inputs/code; lstat size/mtime/mode/link for every pre-existing repository file; model weights are not content-hashed'
    source['existing_file_snapshot_path']='manifests/existing_file_snapshot.json'
    source['existing_file_count']=len(before)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('source_manifest',source),('existing_file_snapshot',{**meta,'files':before}),
      ('dataset_manifest',{**meta,'counts':counts,'reference_counts':EXPECTED_RESULT,'label_schema':'manifests/label_schema.json',
        'category_records':'source_records/category_classification.json','api_positive_records':'source_records/api_classification_positive.json',
        'unified_records':'source_records/unified_services.json','groups':'source_records/groups.json',
        'restored_records':'source_records/restored_mashups.json','excluded_records':len(exclusions['records']),
        'negative_records_generated':0,'data_role_overlap':'API positive records are a subset of category/unified records, not additional independent services'}),
      ('label_schema',{**meta,**schema}),('exclusion_manifest',{**meta,**exclusions})]:write(output/'manifests'/f'{name}.json',value)
    for name,value in [('unified_services',records),('category_classification',records),
      ('api_classification_positive',positive),('groups',groups),('restored_mashups',restored)]:write(output/'source_records'/f'{name}.json',value)
    print(canonical({'phase':1,'counts':counts,'output':str(output)}))


def group_features(records, groups):
    lookup={r['entity_id']:r for r in records};result={}
    for group in groups:
        f=collections.Counter()
        for eid in group['entity_ids']:
            r=lookup[eid]
            f.update('category:'+c for c in r['category_labels'])
            f.update('api:'+a for a in r['api_labels'] or [])
            kind='mashup' if r['source_type']=='mashup' else 'candidate_api' if r['api_classification_eligible'] else 'noncandidate_api'
            f['source:'+kind]+=1
        result[group['group_id']]=f
    return result


def stratify(features, fractions, seed=42):
    """Deterministic rarest-remaining-label group allocation with hard group capacities.

    Multilabel deficits are soft objectives; indivisible groups are never split/moved.
    """
    require(features and all(x>0 for x in fractions) and math.isclose(sum(fractions),1), 'Invalid split inputs')
    gids=sorted(features);rng=random.Random(seed);rng.shuffle(gids);rank={g:i for i,g in enumerate(gids)}
    n=len(gids);caps=[math.floor(n*p) for p in fractions]
    for i in sorted(range(len(caps)),key=lambda j:(-(n*fractions[j]-caps[j]),j))[:n-sum(caps)]:caps[i]+=1
    totals=collections.Counter()
    index=collections.defaultdict(set)
    for g in gids:
        totals.update(features[g])
        for label in features[g]:
            if not label.startswith('source:'):index[label].add(g)
    heap=[(len(gs),lab) for lab,gs in index.items()];heapq.heapify(heap)
    remaining=set(gids);loads=[collections.Counter() for _ in fractions];sizes=[0]*len(fractions);assigned={}
    def put(g, focus):
        options=[i for i in range(len(caps)) if sizes[i]<caps[i]];require(options,'Group capacity exhausted')
        def score(i):
            primary=(totals[focus]*fractions[i]-loads[i][focus]) if focus else 0
            other=sum(v*(totals[lab]*fractions[i]-loads[i][lab])/max(totals[lab],1) for lab,v in features[g].items())
            return (primary,other,(caps[i]-sizes[i])/max(caps[i],1),-i)
        i=max(options,key=score);assigned[g]=i;sizes[i]+=1;loads[i].update(features[g]);remaining.remove(g)
        for label in features[g]:
            if label in index:
                index[label].discard(g)
                if index[label]:heapq.heappush(heap,(len(index[label]),label))
    while heap:
        count,label=heapq.heappop(heap)
        if not index[label] or count!=len(index[label]):continue
        pending=sorted(index[label],key=lambda g:rank[g])
        for g in pending:
            if g in remaining:put(g,label)
    for g in sorted(remaining,key=lambda g:rank[g]):put(g,None)
    require(sizes==caps,'Unexpected group capacity imbalance')
    return assigned


def plan_folds(records, groups, seed=42):
    features=group_features(records,groups);outer=stratify(features,[.2]*5,seed)
    folds={}
    for i in range(5):
        name=f'fold_{i+1:02d}';val=sorted(g for g,k in outer.items() if k==i)
        training={g:f for g,f in features.items() if outer[g]!=i};inner=stratify(training,[.8,.2],seed)
        folds[name]={'outer_train_groups':sorted(training),'train_groups':sorted(g for g,k in inner.items() if k==0),
           'early_stopping_groups':sorted(g for g,k in inner.items() if k==1),'validation_groups':val}
    return outer,folds


def phase_files(output):
    return read(output/'source_records/unified_services.json'),read(output/'source_records/groups.json'),read(output/'manifests/label_schema.json')


def check_provenance(root, output):
    src=read(output/'manifests/source_manifest.json')
    for rel,value in {**src['source_files'],**src['implementation_sha256']}.items():
        require(sha(root/rel)==value,f'Existing input/code changed: {rel}')
    changes=check_inventory(root,read(output/src['existing_file_snapshot_path'])['files'])
    require(not changes,f'Pre-existing files changed: {changes[:10]}')
    return src


def create_folds(root, output):
    require(output.exists(),'Phase 1 output missing')
    require(not (output/'splits').exists() and not (output/'manifests/split_manifest.json').exists(),'Fold outputs already exist; stop')
    src=check_provenance(root,output);records,groups,schema=phase_files(output);outer,folds=plan_folds(records,groups)
    meta={k:src[k] for k in ['schema_version','generated_at_utc','git_commit','implementation_sha256','seed','n_folds']}
    meta.update(generated_at_utc=datetime.now(timezone.utc).isoformat(),source_manifest='manifests/source_manifest.json')
    manifest={**meta,'algorithm':'rarest_remaining_multilabel_group_deficit_v1',
      'inner_early_stopping_ratio':.2,'group_capacities':'largest remainder; hard group counts, no record splitting',
      'tie_breaking':'sorted group IDs shuffled with random.Random(42), then deterministic deficit scores; same seed for each independent inner split',
      'features':'per-group record support for 152 categories, 909 API labels, and source type',
      'role_definitions':{'outer_train':'train + early_stopping','train':'gradient-update subset only','validation':'outer held-out fold','early_stopping':'inner validation only'},
      'shared_by':[p['pattern'] for p in schema['patterns']],
      'group_count':len(groups),'service_count':len(records),'folds':{name:{role:len(gs) for role,gs in roles.items()} for name,roles in folds.items()},
      'forced_sample_moves':0,'trained_models':0}
    (output/'splits').mkdir(exist_ok=False)
    write_csv(output/'splits/group_assignments.csv',[{**g,'validation_fold':f'fold_{outer[g["group_id"]]+1:02d}'} for g in groups])
    write_csv(output/'splits/fold_assignments.csv',[{'entity_id':r['entity_id'],'group_id':r['group_id'],
      'source_type':r['source_type'],'source_id':r['source_id'],'api_classification_eligible':r['api_classification_eligible'],
      'validation_fold':f'fold_{outer[r["group_id"]]+1:02d}'} for r in records])
    for name,roles in folds.items():
        for role,gs in roles.items():write(output/'splits'/name/f'{role}.json',gs)
    write(output/'manifests/split_manifest.json',manifest)
    print(canonical({'phase':2,'groups':len(groups),'services':len(records),'fold_group_counts':manifest['folds']}))


def read_folds(output):
    return {f'fold_{i:02d}':{role:read(output/'splits'/f'fold_{i:02d}'/f'{role}.json') for role in
      ['outer_train_groups','train_groups','early_stopping_groups','validation_groups']} for i in range(1,6)}


def roles_by_entity(records, folds):
    result={}
    for name,roles in folds.items():
        gt={g:role.removesuffix('_groups') for role,gs in roles.items() if role!='outer_train_groups' for g in gs}
        result[name]={r['entity_id']:gt[r['group_id']] for r in records}
    return result


def project_classifiers(record, pattern):
    """One service per parent classifier, labels restricted to that parent's path."""
    paths=pattern['category_paths'];layers=pattern['category_layers'];by=collections.defaultdict(set)
    for cat in record['category_labels']:
        path=paths[cat]
        for depth in range(layers):
            prefix=tuple(path[:depth]);label=cat if depth==layers-1 else '-'.join(map(str,path[:depth+1]))
            by[(depth+1,prefix)].add(label)
    return by


def classifier_definitions(schema):
    definitions=[]
    for pattern in schema['patterns']:
        labels=collections.defaultdict(set)
        for cat,path in pattern['category_paths'].items():
            for depth in range(pattern['category_layers']):
                labels[(depth+1,tuple(path[:depth]))].add(cat if depth==pattern['category_layers']-1 else '-'.join(map(str,path[:depth+1])))
        for (layer,prefix),ls in sorted(labels.items()):
            definitions.append({'pattern':pattern['pattern'],'task':'category_classification','layer':layer,'parent_path':list(prefix),
              'classifier_id':'category/layer_'+str(layer)+'/'+('-'.join(map(str,prefix)) or 'root'),'output_labels':sorted(ls),
              'membership_rule':'any category_labels path has this parent_path prefix; preserve shared group assignment',
              'existing_model_directory':pattern['existing_model_directory'],'hierarchy_definition':pattern['definition_path']})
    cats=collections.defaultdict(list)
    for api,cat in schema['api_categories'].items():cats[cat].append(api)
    for cat,ls in sorted(cats.items()):
        definitions.append({'pattern':'shared_all_16','task':'api_classification','layer':'api','parent_path':[cat],
          'classifier_id':'api/'+cat,'output_labels':sorted(ls),
          'membership_rule':'api_classification_eligible and positive API in this category; intersect target with output_labels',
          'existing_model_directory':schema['patterns'][0]['existing_api_model_directory']})
    return definitions


def summarize(root, output):
    require(not (output/'reports').exists(),'Reports output already exists; stop')
    require(not (output/'README.md').exists() and not (output/'manifests/classifier_manifest.json').exists(),'Summary output already exists; stop')
    src=check_provenance(root,output);records,groups,schema=phase_files(output);folds=read_folds(output);rolemap=roles_by_entity(records,folds)
    dataset=read(output/'manifests/dataset_manifest.json');ex=read(output/'manifests/exclusion_manifest.json')
    allrole=['train','early_stopping','validation','outer_train'];dist=[];catdist=[];apidist=[]
    subsets={};neg_total={}
    for fold,rm in rolemap.items():
        for role in allrole:
            rs=[r for r in records if rm[r['entity_id']]==role or role=='outer_train' and rm[r['entity_id']]!='validation']
            subsets[(fold,role)]=rs
            source=collections.Counter('mashup' if r['source_type']=='mashup' else 'candidate_api' if r['api_classification_eligible'] else 'noncandidate_api' for r in rs)
            neg_total[(fold,role)]=source['mashup']
            dist.append({'fold':fold,'role':role,'services':len(rs),'groups':len({r['group_id'] for r in rs}),
              'api_positive_services':sum(r['api_classification_eligible'] for r in rs),'mashup_services':source['mashup'],
              'candidate_api_services':source['candidate_api'],'noncandidate_api_services':source['noncandidate_api'],
              'mashup_fraction':source['mashup']/len(rs) if rs else 0,'api_fraction':(source['candidate_api']+source['noncandidate_api'])/len(rs) if rs else 0})
            cc=collections.Counter(c for r in rs for c in r['category_labels']);aa=collections.Counter(a for r in rs for a in r['api_labels'] or [])
            catdist.extend({'fold':fold,'role':role,'category_label':c,'positive_services':cc[c]} for c in schema['categories'])
            apidist.extend({'fold':fold,'role':role,'api_label':a,'category_label':schema['api_categories'][a],'positive_services':aa[a]} for a in schema['api_candidates'])
    definitions=classifier_definitions(schema);classrows=[]
    def emit(definition, members):
        labels=definition['output_labels'];cid=definition['classifier_id'];pattern=definition['pattern'];api_task=definition['task']=='api_classification'
        counts={}
        for fold,rm in rolemap.items():
            supports={role:collections.Counter() for role in allrole};nums={role:collections.Counter() for role in allrole};gsets={role:set() for role in allrole}
            for eid,targets,source,gid in members:
                rr=rm[eid]
                for role in [rr]+(['outer_train'] if rr!='validation' else []):
                    nums[role]['services']+=1;nums[role][source]+=1;supports[role].update(targets);gsets[role].add(gid)
            for role in allrole:
                n=nums[role]['services'];zero=sum(supports[role][l]==0 for l in labels);one=sum(supports[role][l]==1 for l in labels)
                counts[role]=(n,zero)
                classrows.append({'fold':fold,'pattern':pattern,'task':definition['task'],'layer':definition['layer'],
                  'classifier_id':cid,'parent_path':definition['parent_path'],'role':role,'positive_services':n,
                  'groups':len(gsets[role]),'mashup_services':nums[role]['mashup'],'api_services':nums[role]['api'],
                  'output_label_count':len(labels),'zero_positive_labels':zero,'one_positive_labels':one,
                  'all_positive_labels':sum(n>0 and supports[role][l]==n for l in labels),
                  'known_mashup_negative_candidates':neg_total[(fold,role)]-nums[role]['mashup'] if api_task else '',
                  'mashup_fraction':nums[role]['mashup']/n if n else 0,'api_fraction':nums[role]['api']/n if n else 0,
                  'training_unavailable':int(nums['train']['services']==0),
                  'early_stopping_unavailable':int(nums['train']['services']==0 or nums['early_stopping']['services']==0),
                  'outer_validation_unavailable':int(nums['validation']['services']==0)})
    lookup={r['entity_id']:r for r in records}
    for pattern in schema['patterns']:
        memberships=collections.defaultdict(list)
        for r in records:
            for key,ls in project_classifiers(r,pattern).items():memberships[key].append((r['entity_id'],ls,r['source_type'],r['group_id']))
        for de in definitions:
            if de['pattern']==pattern['pattern']:emit(de,memberships[(de['layer'],tuple(de['parent_path']))])
    ac=schema['api_categories'];apimembers=collections.defaultdict(list)
    for r in records:
        if not r['api_classification_eligible']:continue
        labels=collections.defaultdict(set)
        for a in r['api_labels']:labels[ac[a]].add(a)
        for cat,ls in labels.items():apimembers['api/'+cat].append((r['entity_id'],ls,r['source_type'],r['group_id']))
    for de in definitions:
        if de['task']=='api_classification':emit(de,apimembers[de['classifier_id']])
    # API classifiers are identical/shared, but provide the requested complete 16-pattern report.
    apirows=[r for r in classrows if r['task']=='api_classification']
    classrows=[r for r in classrows if r['task']!='api_classification']+[dict(r,pattern=p['pattern']) for p in schema['patterns'] for r in apirows]
    classrows.sort(key=lambda r:(r['fold'],r['pattern'],r['task'],str(r['layer']),r['classifier_id'],r['role']))
    support=collections.Counter(a for r in records for a in r['api_labels'] or [])
    rare=[]
    for fold in folds:
        rr=[r for r in classrows if r['fold']==fold and r['role']=='train']
        # API rows repeated for 16 configurations: do not sum them as independent models.
        baseapi=[r for r in rr if r['task']=='api_classification' and r['pattern']==schema['patterns'][0]['pattern']]
        byapi={role:[r['positive_services'] for r in apidist if r['fold']==fold and r['role']==role] for role in allrole}
        rare.append({'fold':fold,'full_development_single_positive_api_labels':sum(v==1 for v in support.values()),
          'train_zero_positive_api_labels':sum(v==0 for v in byapi['train']),
          'outer_train_zero_positive_api_labels':sum(v==0 for v in byapi['outer_train']),
          'validation_zero_positive_api_labels':sum(v==0 for v in byapi['validation']),
          'early_stopping_zero_positive_api_labels':sum(v==0 for v in byapi['early_stopping']),
          'train_single_positive_api_labels':sum(v==1 for v in byapi['train']),
          'shared_api_classifiers_train_empty':sum(r['training_unavailable'] for r in baseapi),
          'shared_api_classifiers_early_stopping_unavailable':sum(r['early_stopping_unavailable'] for r in baseapi),
          'category_classifiers_train_empty_all_patterns':sum(r['training_unavailable'] for r in rr if r['task']=='category_classification'),
          'category_classifiers_early_stopping_unavailable_all_patterns':sum(r['early_stopping_unavailable'] for r in rr if r['task']=='category_classification')})
    usage=[]
    for title,kind in [('mashup','mashup'),('candidate_api','candidate_api'),('noncandidate_api','noncandidate_api')]:
        rs=[r for r in records if ('mashup' if r['source_type']=='mashup' else 'candidate_api' if r['api_classification_eligible'] else 'noncandidate_api')==kind]
        original=src['observed_source_counts']['mashups']-len(ex['collision_names']) if kind=='mashup' else len(rs)
        excluded=sum(e['source_type']=='mashup' and e['present_as_entity_in_expanded_data'] for e in ex['records']) if kind=='mashup' else 0
        usage.append({'data_type':title,'original_expanded_entities':original,'excluded_original_entities':excluded,
          'restored_entities':sum(r['restored_mashup'] for r in rs),'final_services':len(rs)})
    usage.extend([{'data_type':name,'original_expanded_entities':sum(r['original_expanded_entities'] for r in usage[:end]),
      'excluded_original_entities':sum(r['excluded_original_entities'] for r in usage[:end]),'restored_entities':len(read(output/'source_records/restored_mashups.json')),
      'final_services':sum(r['final_services'] for r in usage[:end])} for name,end in [('category_classification',3),('api_classification_positive',2)]])
    # All original counts derive from measured inputs, not selection/padding targets.
    reports=output/'reports';reports.mkdir(exist_ok=False)
    for name,rows in [('data_usage_summary',usage),('fold_distribution',dist),('category_label_distribution',catdist),
      ('api_label_distribution',apidist),('classifier_sample_counts',classrows),('rare_classifier_summary',rare),
      ('unavailable_classifiers',[r for r in classrows if r['role']=='train' and (r['training_unavailable'] or r['early_stopping_unavailable'])])]:
        columns=list(classrows[0]) if name=='unavailable_classifiers' else None
        write_csv(reports/f'{name}.csv',rows,columns)
    write(output/'manifests/classifier_manifest.json',{'source_manifest':'manifests/source_manifest.json','generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'implementation_sha256':src['implementation_sha256'],'git_commit':src['git_commit'],'seed':42,'n_folds':5,'schema_version':1,'classifiers':definitions,'shared_api_classifiers':152,
      'all_patterns_share_assignment':'splits/fold_assignments.csv','negative_candidates':'count only, known development Mashups with no API in the branch category; no API-description negatives'})
    count_one=sum(len(v)==1 for v in apimembers.values());full_api_rows=sum(len(v) for v in apimembers.values())
    leakage=f'''# Data leakage audit\n\nEvaluation membership and all normalized evaluation descriptions were excluded before grouping/expansion.\n\n- Excluded fixed evaluation Mashup entities: {dataset['counts']['excluded_evaluation_entities']} (includes four collision Mashups not restored).\n- Additional text-only alias: {dataset['counts']['excluded_text_only']}; empty descriptions: {dataset['counts']['excluded_empty_description']}.\n- Retained distinct API entities sharing evaluation Mashup names: {len(ex['same_name_distinct_apis_retained'])}.\n- Restored non-evaluation collision Mashups: {dataset['counts']['restored_mashups']}.\n- Normalization: {NORMALIZATION}. Labels were not combined for identical text.\n- Fixed evaluation category GT differences: {canonical(ex['fixed_category_gt_warnings'])}. Evaluation files were retained.\n- Fold roles are shared across all sources, all layers and all 16 structures; no sample moves or negative records.\n\nFull automated integrity, source preservation and deterministic recomputation results: `../checks/validation_results.json`.\nThe fixed evaluation was used in historical experiments; this work does not erase that history.\n'''
    write(reports/'data_leakage_audit.md',leakage)
    lines=['# Phase 1/2 implementation report','',f"Category services: {len(records)}; API positives: {sum(r['api_classification_eligible'] for r in records)}; groups: {len(groups)}.",
      f'API branch positive rows: {full_api_rows}; single-service API classifiers: {count_one}; singleton API labels: {sum(v==1 for v in support.values())}.','',
      'No training, inference, checkpoint replacement, hierarchy rebuilding or negative-record generation was executed.',
      'CSV classifier rows repeat the same 152 API classifiers for 16 patterns for traceability; do not add them as distinct services.',
      'Training/ES unavailability and zero-positive labels are recorded; no fallback algorithm is implemented.',
      'Labels positive in every Train record may have zero pos_weight in legacy trainers. This is a Phase 3 warning, not repaired here.',
      'Reference checks are assertions only: reconstructed data are never adjusted to expected counts.',
      'Pre-existing files: declared input/code SHA-256 plus repository-wide file metadata checks; model contents are not hashed.',
      '', '## Fold service counts','', '| Fold | Outer Train | Gradient Train | Inner ES | Outer Validation |', '| --- | --- | --- | --- | --- |']
    for fold in folds:
        nums={r['role']:r['services'] for r in dist if r['fold']==fold}
        lines.append(f"| {fold} | {nums['outer_train']} | {nums['train']} | {nums['early_stopping']} | {nums['validation']} |")
    lines+=['','## Rare classifiers','', '```json',json.dumps(rare,ensure_ascii=False,indent=2),'```','',
      '## Completion checks','', 'See checks/validation_results.json. Runtime generation timestamps are intentionally variable; records, groups, fold assignments and CSV counts are deterministic.',
      'No original paper manuscript was located during input inventory; terminology and structure IDs follow the existing README, audit reports and inference code.']
    write(reports/'implementation_report.md','\n'.join(lines)+'\n')
    write(output/'README.md',dataset_readme(dataset,groups,full_api_rows))
    print(canonical({'reports':'created','category_services':len(records),'api_branch_positive_rows':full_api_rows,'rare_classifier_summary':rare}))


def dataset_readme(dataset, groups, api_rows):
    return f'''# Hierarchical API Selection — Group 5-Fold v1 (Phase 1/2)

Mashup requirements -> 909 APIs; 152 fixed categories, 16 existing Balanced K-means structures. No hierarchy rebuilding, training or inference.

## Data usage

Category services: {dataset['counts']['category_services']}; API positive subset: {dataset['counts']['api_positive_services']}; groups: {len(groups)}; API branch-positive rows: {api_rows}.
API positives are reused category services, not additional independent examples. Full counts: reports/data_usage_summary.csv.
61 overwritten non-evaluation Mashups are restored from immutable sources. 336 evaluation Mashup entities, text aliases and empty text are excluded before branch expansion; four distinct same-name APIs are retained. Fixed selection Validation 100 and final Test 236 are untouched, outside every fold.

## Files and schema

- manifests/source_manifest.json: relative input paths, SHA-256, source counts, code hashes, Git commit, UTC time and conditions.
- manifests/existing_file_snapshot.json: all pre-existing file metadata, including models/results. SHA-256 covers declared inputs/code; model contents are not hashed.
- manifests/dataset_manifest.json: counts and reusable dataset paths; exclusion_manifest.json: entity-specific reasons, retained API homonyms, evaluation identity/text fingerprints.
- manifests/label_schema.json: global 909 API/152 category order, API-category mapping, fixed category_paths, existing model/inference correspondence for 16 patterns.
- manifests/classifier_manifest.json: per-parent outputs and membership rules. All API classifiers are shared and use fixed catalog subsets, never fold-observed output lists.
- source_records/unified_services.json and category_classification.json: identical lists of accepted services.
- source_records/api_classification_positive.json: exact eligible subset; restored_mashups.json: 61 recovered services with original provenance; groups.json: group members and input hashes.
- splits/group_assignments.csv: one row/group, validation_fold, entity list, source counts. fold_assignments.csv: one row/entity, common group and outer validation fold.
- splits/fold_01..fold_05/: outer_train_groups.json is the outer development Train; train_groups.json excludes inner ES; early_stopping_groups.json and validation_groups.json are disjoint. Lists are group IDs, not copied training records.
- reports/fold_distribution.csv: source/task/service/group counts for each role; category_label_distribution.csv and api_label_distribution.csv: full label support including zeros.
- reports/classifier_sample_counts.csv: 16 patterns × layers/parent classifiers × folds/roles; API positive/known-Mashup negative candidate counts. No actual negatives are generated.
- reports/unavailable_classifiers.csv and rare_classifier_summary.csv: Train/ES unavailability, singleton/zero-support labels. Fallback learning is not implemented.
- reports/data_leakage_audit.md and implementation_report.md: evidence and limits. checks/validation_results.json: automated checks and generated-artifact hashes.

Every service has entity_id/source_type/source_id/source_key/source_path/teacher_source_path/source_record_sha256, unchanged description, category_labels, original_source_categories, used_apis, api_labels, api_classification_eligible, api_category_pairs, normalized_text_sha256, restored_mashup and group_id.
Raw Mashup/API integer IDs are checked unique, namespaced as mashup:<id>/api:<id>. Source hashes bind those IDs to the source revision. used_apis preserves the source Mashup list; api_labels is its sorted set or the candidate API's own name. Noncandidate APIs have api_labels=null and eligible=false, never an all-zero negative target.
Categories preserve the original rule: Mashup -> first category of each used API; API -> its first category. Duplicate category labels are represented as a set; descriptions are never normalized in the model input.

## Grouping and split algorithm

Comparison text: {NORMALIZATION}. Identical nonempty comparison input links records across types, even with different labels. Confirmed source identities cannot be duplicated; records sharing text keep separate labels. group_id = group:SHA256(canonical sorted entity IDs). Group members are not deleted or merged.
Deterministic rarest-remaining-multilabel deficit allocation uses category/API/source supports, hard group-count capacities with largest-remainder rounding, and seed 42. It approximates label balance, does not guarantee impossible rare-label stratification. Input group IDs are sorted before seed-42 shuffling. Inner splits independently reuse seed 42 on each outer Train's group subset, fraction 0.8/0.2. No after-the-fact sample moves.
All 16 structures and API subset inherit the same outer and inner groups. No individual classifier can repartition its examples. A record may train multiple branches in the same role; it may never cross roles in one fold. Fixed category paths are reused verbatim.

## Reproduce

From repository root, using Python >=3.10 and the standard library only (no dependency install):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/prepare_training_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/create_group_folds.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/summarize_data_usage.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/validate_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1 --write-report
```

These are creation commands, not overwrite commands. To regenerate, choose a different nonexistent --output-root (for example a separate temporary directory); never delete/replace existing artifacts. Each phase stops on existing outputs. No --force flag exists. Timestamps/output locations differ; record content/group/fold assignment is deterministic. Default validation is read-only and can be repeated:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hierarchical_api_selection/validate_dataset.py --output-root data/hierarchical_api_selection/group_5fold_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/hierarchical_api_selection/tests -v
```

## Phase 3 handoff

Use group lists to select records; project category_labels through the fixed pattern's paths. For API category c, use only eligible records and target api_labels intersect that category's fixed catalog. Preserve entity/group/fold IDs. Negative candidates are known development Mashups not using that category, counted only here; generate actual negatives only in a future approved phase.
Train means gradient subset, ES means inner validation, Validation means outer CV evaluation. Fixed Validation100 selects hierarchy after CV; Test236 evaluates recommendation. Do not use either fixed set for ES, fitting or fold selection. Minority failures, train-positive-zero labels, and legacy zero-pos_weight cases require Phase 3 decisions; there is no automatic fix, moved data or learning fallback here.
'''


def load_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def validate(root, output, write_report=False):
    if write_report:require(not (output/'checks').exists(),'Validation output already exists; stop instead of overwriting')
    checks=[]
    def checked(name,condition,detail=None):
        require(condition,f'Validation failed: {name}: {detail}')
        checks.append({'check':name,'passed':True,'detail':detail})
    src=check_provenance(root,output);checked('all_declared_input_and_code_sha256_unchanged',True,len(src['source_files']))
    checked('all_preexisting_file_metadata_unchanged',True,src['existing_file_count'])
    records,groups,schema=phase_files(output);positive=read(output/'source_records/api_classification_positive.json')
    checked('entity_ids_unique',len({r['entity_id'] for r in records})==len(records))
    checked('required_source_identity_present',all(r['source_type'] in ['mashup','api'] and isinstance(r['source_id'],int) for r in records))
    checked('no_empty_description',all(normalize(r['description']) for r in records))
    checked('api_labels_in_fixed_catalog',all((r['api_labels'] is None and not r['api_classification_eligible']) or
       (r['api_classification_eligible'] and bool(r['api_labels']) and set(r['api_labels'])<=set(schema['api_candidates'])) for r in records))
    checked('category_labels_in_fixed_catalog',all(r['category_labels'] and set(r['category_labels'])<=set(schema['categories']) for r in records))
    checked('api_positive_exact_subset',positive==[r for r in records if r['api_classification_eligible']])
    checked('category_exact_unified_records',read(output/'source_records/category_classification.json')==records)
    # Full deterministic reconstruction verifies unchanged model inputs and labels against sources.
    d,expected,ep,eg,er,counts,es,ee=reconstruct(root)
    checked('source_text_labels_provenance_and_restoration_reconstruction',records==expected)
    checked('restored_entities_exact',read(output/'source_records/restored_mashups.json')==er,len(er))
    checked('normalization_groups_exact',groups==eg)
    checked('label_and_hierarchy_schema_exact',all(schema[k]==v for k,v in es.items()))
    stored_ex=read(output/'manifests/exclusion_manifest.json')
    checked('exclusion_evidence_exact',all(stored_ex[k]==v for k,v in ee.items()))
    checked('dataset_counts_exact',read(output/'manifests/dataset_manifest.json')['counts']==counts,counts)
    checked('fixed_evaluation_entity_and_text_overlap_zero',not ({r['entity_id'] for r in records}&set(ee['fixed_evaluation_entity_ids'])) and
      not ({r['normalized_text_sha256'] for r in records}&set(ee['fixed_evaluation_text_sha256'])))
    checked('all_16_fixed_hierarchies_and_909_152_spaces',len(schema['patterns'])==16 and len(schema['categories'])==152 and len(schema['api_candidates'])==909)
    for p in schema['patterns']:checked('hierarchy_category_coverage/'+p['pattern'],set(p['category_paths'])==set(schema['categories']))
    folds=read_folds(output);outer,planned=plan_folds(records,groups)
    checked('seed42_repeated_fold_assignment_identical',folds==planned)
    # Reverse input order: tie-breaking cannot depend on input JSON iteration order.
    ro,rf=plan_folds(list(reversed(records)),list(reversed(groups)))
    checked('fold_assignment_independent_of_input_order',ro==outer and rf==planned)
    allgroups={g['group_id'] for g in groups};seen=collections.Counter()
    for fold,roles in folds.items():
        role_sets={k:set(v) for k,v in roles.items()}
        checked('unique_group_lists/'+fold,all(len(v)==len(role_sets[k]) for k,v in roles.items()))
        train,eset,val,ot=(role_sets[k] for k in ['train_groups','early_stopping_groups','validation_groups','outer_train_groups'])
        checked('disjoint_inner_and_outer_roles/'+fold,not train&eset and not train&val and not eset&val and train|eset==ot and ot|val==allgroups)
        seen.update(val)
    checked('each_group_is_outer_validation_exactly_once',set(seen)==allgroups and all(n==1 for n in seen.values()))
    ga=load_csv(output/'splits/group_assignments.csv');fa=load_csv(output/'splits/fold_assignments.csv')
    checked('group_assignment_csv_unique_complete',len(ga)==len(groups) and {r['group_id'] for r in ga}==allgroups)
    gm={r['group_id']:r for r in ga}
    checked('group_assignment_csv_content',all(gm[g['group_id']]['validation_fold']==f'fold_{outer[g["group_id"]]+1:02d}' and
       json.loads(gm[g['group_id']]['entity_ids'])==g['entity_ids'] and int(gm[g['group_id']]['service_count'])==len(g['entity_ids']) for g in groups))
    fm={r['entity_id']:r for r in fa}
    checked('entity_fold_csv_exact',len(fa)==len(records) and len(fm)==len(records) and all(r['entity_id'] in fm and
      fm[r['entity_id']]['group_id']==r['group_id'] and fm[r['entity_id']]['validation_fold']==f'fold_{outer[r["group_id"]]+1:02d}' for r in records))
    definitions=classifier_definitions(schema)
    checked('all_classifiers_share_fixed_schema_and_assignment',read(output/'manifests/classifier_manifest.json')['classifiers']==definitions)
    rm=roles_by_entity(records,folds);distribution=load_csv(output/'reports/fold_distribution.csv')
    checked('fold_distribution_complete',len(distribution)==20 and len({(r['fold'],r['role']) for r in distribution})==20)
    for row in distribution:
        roles=rm[row['fold']];rs=[r for r in records if roles[r['entity_id']]==row['role'] or row['role']=='outer_train' and roles[r['entity_id']]!='validation']
        checked('service_and_group_counts/'+row['fold']+'/'+row['role'],int(row['services'])==len(rs) and int(row['groups'])==len({r['group_id'] for r in rs}) and
          int(row['api_positive_services'])==sum(r['api_classification_eligible'] for r in rs))
    cat_rows=load_csv(output/'reports/category_label_distribution.csv');api_rows=load_csv(output/'reports/api_label_distribution.csv')
    for rows,field,labels in [(cat_rows,'category_label',schema['categories']),(api_rows,'api_label',schema['api_candidates'])]:
        checked(field+'_distribution_complete',len(rows)==20*len(labels) and len({(r['fold'],r['role'],r[field]) for r in rows})==len(rows))
        actual={(r['fold'],r['role'],r[field]):int(r['positive_services']) for r in rows}
        for fold,roles in rm.items():
            for role in ['train','early_stopping','validation','outer_train']:
                rs=[r for r in records if roles[r['entity_id']]==role or role=='outer_train' and roles[r['entity_id']]!='validation']
                support=collections.Counter(z for r in rs for z in (r['category_labels'] if field=='category_label' else r['api_labels'] or []))
                checked(field+'_supports/'+fold+'/'+role,all(actual[(fold,role,z)]==support[z] for z in labels))
    # Verify all classifier supports with an independent parent-path traversal and role counters.
    expected_stats={}
    for p in schema['patterns']:
        for r in records:
            parents=collections.defaultdict(set)
            for c in r['category_labels']:
                path=p['category_paths'][c]
                for depth in range(p['category_layers']):
                    cid='category/layer_'+str(depth+1)+'/'+('-'.join(map(str,path[:depth])) or 'root')
                    parents[cid].add(c if depth==p['category_layers']-1 else '-'.join(map(str,path[:depth+1])))
            for fold,roles in rm.items():
                rr=roles[r['entity_id']]
                for role in [rr]+(['outer_train'] if rr!='validation' else []):
                    for cid,ls in parents.items():
                        key=(fold,p['pattern'],cid,role)
                        st=expected_stats.setdefault(key,{'n':0,'m':0,'groups':set(),'labels':collections.Counter()})
                        st['n']+=1;st['m']+=r['source_type']=='mashup';st['groups'].add(r['group_id']);st['labels'].update(ls)
    # API classification has the same memberships across patterns, computed once.
    for r in positive:
        cats=collections.defaultdict(set)
        for a in r['api_labels']:cats[schema['api_categories'][a]].add(a)
        for fold,roles in rm.items():
            rr=roles[r['entity_id']]
            for role in [rr]+(['outer_train'] if rr!='validation' else []):
                for cat,ls in cats.items():
                    key=(fold,'shared_all_16','api/'+cat,role)
                    st=expected_stats.setdefault(key,{'n':0,'m':0,'groups':set(),'labels':collections.Counter()})
                    st['n']+=1;st['m']+=r['source_type']=='mashup';st['groups'].add(r['group_id']);st['labels'].update(ls)
    defmap={(de['pattern'],de['classifier_id']):de for de in definitions}
    rows=load_csv(output/'reports/classifier_sample_counts.csv');numcategory=sum(de['task']=='category_classification' for de in definitions)
    checked('complete_16_pattern_classifier_report',len(rows)==20*(numcategory+16*152) and len({(r['fold'],r['pattern'],r['classifier_id'],r['role']) for r in rows})==len(rows))
    row_errors=[]
    for row in rows:
        pat='shared_all_16' if row['task']=='api_classification' else row['pattern'];key=(row['fold'],pat,row['classifier_id'],row['role'])
        st=expected_stats.get(key,{'n':0,'m':0,'groups':set(),'labels':collections.Counter()});de=defmap[(pat,row['classifier_id'])];ls=de['output_labels']
        numeric={'positive_services':st['n'],'mashup_services':st['m'],'api_services':st['n']-st['m'],'groups':len(st['groups']),
          'output_label_count':len(ls),'zero_positive_labels':sum(st['labels'][z]==0 for z in ls),
          'one_positive_labels':sum(st['labels'][z]==1 for z in ls),'all_positive_labels':sum(st['n']>0 and st['labels'][z]==st['n'] for z in ls)}
        tr=expected_stats.get((row['fold'],pat,row['classifier_id'],'train'),{'n':0})
        esst=expected_stats.get((row['fold'],pat,row['classifier_id'],'early_stopping'),{'n':0})
        vst=expected_stats.get((row['fold'],pat,row['classifier_id'],'validation'),{'n':0})
        numeric.update(training_unavailable=int(tr['n']==0),early_stopping_unavailable=int(tr['n']==0 or esst['n']==0),outer_validation_unavailable=int(vst['n']==0))
        if row['task']=='api_classification':
            totalm=next(int(z['mashup_services']) for z in distribution if z['fold']==row['fold'] and z['role']==row['role'])
            numeric['known_mashup_negative_candidates']=totalm-st['m']
        if any(int(row[k])!=v for k,v in numeric.items()):row_errors.append(key)
    checked('all_classifier_and_negative_candidate_counts',not row_errors,row_errors[:10])
    unavail=load_csv(output/'reports/unavailable_classifiers.csv')
    checked('unavailable_classifier_list_exact',unavail==[r for r in rows if r['role']=='train' and (int(r['training_unavailable']) or int(r['early_stopping_unavailable']))])
    # If a validation report exists, also verify its artifact hashes on read-only repeat runs.
    report_path=output/'checks/validation_results.json'
    if report_path.exists():
        prior=read(report_path)
        checked('previous_generated_artifact_hashes',all(sha(output/rel)==h for rel,h in prior['generated_artifact_sha256'].items()))
    result={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
      'counts':counts,'checks':checks,'warnings':[
        'Fixed category GT differs from original Mashup-derived categories for trackthis and rosetta; fixed data are unchanged.',
        'Minority/no-positive/ES-empty cases are recorded, not repaired. No learning fallback was implemented.',
        'Legacy pos_weight may be zero for all-positive labels; trainer changes belong to Phase 3.',
        'Existing model contents were not hashed; all pre-existing file metadata and declared-input/code SHA-256 were verified.',
        'Similar/paraphrased text and unknown aliases are outside exact normalized matching guarantees.'],
      'generated_artifact_sha256':{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and 'checks' not in p.relative_to(output).parts}}
    if write_report:write(report_path,result)
    print(canonical({'validation':'passed','checks':len(checks),'counts':counts,'report_written':write_report}))
    return result


def main(command):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    if command=='validate':parser.add_argument('--write-report',action='store_true')
    args=parser.parse_args();root=args.repository_root.resolve();output=args.output_root
    if not output.is_absolute():output=root/output
    output=output.resolve()
    try:
        if command=='prepare':prepare(root,output)
        elif command=='folds':create_folds(root,output)
        elif command=='summarize':summarize(root,output)
        elif command=='validate':validate(root,output,args.write_report)
    except (ValueError,FileExistsError,FileNotFoundError,KeyError,TypeError) as error:
        parser.exit(1,f'STOP: {error}\nNo overwrite/recovery action was attempted. Inspect the error before continuing.\n')
