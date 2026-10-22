"""Aggregate all saved hierarchy validation/test predictions, without inference."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/hierarchy/validation_test_comparison'
H=ROOT/'results/hierarchy/hierarchy_existing_models_api_name_masked'
def read(p):return json.loads(p.read_text())
def unpack(e):return (e['description'],e['apis']) if isinstance(e,dict) else (e[1],e[2])
def decode(v):return json.loads(v) if isinstance(v,str) else v
OUT.mkdir(parents=True,exist_ok=True)
rows=[];verified=0;scope_counts={};selected=read(H/'validation/selection.json')['selected_pattern']
for split,expected in [('validation',100),('test',236)]:
 canonical=read(ROOT/'data/splits/model_selection'/split/'layer1_direct_api/test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json')
 assert len(canonical)==expected
 tasks=read(H/split/'test_data_manifest.json');assert len(tasks)==16
 for task in tasks:
  dataset=read(Path(task['evaluation_data_path']))
  with Path(task['out_csv_path']).open(newline='') as f:records=list(csv.DictReader(f,escapechar='\\'))
  assert len(dataset)==len(records)==expected
  values=[]
  for i,((key,e),r) in enumerate(zip(dataset.items(),records)):
   text,true=unpack(e);ct,ca=unpack(canonical[key]);t=set(decode(r['true_apis']));p=set(decode(r['pred_apis']))
   assert int(r['idx'])==i and r['text']==text==ct and t==set(true)==set(ca)
   tp=len(t&p);fp=len(p-t);fn=len(t-p);pr=tp/len(p) if p else 0;rec=tp/len(t) if t else 0;f1=2*tp/(len(t)+len(p)) if t or p else 0
   for k,v in [('precision',pr),('recall',rec),('f1',f1)]:assert math.isclose(float(r[k]),v,abs_tol=1e-12),(split,task['name'],i,k)
   values.append(dict(key=key,t=t,p=p,tp=tp,fp=fp,fn=fn,precision=pr,recall=rec,f1=f1));verified+=1
  for scope in ['通常（全件）','正解API数2以上']:
   subset=values if scope=='通常（全件）' else [v for v in values if len(v['t'])>=2]
   n=len(subset);assert n>0
   keys=[v['key'] for v in subset];group=(split,scope)
   if group in scope_counts:assert scope_counts[group]==set(keys)
   else:scope_counts[group]=set(keys)
   tp=sum(v['tp'] for v in subset);fp=sum(v['fp'] for v in subset);fn=sum(v['fn'] for v in subset)
   row={'分割':split,'評価対象':scope,'件数':n,'構造':task['name'],'層数':task['layer'],'validation選定構造':task['name']==selected,'Precision':math.fsum(v['precision'] for v in subset)/n,'Recall':math.fsum(v['recall'] for v in subset)/n,'F1':math.fsum(v['f1'] for v in subset)/n,'micro_Precision':tp/(tp+fp) if tp+fp else 0,'micro_Recall':tp/(tp+fn) if tp+fn else 0,'micro_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'平均推薦API数':sum(len(v['p']) for v in subset)/n,'推薦なし件数':sum(not v['p'] for v in subset),'正解API2件以上推薦件数':sum(v['tp']>=2 for v in subset),'全正解API網羅件数':sum(v['t']<=v['p'] for v in subset),'正解集合完全一致件数':sum(v['t']==v['p'] for v in subset),'正解API検出0件数':sum(v['tp']==0 for v in subset),'TP':tp,'FP':fp,'FN':fn}
   for count,rate in [('正解API2件以上推薦件数','正解API2件以上推薦率'),('全正解API網羅件数','全正解API網羅率'),('正解集合完全一致件数','正解集合完全一致率')]:row[rate]=row[count]/n
   rows.append(row)
# Display and export each subset by its F1; test rank is descriptive, not model selection.
rows.sort(key=lambda r:({'validation':0,'test':1}[r['分割']],{'通常（全件）':0,'正解API数2以上':1}[r['評価対象']],-r['F1'],r['構造']))
def export(filename,rs):
 with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 with (OUT/filename).open(encoding='utf-8-sig',newline='') as f:back=list(csv.DictReader(f))
 assert len(back)==len(rs)
 for a,b in zip(rs,back):
  for k,v in a.items():
   if isinstance(v,bool):assert str(v)==b[k]
   elif isinstance(v,(int,float)):assert math.isclose(v,float(b[k]),abs_tol=1e-12)
   else:assert v==b[k]
export('hierarchy_validation_test_combined.csv',rows)
for split in ['validation','test']:
 rs=[r for r in rows if r['分割']==split];export(f'hierarchy_{split}_comparison.csv',rs)
 basic=['構造','Precision','Recall','F1','micro_F1','平均推薦API数'];tsv=[];md=['# '+split+' 階層構造別結果','', 'API名マスクあり。P/R/F1はサンプル平均。推薦なしは0。保存結果を再集計し、分岐スキップはそのまま扱う。','', '★はvalidation全100件で選定した12-4-3。各表は表示対象のF1順であり、testで構造を選び直していない。','']
 for scope in ['通常（全件）','正解API数2以上']:
  selected_rows=[r for r in rs if r['評価対象']==scope];n=selected_rows[0]['件数'];title=f'{scope}（{n}件）'
  table=[[('★ ' if r['validation選定構造'] else '')+r['構造'],*(f'{r[k]:.4f}' for k in ['Precision','Recall','F1','micro_F1']),f'{r["平均推薦API数"]:.2f}'] for r in selected_rows]
  md+=['## '+title,'','| '+' | '.join(basic)+' |','|---|---:|---:|---:|---:|---:|']+['| '+' | '.join(x)+' |' for x in table]+['']
  tsv.append(title+'\n'+'\n'.join('\t'.join(x) for x in [basic]+table))
  cols=['構造','正解API2件以上推薦','全正解API網羅','正解集合完全一致'];counttable=[[('★ ' if r['validation選定構造'] else '')+r['構造'],*(f'{r[k]}件（{r[k]/n*100:.1f}%）' for k in ['正解API2件以上推薦件数','全正解API網羅件数','正解集合完全一致件数'])] for r in selected_rows]
  md+=['## '+title+'：検出・網羅','','| '+' | '.join(cols)+' |','|---|---:|---:|---:|']+['| '+' | '.join(x)+' |' for x in counttable]+['']
  tsv.append(title+'：検出・網羅\n'+'\n'.join('\t'.join(x) for x in [cols]+counttable))
 md+=['全正解API網羅は余分な推薦を含んでも可。完全一致は余分な推薦なし。通常全件の「2件以上推薦率」は正解APIが1件のサンプルも分母に含む。','validationのbaseline結果は保存されていないため、validation表は階層16構造のみ。']
 (OUT/f'{split}_powerpoint_tables.md').write_text('\n'.join(md)+'\n')
 (OUT/f'{split}_powerpoint_tables.tsv').write_text('\n\n'.join(tsv)+'\n',encoding='utf-8-sig')
selection=read(H/'validation/selection.json');winner=next(r for r in rows if r['分割']=='validation' and r['評価対象']=='通常（全件）');assert winner['構造']==selected and math.isclose(winner['F1'],selection['sample_mean_f1'],abs_tol=1e-12)
previous=read(ROOT/'analysis/baselines/validation_selected_test236_masked/multi_api_all_baselines.json')['results']['hierarchy'];chosen=next(r for r in rows if r['分割']=='test' and r['評価対象']=='正解API数2以上' and r['構造']==selected)
assert math.isclose(chosen['F1'],previous['mean_f1'],abs_tol=1e-12)
report={'validated_prediction_rows':verified,'export_rows':len(rows),'counts':{f'{s}/{scope}':len(keys) for (s,scope),keys in scope_counts.items()},'validation_winner_matches_selection':True,'selected_test_multi_api_matches_previous':True,'saved_csv_roundtrip_verified':True}
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2));print('Output:',OUT)
for split in ['validation','test']:
 for scope in ['通常（全件）','正解API数2以上']:
  print(split,scope,'top3',[(r['構造'],round(r['F1'],6)) for r in rows if r['分割']==split and r['評価対象']==scope][:3])
