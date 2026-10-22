import json,sys,html,re,collections
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent;b=root/'results/baselines/test236_masked/tfidf_ovr_logistic'
def read(p):return json.loads(p.read_text())
def unpack(e):return (e.get('description',''),e.get('apis',[])) if isinstance(e,dict) else (e[1],e[2])
def norm(s):return ' '.join(re.findall(r'\w+',html.unescape(s).lower()))
train=read(root/'data/processed/root_json/study_data_mashup_apis.json');rs=read(b/'predictions.json');cfg=read(b/'config.json');test=read(Path(cfg['test_path']));unmask=read(Path(cfg['test_path'].replace('_api_name_masked.json','.json')))
tr=[(k,unpack(e)[0].strip(),unpack(e)[1]) for k,e in train.items() if unpack(e)[0].strip() and unpack(e)[1]]; names=[a for a,i in sorted(read(b/'labels.json')['api_to_idx'].items(),key=lambda x:x[1])]
idx={a:i for i,a in enumerate(names)};assert names==sorted({a for k,t,l in tr for a in l})
report={'train_count':len(tr),'test_count':len(rs),'name_overlap':sorted(set(train)&set(test)),'id_overlap':sorted({str(e.get('id',k) if isinstance(e,dict) else e[3]) for k,e in train.items()} & {str(e.get('id',k) if isinstance(e,dict) else e[3]) for k,e in test.items()})}
bytext=collections.defaultdict(list)
for k,t,l in tr:bytext[norm(t)].append(k)
report['exact_normalized_unmasked_text_overlap']=[{'test':k,'train':bytext[norm(unpack(e)[0])]} for k,e in unmask.items() if norm(unpack(e)[0]) in bytext]
v=joblib.load(b/'vectorizer.joblib');c=joblib.load(b/'model.joblib');xtr=v.transform([t for k,t,l in tr]);xt=v.transform([r['text'] for r in rs]);sim=cosine_similarity(xt,xtr);nn=sim.argmax(axis=1);maximum=sim[np.arange(len(rs)),nn]
report['similarity_counts']={str(t):int((maximum>=t).sum()) for t in [.8,.9,.95,.99]};report['nearest_pairs']=[{'test':list(test)[i],'train':tr[nn[i]][0],'cosine':float(maximum[i]),'test_f1':rs[i]['f1']} for i in np.argsort(-maximum)[:15]]
# Fit from raw train only, independently of the stored vectorizer/model.
params=dict(cfg['tfidf_params']);params['ngram_range']=tuple(params['ngram_range']);v2=TfidfVectorizer(**params);x2=v2.fit_transform([t for k,t,l in tr]);y=np.zeros((len(tr),len(names)),dtype=np.int8)
for i,(k,t,ls) in enumerate(tr):
 for a in ls:y[i,idx[a]]=1
c2=OneVsRestClassifier(LogisticRegression(C=1.,max_iter=1000,class_weight='balanced',solver='liblinear',random_state=42),n_jobs=1);c2.fit(x2,y);prob=c2.predict_proba(v2.transform([r['text'] for r in rs]));savedprob=c.predict_proba(xt)
report['independent_train_refit']={'vocabulary_matches':v2.vocabulary_==v.vocabulary_,'max_probability_difference':float(np.max(np.abs(prob-savedprob))),'all_236_prediction_sets_match':all(set(r['predicted_apis'])=={a for a,s in zip(names,p) if s>=.5} for r,p in zip(rs,prob))}
assert report['independent_train_refit']['all_236_prediction_sets_match']
tp=collections.Counter();pred=collections.Counter();true=collections.Counter()
for r in rs:
 p=set(r['predicted_apis']);t=set(r['true_apis']);tp.update(p&t);pred.update(p);true.update(t)
report['top_true_positive_apis']=[{'api':a,'tp':n,'predicted':pred[a],'true':true[a]} for a,n in tp.most_common(12)]
report['total_tp']=sum(tp.values());report['perfect_f1_records']=sum(r['f1']==1 for r in rs);report['zero_f1_records']=sum(r['f1']==0 for r in rs);report['mean_f1']=sum(r['f1'] for r in rs)/len(rs)
for title,fn in [('single_true_api',lambda r:len(set(r['true_apis']))==1),('multiple_true_apis',lambda r:len(set(r['true_apis']))>1),('high_similarity_ge_09',lambda r:maximum[r['idx']]>=.9),('similarity_lt_09',lambda r:maximum[r['idx']]<.9)]:
 selected=[r for r in rs if fn(r)];report[title]={'n':len(selected),'mean_f1':sum(r['f1'] for r in selected)/len(selected) if selected else None}
report['test_masked_text_count']=sum(unpack(test[k])[0]!=unpack(unmask[k])[0] for k in test)
(out/'tfidf_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
