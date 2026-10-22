"""Local pretrained BERT training and configuration selection. No legacy imports with side effects."""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
sys.dont_write_bytecode=True
from pipeline import ROOT, OUTPUT, PHASE_ONE, PHASE_TWO, POLICY, read, write, require, safe_weights, sha


def choose_config(results, expected_fold_ids, task='api_classification'):
    """Reject incomplete/nonfinite folds; LR loss-first within ratio, ratio F1-first."""
    grouped={}
    for r in results:
        require(r['fold_id'] in expected_fold_ids,'Unexpected fold')
        require(r['learning_rate'] in POLICY['learning_rates'],'Unexpected learning rate')
        require(1<=r['best_epoch']<=50,'Invalid epoch')
        require(all(math.isfinite(r[k]) for k in ['validation_loss','precision','recall','micro_f1']),'Nonfinite result')
        grouped.setdefault((r['ratio'],r['learning_rate']),[]).append(r)
    candidates=[]
    expected_ratios=sorted({r['ratio'] for r in results})
    require(expected_ratios,'No selection results')
    for ratio in expected_ratios:
        lrs=[]
        for lr in POLICY['learning_rates']:
            rows=grouped.get((ratio,lr),[])
            require(sorted(r['fold_id'] for r in rows)==sorted(expected_fold_ids),'Missing or duplicated folds')
            avg={k:statistics.mean(r[k] for r in rows) for k in ['validation_loss','precision','recall','micro_f1']}
            lrs.append({'ratio':ratio,'learning_rate':lr,'mean_metrics':avg,
              'best_epochs':[r['best_epoch'] for r in rows]})
        lrs.sort(key=lambda r:(r['mean_metrics']['validation_loss'],-r['mean_metrics']['micro_f1'],r['learning_rate']))
        candidates.append(lrs[0])
    if task=='api_classification':
        candidates.sort(key=lambda r:(-r['mean_metrics']['micro_f1'],-r['mean_metrics']['recall'],-r['mean_metrics']['precision'],r['ratio']))
    else:
        candidates.sort(key=lambda r:(r['mean_metrics']['validation_loss'],-r['mean_metrics']['micro_f1'],r['learning_rate']))
    best=candidates[0]
    return {**best,'epochs':min(50,max(1,math.floor(statistics.median(best['best_epochs'])+.5))),
      'selection_origin':'local development validation; not fixed validation100/test236','per_ratio_best':candidates}


def imports():
    import torch
    from transformers import BertModel, BertTokenizer
    return torch,BertModel,BertTokenizer


def model_class():
    torch,BertModel,_=imports()
    class BertForMultilabelClassification(torch.nn.Module):
        def __init__(self,model_name,num_labels):
            super().__init__()
            self.bert=BertModel.from_pretrained(model_name,local_files_only=True)
            self.classifier=torch.nn.Linear(self.bert.config.hidden_size,num_labels)
        def forward(self,input_ids,attention_mask):
            return self.classifier(self.bert(input_ids=input_ids,attention_mask=attention_mask).pooler_output)
    return BertForMultilabelClassification


def seed_everything(seed=42):
    torch,_,_=imports()
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    # User enabling CUDA must set CUBLAS_WORKSPACE_CONFIG before starting the process.


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def legacy_inference_class():
    """Execute only existing class AST, never its module-level evaluation loop."""
    import ast
    import torch
    from transformers import BertModel
    path=ROOT/'scripts/inference/mashup_dis_nosp_category_api.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BertForMultilabelClassification')
    # Existing Lightning class has an ordinary nn.Module-compatible __init__/forward.
    import pytorch_lightning as pl
    ns={'torch':torch,'nn':torch.nn,'pl':pl,'BertModel':BertModel}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[node.name]


def encode(ids,lookup,targets,labels,tokenizer,max_length):
    torch,_,_=imports()
    enc=tokenizer([lookup[e]['description'] for e in ids],padding='max_length',truncation=True,
      max_length=max_length,return_tensors='pt',return_token_type_ids=False)
    y=torch.tensor([[float(z in targets[e]) for z in labels] for e in ids])
    return torch.utils.data.TensorDataset(enc['input_ids'],enc['attention_mask'],y)


def make_loss(ids,targets,labels,device):
    torch,_,_=imports()
    weights=safe_weights([targets[x] for x in ids],labels)
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w['pos_weight'] for w in weights],device=device)),weights


def metrics_from_counts(tp,fp,fn):
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return {'precision':precision,'recall':recall,'micro_f1':2*precision*recall/(precision+recall) if precision+recall else 0.0,
      'tp':tp,'fp':fp,'fn':fn}


def evaluate(model,loader,loss_fn,device):
    torch,_,_=imports();model.eval();total=0.;n=0;tp=fp=fn=0
    with torch.no_grad():
        for a,b,y in loader:
            a,b,y=a.to(device),b.to(device),y.to(device)
            logits=model(a,b);loss=loss_fn(logits,y);require(torch.isfinite(loss).item(),'Nonfinite validation loss')
            total+=float(loss)*len(y);n+=len(y)
            pred=torch.sigmoid(logits)>=.5;true=y.bool()
            tp+=int((pred&true).sum());fp+=int((pred&~true).sum());fn+=int((~pred&true).sum())
    require(n>0,'Empty validation')
    return {'validation_loss':total/n,**metrics_from_counts(tp,fp,fn),'services':n}


def train_one(model_name, labels, train_ids, val_ids, lookup, targets, lr, epochs,
              device='cpu',max_length=512,batch_size=8,early_stop=True):
    torch,_,Tokenizer=imports();seed_everything()
    tokenizer=Tokenizer.from_pretrained(model_name,local_files_only=True)
    model=model_class()(model_name,len(labels)).to(device)
    initial_hash=state_hash(model)
    ds=encode(train_ids,lookup,targets,labels,tokenizer,max_length)
    generator=torch.Generator().manual_seed(42)
    loader=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0)
    vl=None
    if val_ids:
        vl=torch.utils.data.DataLoader(encode(val_ids,lookup,targets,labels,tokenizer,max_length),batch_size=batch_size,num_workers=0)
    loss_fn,weights=make_loss(train_ids,targets,labels,device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0,eps=1e-8)
    best=None;best_value=math.inf;best_epoch=None;stale=0;history=[];best_hash=None;best_metrics=None
    for epoch in range(1,epochs+1):
        model.train();running=0.;n=0
        for a,b,y in loader:
            a,b,y=a.to(device),b.to(device),y.to(device)
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(a,b),y)
            require(torch.isfinite(loss).item(),'Nonfinite training loss')
            loss.backward();opt.step();running+=float(loss.detach())*len(y);n+=len(y)
        row={'epoch':epoch,'train_loss':running/n}
        if vl is not None:
            row.update(evaluate(model,vl,loss_fn,device))
            if row['validation_loss']<best_value:
                best_value=row['validation_loss'];best_epoch=epoch;stale=0
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_hash=state_hash(model)
                best_metrics={k:row[k] for k in ['validation_loss','precision','recall','micro_f1']}
            else:stale+=1
        history.append(row)
        print(json.dumps({'epoch':epoch,'lr':lr,'train_loss':row['train_loss'],'val_loss':row.get('validation_loss')}),flush=True)
        if early_stop and vl is not None and stale>=POLICY['early_stopping_patience']:break
    restored=False
    if best is not None:
        model.load_state_dict(best);restored=state_hash(model)==best_hash
        require(restored,'Best epoch restoration failed')
    return model,{'history':history,'best_epoch':best_epoch,'best_metrics':best_metrics,
      'best_weight_sha256':best_hash,'best_weights_restored':restored,'initial_model_sha256':initial_hash,
      'train_label_weights':weights,'train_services':len(train_ids),'validation_services':len(val_ids)}


def verify_inputs(dataset=OUTPUT):
    source=read(dataset/'manifests/source_manifest.json')
    for p,h in {**source['input_sha256'],**source['implementation_sha256']}.items():require(sha(ROOT/p)==h,'Protected input/code changed: '+p)
    checkpath=dataset/'checks/validation_results.json'
    if not checkpath.exists():checkpath=dataset/'checks/data_validation_results.json'
    checks=read(checkpath);require(checks['passed'],'Unvalidated dataset')
    for p,h in checks['generated_artifact_sha256'].items():require(sha(dataset/p)==h,'Dataset artifact changed: '+p)


def load_classifier(uid,dataset=OUTPUT):
    records=read(PHASE_ONE/'source_records/unified_services.json');lookup={r['entity_id']:r for r in records}
    if uid.startswith('api_classification/'):
        entries=read(dataset/'manifests/classifier_manifest.json')['classifiers']
        entry=next(e for e in entries if e['classifier_uid']==uid)
        obj=read(dataset/entry['path']);ratios={int(r):read(dataset/d['path']) for r,d in entry['ratios'].items()}
    else:
        obj=read(PHASE_TWO/('splits/'+uid+'.json'));ratios={0:{'folds':[
          {'fold_id':f['fold_id'],'train_entity_ids':f['train_entity_ids'],'validation_entity_ids':f['validation_entity_ids']}
          for f in obj['partition']['folds']],
          'final_training':{'train_entity_ids':[r['entity_id'] for r in obj['members']]}}}
    return obj,ratios,lookup,{r['entity_id']:r['target_labels'] for r in obj['members']}


def bounded_subset(ids,targets,limit):
    """Smoke-only: keep a positive and a negative when present, report exact IDs."""
    if len(ids)<=limit:return ids
    pos=[e for e in ids if targets[e]];neg=[e for e in ids if not targets[e]]
    selected=(pos[:1]+neg[:1]);selected+= [e for e in ids if e not in selected][:max(0,limit-len(selected))]
    return sorted(selected)


def run(uid,out,dataset=OUTPUT,model_name='bert-base-uncased',device='cpu',smoke=False,ratios_requested=None):
    torch,_,_=imports();torch.set_num_threads(2)
    out=Path(out);require(not out.exists(),f'Refuse existing training output: {out}')
    verify_inputs(dataset)
    # Verify local pretrained origin even when no-validation fallback is used. No checkpoint argument exists.
    from huggingface_hub import snapshot_download
    cached=Path(snapshot_download('bert-base-uncased',local_files_only=True)).resolve()
    require(model_name=='bert-base-uncased' or Path(model_name).resolve()==cached,
      'Only existing cached bert-base-uncased pretrained initialization is allowed; classifier checkpoints prohibited')
    model_path=cached
    require(model_path.exists(),'Local pretrained BERT missing; no download permitted')
    obj,ratios,lookup,targets=load_classifier(uid,dataset);labels=obj['output_labels']
    if ratios_requested is not None:
        require(uid.startswith('api_classification/'),'--ratio applies only to API classification')
        ratios={r:ratios[r] for r in ratios_requested}
    require(bool(ratios),'No ratios requested')
    if not smoke and uid.startswith('api_classification/'):require(set(ratios)=={1,3,5},'Production selection must compare all three ratios')
    out.mkdir(parents=True,exist_ok=False)
    manifest={'classifier_uid':uid,'task':obj['task'],'smoke_only':smoke,'production_training':not smoke,
      'model_name':model_name,'local_pretrained_path':str(model_path),'pretrained_sha256':{
      p.name:sha(p) for p in sorted(model_path.iterdir()) if p.is_file() and p.suffix in ['.json','.txt','.safetensors','.bin']},
      'policy':POLICY,'output_labels':labels,'device':device,'maximum_length':32 if smoke else 512,
      'batch_size':2 if smoke else 8,'maximum_epochs':2 if smoke else 50,
      'selection_complete':not smoke,'fixed_evaluation_used':False,'training_encoder':True}
    write(out/'run_manifest.json',manifest)
    results=[];runs=[];selection_enabled=obj['partition'].get('selection_enabled',True)
    folds=next(iter(ratios.values()))['folds']
    expected=[f['fold_id'] for f in (folds[:1] if smoke else folds)]
    if selection_enabled:
        for ratio,rs in sorted(ratios.items()):
            for f in rs['folds'][:1] if smoke else rs['folds']:
                tr=f['train_entity_ids'];va=f['validation_entity_ids']
                if smoke:tr=bounded_subset(tr,targets,4);va=bounded_subset(va,targets,4)
                for lr in POLICY['learning_rates']:
                    model,info=train_one(str(model_path),labels,tr,va,lookup,targets,lr,2 if smoke else 50,
                      device=device,max_length=32 if smoke else 512,batch_size=2 if smoke else 8)
                    row={'ratio':ratio,'learning_rate':lr,'fold_id':f['fold_id'],'best_epoch':info['best_epoch'],**info['best_metrics']}
                    checkpoint=out/f'best_ratio_{ratio}_{f["fold_id"]}_lr_{lr:.0e}.pth'
                    if not smoke or lr==POLICY['learning_rates'][0]:
                        with checkpoint.open('xb') as stream:torch.save(model.state_dict(),stream)
                        info['best_checkpoint_path']=checkpoint.name
                    else:info['best_checkpoint_path']=None;info['best_checkpoint_storage']='second-LR smoke checkpoint in memory; SHA-256 and restoration recorded'
                    results.append(row);runs.append({**row,**info,'train_entity_ids':tr,'validation_entity_ids':va})
                    del model
        selected=choose_config(results,expected,obj['task'])
        require(len({r['initial_model_sha256'] for r in runs})==1,'Initialization differs across conditions')
    else:
        require(1 in ratios,'Fallback uses prespecified ratio 1; do not optimize on negative-only validation')
        selected={'ratio':POLICY['fallback_config']['negative_ratio'],
          'learning_rate':POLICY['fallback_config']['learning_rate'],'epochs':POLICY['fallback_config']['epochs'],
          'selection_origin':POLICY['fallback_config']['origin'],'reason':obj['partition']['reason']}
    chosen=ratios[selected['ratio']]['final_training']['train_entity_ids']
    if smoke:chosen=bounded_subset(chosen,targets,4)
    final,info=train_one(str(model_path),labels,chosen,[],lookup,targets,selected['learning_rate'],
      min(selected['epochs'],2) if smoke else selected['epochs'],device=device,max_length=32 if smoke else 512,
      batch_size=2 if smoke else 8,early_stop=False)
    final=final.cpu().eval()
    # A bare bert.* / classifier.* state_dict plus ordered JSON labels matches legacy inference.
    with (out/'model_weights.pth').open('xb') as f:torch.save(final.state_dict(),f)
    write(out/('api_labels.json' if obj['task']=='api_classification' else 'category_labels.json'),labels)
    reload_model=model_class()(str(model_path),len(labels))
    reload_model.load_state_dict(torch.load(out/'model_weights.pth',map_location='cpu',weights_only=True),strict=True)
    reload_model.eval()
    _,_,Tokenizer=imports();tokenizer=Tokenizer.from_pretrained(str(model_path),local_files_only=True)
    enc=tokenizer(lookup[chosen[0]]['description'],return_tensors='pt',max_length=32,truncation=True,return_token_type_ids=False)
    with torch.no_grad():before=final(enc['input_ids'],enc['attention_mask']);after=reload_model(enc['input_ids'],enc['attention_mask'])
    require(torch.equal(before,after),'Saved/reloaded predictions differ')
    del reload_model
    legacy=legacy_inference_class()(str(model_path),len(labels));legacy.load_state_dict(torch.load(out/'model_weights.pth',map_location='cpu',weights_only=True),strict=True);legacy.eval()
    with torch.no_grad():legacy_logits=legacy(enc['input_ids'],enc['attention_mask'])
    require(torch.equal(before,legacy_logits),'Legacy inference class prediction mismatch')
    del legacy
    report={'classifier_uid':uid,'smoke_only':smoke,'complete_production_selection':not smoke,
      'selection_performed':selection_enabled,'folds_tested':expected if selection_enabled else [],
      'results':results,'runs':runs,'selected':selected,'final_training':{**info,'train_entity_ids':chosen,
      'actual_epochs':len(info['history']),'fresh_initialization':True,'uses_previous_fold_weights':False},
      'roundtrip_predictions_equal':True,'legacy_inference_strict_load_and_logits_equal':True,
      'single_output_sigmoid_shape':list(before.shape),'fixed_evaluation_used':False,
      'model_sha256':sha(out/'model_weights.pth')}
    write(out/'training_results.json',report)
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--classifier',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,default=OUTPUT);ap.add_argument('--model',default='bert-base-uncased')
    ap.add_argument('--device',default='cpu');ap.add_argument('--smoke',action='store_true');ap.add_argument('--ratio',type=int,choices=[1,3,5])
    a=ap.parse_args();run(a.classifier,a.output,a.dataset,a.model,a.device,a.smoke,[a.ratio] if a.ratio else None)
if __name__=='__main__':main()
