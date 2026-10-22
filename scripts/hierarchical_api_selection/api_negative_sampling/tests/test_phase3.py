import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import build_split,sample_negatives,safe_weights,create,write
from training import choose_config,metrics_from_counts,bounded_subset

def member(i,group=None,source='mashup',targets=None):
    return {'entity_id':str(i),'group_id':str(group or i),'source_type':source,'target_labels':targets or []}

def population(pos=5,neg=50):
    return [member('api',source='api',targets=['a'])]+[member('p'+str(i),targets=['a']) for i in range(pos)]+[member('n'+str(i)) for i in range(neg)]

class PhaseThreeTests(unittest.TestCase):
    def test_ratio_nesting_and_reproduction(self):
        rows=[member(i) for i in range(40)]
        a=[sample_negatives(rows,3,r,'classifier') for r in [1,3,5]]
        self.assertTrue(set(a[0]['entity_ids'])<=set(a[1]['entity_ids'])<=set(a[2]['entity_ids']))
        self.assertEqual(a[1],sample_negatives(rows,3,3,'classifier'))
    def test_whole_group_and_shortage(self):
        rows=[member('a','same'),member('b','same')]
        r=sample_negatives(rows,1,1,'x');self.assertEqual(r['actual_negatives'],2);self.assertEqual(r['group_rounding_excess'],1)
        r=sample_negatives(rows,1,5,'x');self.assertTrue(r['shortage']);self.assertEqual(len(set(r['entity_ids'])),2)
    def test_mixed_positive_negative_group_mandatory(self):
        rows=[member('negative','positive-group'),member('other')]
        r=sample_negatives(rows,1,1,'x',{'positive-group'})
        self.assertIn('negative',r['entity_ids']);self.assertEqual(r['mandatory_group_negatives'],1)
    def test_api_anchor_and_mashup_validation(self):
        rows=population();rows.append(member('alias','api'))
        s=build_split(rows,['a'],'x');self.assertEqual(s['method'],'kfold')
        for f in s['folds']:
            self.assertIn('api',f['positive_train_entity_ids']);self.assertNotIn('alias',f['validation_entity_ids'])
            self.assertGreater(f['validation_positive'],0);self.assertGreater(f['validation_negative'],0)
    def test_reduced_folds_depend_on_mashup_positive_groups(self):
        s=build_split(population(pos=2),['a'],'x');self.assertEqual(s['n_splits'],2)
    def test_single_positive_never_cv_or_select(self):
        s=build_split(population(pos=0),['a'],'x');self.assertEqual(s['method'],'fixed_config');self.assertFalse(s['selection_enabled'])
        self.assertIn('api',s['folds'][0]['positive_train_entity_ids']);self.assertEqual(s['reason'],'one_positive_record')
    def test_one_mashup_positive_uses_common_default(self):
        s=build_split(population(pos=1),['a'],'x');self.assertEqual(s['method'],'fixed_config')
        self.assertIn('p0',s['folds'][0]['positive_train_entity_ids'])
    def test_holdout_positive_and_group_isolation(self):
        rows=population(pos=25);rows.append(member('copy','p0',targets=['a']))
        s=build_split(rows,['a'],'x');self.assertEqual(s['method'],'holdout')
        f=s['folds'][0];lookup={r['entity_id']:r for r in rows}
        tr=set(f['positive_train_entity_ids']+f['negative_train_candidate_entity_ids'])
        self.assertFalse({lookup[e]['group_id'] for e in tr}&{lookup[e]['group_id'] for e in f['validation_entity_ids']})
    def test_weights_all_positive_all_negative_and_normal(self):
        w=safe_weights([['a'],['a']],['a','b']);self.assertEqual([x['pos_weight'] for x in w],[1.,1.]);self.assertTrue(all(x['correction'] for x in w))
        w=safe_weights([['a'],[],[]],['a']);self.assertEqual(w[0]['pos_weight'],2.)
    def test_weight_extremes_finite(self):
        w=safe_weights([['a']]+[[]]*10001,['a']);self.assertEqual(w[0]['pos_weight'],10000.)
    def test_empty_train_rejected(self):
        with self.assertRaises(ValueError):safe_weights([],['a'])
    def test_empty_output_refused_before_reading(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):create(Path(d))
            p=Path(d)/'x';write(p,{})
            with self.assertRaises(FileExistsError):write(p,{})
    def test_lr_loss_first_ratio_f1_first_median(self):
        rows=[]
        for ratio in [1,3,5]:
            for lr in [2e-5,3e-5]:
                for fold,epoch in [('a',2),('b',5)]:
                    rows.append({'ratio':ratio,'learning_rate':lr,'fold_id':fold,'best_epoch':epoch,
                      'validation_loss':10 if ratio==3 else 1 if lr==2e-5 else 2,
                      'precision':.5,'recall':.5,'micro_f1':.9 if ratio==3 else .5})
        s=choose_config(rows,['a','b']);self.assertEqual((s['ratio'],s['learning_rate'],s['epochs']),(3,2e-5,4))
    def test_ties_choose_smaller_ratio_and_lr(self):
        rows=[{'ratio':r,'learning_rate':lr,'fold_id':'a','best_epoch':1,'validation_loss':1,'precision':.5,'recall':.5,'micro_f1':.5} for r in [1,3,5] for lr in [2e-5,3e-5]]
        s=choose_config(rows,['a']);self.assertEqual((s['ratio'],s['learning_rate']),(1,2e-5))
    def test_missing_duplicate_or_nonfinite_result_rejected(self):
        r={'ratio':1,'learning_rate':2e-5,'fold_id':'a','best_epoch':1,'validation_loss':1,'precision':.5,'recall':.5,'micro_f1':.5}
        with self.assertRaises(ValueError):choose_config([r],['a'])
        with self.assertRaises(ValueError):choose_config([r,r],['a'])
        with self.assertRaises(ValueError):choose_config([{**r,'validation_loss':math.nan}],['a'])
    def test_micro_f1_all_negative_explicit_zero(self):
        self.assertEqual(metrics_from_counts(0,0,0)['micro_f1'],0.)
    def test_smoke_subset_keeps_both_roles(self):
        ids=['a','b','c','d'];t={'a':['x'],'b':[],'c':[],'d':[]}
        self.assertEqual(bounded_subset(ids,t,2),['a','b'])
    def test_early_stopping_restores_earlier_best_epoch(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable')
        from unittest.mock import patch
        import training
        class Tokenizer:
            @classmethod
            def from_pretrained(cls,*a,**k):return cls()
            def __call__(self,texts,**kw):
                return {'input_ids':torch.ones((len(texts),2),dtype=torch.long),'attention_mask':torch.ones((len(texts),2),dtype=torch.long)}
        class Model(torch.nn.Module):
            def __init__(self,*a):super().__init__();self.linear=torch.nn.Linear(2,1)
            def forward(self,a,b):return self.linear(a.float())
        scores=iter([1.,.5,.7,.8,.9]);hashes=[]
        def evaluate(model,*args):
            hashes.append(training.state_hash(model))
            return {'validation_loss':next(scores),'precision':.5,'recall':.5,'micro_f1':.5}
        with patch.object(training,'imports',return_value=(torch,None,Tokenizer)),patch.object(training,'model_class',return_value=Model),patch.object(training,'evaluate',side_effect=evaluate):
            model,info=training.train_one('local',['a'],['p','n'],['v'],{'p':{'description':'p'},'n':{'description':'n'},'v':{'description':'v'}},{'p':['a'],'n':[],'v':['a']},2e-5,50)
        self.assertEqual(info['best_epoch'],2);self.assertEqual(len(info['history']),5)
        self.assertTrue(info['best_weights_restored']);self.assertEqual(training.state_hash(model),hashes[1])
    def test_single_output_loss_nonzero_gradient(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable')
        for ys in [[[1.],[0.]],[[1.],[1.]],[[0.],[0.]]]:
            y=torch.tensor(ys);labels=[['a'] if row[0] else [] for row in ys];w=safe_weights(labels,['a'])[0]['pos_weight']
            x=torch.zeros_like(y,requires_grad=True);loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))(x,y);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((x.grad!=0).all())
if __name__=='__main__':unittest.main()
