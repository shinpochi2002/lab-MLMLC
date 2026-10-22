import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import classifier_pipeline as p

def members(n,labels=('a','b'),group_size=1):
    return [{'entity_id':f'mashup:{i:03d}','group_id':f'group:{i//group_size:03d}',
      'target_labels':[labels[i%len(labels)]],'source_type':'mashup'} for i in range(n)]

class LocalPartitionTests(unittest.TestCase):
    def test_single_record_never_attempts_twofold(self):
        r=p.make_partition(members(1),['a','b'])
        self.assertEqual(r['method'],'unavailable');self.assertEqual(r['folds'],[])
    def test_multiple_records_one_text_group_still_unavailable(self):
        r=p.make_partition(members(4,group_size=4),['a','b'])
        self.assertEqual(r['reason'],'one_independent_group')
    def test_fold_count_uses_independent_groups(self):
        for n,k in [(2,2),(3,3),(4,4),(6,5),(19,5)]:
            with self.subTest(n=n):self.assertEqual(p.make_partition(members(n),['a','b'])['n_splits'],k)
        self.assertEqual(p.make_partition(members(8,group_size=2),['a','b'])['n_splits'],4)
    def test_twenty_record_boundary_is_holdout(self):
        self.assertEqual(p.make_partition(members(19),['a','b'])['method'],'kfold')
        self.assertEqual(p.make_partition(members(20),['a','b'])['method'],'holdout')
    def test_single_output_all_positive_attempts_but_does_not_claim_normal_success(self):
        r=p.make_partition(members(7,('a',)),['a'])
        self.assertEqual(r['method'],'kfold');self.assertEqual(r['n_splits'],5)
        self.assertEqual(r['training_distribution_status'],'constant_targets_not_normal_cv_success')
        self.assertTrue(r['requires_phase_3_decision'])
        for f in r['folds']:
            self.assertEqual(f['train_legacy_pos_weight_zero_labels'],['a'])
            self.assertIsNone(f['training_success'])
    def test_single_output_binary_both_classes_supported(self):
        rows=members(12,('a',))
        for i in range(0,12,2):rows[i]['target_labels']=[]
        r=p.make_partition(rows,['a'])
        self.assertTrue(r['full_profile']['single_output_has_positive_and_negative'])
        self.assertEqual(r['method'],'kfold')
    def test_fold_group_disjoint_and_validation_complete(self):
        rows=members(18,group_size=2);r=p.make_partition(rows,['a','b']);lookup={x['entity_id']:x for x in rows};seen=[]
        for f in r['folds']:
            tr={lookup[e]['group_id'] for e in f['train_entity_ids']};va={lookup[e]['group_id'] for e in f['validation_entity_ids']}
            self.assertFalse(tr&va);seen+=list(va)
        self.assertEqual(len(seen),len(set(seen)));self.assertEqual(set(seen),{x['group_id'] for x in rows})
    def test_holdout_two_groups_never_rounds_validation_to_zero(self):
        r=p.make_partition(members(20,group_size=10),['a','b'])
        f=r['folds'][0]
        self.assertEqual(len(f['train_entity_ids']),10);self.assertEqual(len(f['validation_entity_ids']),10)
        self.assertIn('service_ratio_deviation_over_0_10_due_to_group_constraints',f['warnings'])
    def test_no_silent_move_of_validation_only_label(self):
        rows=members(8,('a',));rows[0]['target_labels']=['b'];r=p.make_partition(rows,['a','b'])
        self.assertTrue(any('b' in f['train_profile']['zero_positive_labels'] for f in r['folds']))
        self.assertEqual(sum(rows[0]['entity_id'] in f['validation_entity_ids'] for f in r['folds']),1)
    def test_reproducible_reordered_and_input_not_mutated(self):
        rows=members(16);before=copy.deepcopy(rows);one=p.make_partition(rows,['a','b'])
        self.assertEqual(one,p.make_partition(list(reversed(rows)),['a','b']))
        self.assertEqual(rows,before)
    def test_group_can_change_role_across_independent_classifiers(self):
        a=members(19);b=members(20)
        self.assertEqual(p.make_partition(a,['a','b'])['method'],'kfold')
        self.assertEqual(p.make_partition(b,['a','b'])['method'],'holdout')
    def test_profile_validator_independently_reproduces_counts(self):
        rows=members(7,group_size=2)
        self.assertEqual(p.target_profile(rows,['a','b','unused']),p.independent_profile(rows,['a','b','unused']))
    def test_no_overwrite_and_no_inputs_read_when_output_exists(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)
            with self.assertRaisesRegex(ValueError,'already exists'):p.create(path,path/'absent',path)
            target=path/'file';p.write(target,{'keep':1});old=target.read_bytes()
            with self.assertRaises(FileExistsError):p.write(target,{'keep':0})
            self.assertEqual(old,target.read_bytes())

if __name__=='__main__':unittest.main()
