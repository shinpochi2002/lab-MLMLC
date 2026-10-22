"""Integrity tests for group isolation, deterministic splitting, immutable outputs and lineage."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as p

class SplitSafetyTests(unittest.TestCase):
    def records(self):
        rows=[]
        for i in range(60):
            rows.append({'entity_id':f'mashup:{i}','source_type':'mashup','description':f'text {i}',
              'category_labels':['x' if i%2 else 'y'],'api_labels':['a' if i%3 else 'b'],'api_classification_eligible':True})
        rows.append({'entity_id':'api:1','source_type':'api','description':'text 0','category_labels':['z'],
          'api_labels':None,'api_classification_eligible':False})
        return rows
    def test_normalization_preserves_case_and_original_input(self):
        text=' Ａ &amp; B\n C '
        self.assertEqual(p.normalize(text),'A & B C');self.assertNotEqual(p.normalize('Foo'),p.normalize('foo'))
        self.assertEqual(text,' Ａ &amp; B\n C ')
    def test_cross_origin_same_text_different_labels_remain_records_in_one_group(self):
        rows=self.records();original=copy.deepcopy(rows);groups=p.make_groups(rows)
        self.assertEqual(len(groups),60);self.assertEqual(rows[0]['group_id'],rows[-1]['group_id'])
        for r,o in zip(rows,original):
            self.assertEqual(r['category_labels'],o['category_labels']);self.assertEqual(r['api_labels'],o['api_labels'])
        self.assertEqual(len(rows),61)
    def test_all_group_roles_exclusive_and_each_group_held_out_once(self):
        rows=self.records();groups=p.make_groups(rows);outer,folds=p.plan_folds(rows,groups)
        self.assertEqual(len(outer),len(groups));counts={g['group_id']:0 for g in groups}
        for fold in folds.values():
            tr,es,val=[set(fold[k]) for k in ['train_groups','early_stopping_groups','validation_groups']]
            self.assertFalse(tr&es or tr&val or es&val)
            self.assertEqual(tr|es,set(fold['outer_train_groups']))
            for g in val:counts[g]+=1
        self.assertEqual(set(counts.values()),{1})
    def test_reordered_inputs_identical_and_seed_changes_assignment(self):
        rows=self.records();groups=p.make_groups(rows);one=p.plan_folds(rows,groups)
        self.assertEqual(one,p.plan_folds(list(reversed(rows)),list(reversed(groups))))
        self.assertNotEqual(one,p.plan_folds(rows,groups,43))
    def test_write_refuses_existing_file_without_modification(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'artifact.json';p.write(path,{'unchanged':1});before=path.read_bytes()
            with self.assertRaises(FileExistsError):p.write(path,{'unchanged':0})
            self.assertEqual(before,path.read_bytes())
    def test_prepare_refuses_existing_directory_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'already exists'):p.prepare(Path(t)/'no_repository',Path(t))
    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'bad.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):p.read(path)
    def test_parent_projection_restricts_labels_and_deduplicates_membership(self):
        pattern={'category_layers':3,'category_paths':{'a':[0,1],'b':[0,2],'c':[1,0]}}
        r={'category_labels':['a','b','c']};actual=p.project_classifiers(r,pattern)
        self.assertEqual(actual[(1,())],{'0','1'})
        self.assertEqual(actual[(2,(0,))],{'0-1','0-2'})
        self.assertEqual(actual[(3,(0,1))],{'a'})
        self.assertEqual(len(actual),6)
    def test_inventory_detects_same_size_input_change(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path=root/'input';path.write_text('abc');before=p.inventory(root)
            path.write_text('xyz');self.assertTrue(p.check_inventory(root,before))
    def test_same_name_distinct_origins_with_different_text_are_not_deleted_or_linked(self):
        rows=self.records()[:2]
        rows[0].update(source_key='same_name',entity_id='mashup:1')
        rows[1].update(source_key='same_name',entity_id='api:1',source_type='api')
        groups=p.make_groups(rows)
        self.assertEqual(len(groups),2)
        self.assertNotEqual(rows[0]['group_id'],rows[1]['group_id'])
    def test_empty_inputs_cannot_form_a_shared_group(self):
        rows=self.records()[:1];rows[0]['description']='   '
        with self.assertRaisesRegex(ValueError,'Empty normalized text'):p.make_groups(rows)
    def test_singleton_label_groups_never_forced_across_roles(self):
        features={f'g{i}':{'api:unique_'+str(i):1,'category:c':1} for i in range(17)}
        one=p.stratify(features,[.2]*5);self.assertEqual(set(one),set(features))
        self.assertLessEqual(max(list(one.values()).count(i) for i in range(5))-min(list(one.values()).count(i) for i in range(5)),1)

if __name__=='__main__':unittest.main()
