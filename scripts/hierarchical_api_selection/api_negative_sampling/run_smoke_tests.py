"""Four bounded CPU BERT runs, no fixed evaluation and no production model selection."""
import argparse
from pathlib import Path
import time
from pipeline import ROOT, write, read, require, sha, old
from training import run
CASES=[('sufficient_multi','api_classification/category_mapping'),
       ('few_positive_multi','api_classification/category_internet_things'),
       ('single_output','api_classification/category_content'),
       ('one_positive','api_classification/category_booking')]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Refuse existing smoke directory');a.output.mkdir(parents=True)
    started=time.monotonic();results=[]
    for name,uid in CASES:
        begin=time.monotonic()
        result=run(uid,a.output/name,smoke=True,ratios_requested=[1])
        results.append({'case':name,'classifier_uid':uid,'elapsed_seconds':time.monotonic()-begin,
          'result_path':f'{name}/training_results.json','result_sha256':sha(a.output/name/'training_results.json'),
          'selection_exercised':result['selection_performed'],'roundtrip':result['roundtrip_predictions_equal'],
          'legacy_inference_compatible':result['legacy_inference_strict_load_and_logits_equal'],
          'selected':result['selected']})
    changes=old.check_inventory(ROOT,read('/tmp/phase3_initial_inventory.json')) if Path('/tmp/phase3_initial_inventory.json').exists() else None
    require(changes in (None,[]),'Preexisting files changed during smoke')
    write(a.output/'smoke_results.json',{'passed':True,'smoke_only':True,'cpu_threads':2,
      'maximum_length':32,'per_fold_train_limit':4,'per_fold_validation_limit':4,'maximum_epochs':2,
      'ratio_conditions_exercised':[1],'validation_folds_exercised_per_classifier':1,
      'two_learning_rates_exercised':True,'encoder_gradients_enabled':True,
      'not_a_research_performance_estimate':True,'elapsed_seconds':time.monotonic()-started,'cases':results,
      'existing_file_changes':changes,'fixed_evaluation_used':False})
if __name__=='__main__':main()
