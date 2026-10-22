import argparse
from pathlib import Path
from pipeline import read, write
from training import choose_config
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=read(a.input)
    write(a.output,choose_config(d['results'],d['expected_fold_ids'],d.get('task','api_classification')))
