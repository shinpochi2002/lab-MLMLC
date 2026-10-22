"""Reproducibility check of saved splits; creation is inseparable from preparation to enforce split-before-sample."""
import argparse
from pathlib import Path
from pipeline import OUTPUT, validate
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    validate(p.parse_args().output)
