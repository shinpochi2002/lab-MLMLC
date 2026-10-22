"""Create all candidates, group splits and three Train-only sampling conditions atomically by policy."""
import argparse
from pathlib import Path
from pipeline import create, OUTPUT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    create(p.parse_args().output)
