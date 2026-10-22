import argparse
from pathlib import Path
from pipeline import OUTPUT, validate, write
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--write-report',action='store_true');p.add_argument('--readiness-report',action='store_true')
    a=p.parse_args();result=validate(a.output,a.write_report)
    if a.readiness_report:write(a.output/'checks/data_validation_results.json',result)
