import json

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# JSONファイルを読み込む
with open(legacy_path("/home/nakagawa/study_data_mashup_apis.json"), 'r', encoding='utf-8') as file:
    data = json.load(file)
    
a = len(data)
print(a)