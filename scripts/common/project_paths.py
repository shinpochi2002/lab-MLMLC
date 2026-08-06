import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ROOT_JSON_DIR = PROCESSED_DATA_DIR / "root_json"
SPLITS_DIR = DATA_DIR / "splits"
CV_DIR = DATA_DIR / "cv"
API_CATEGORY_LIST_DIR = DATA_DIR / "api_category_list"
CLUSTER_DEFINITIONS_DIR = DATA_DIR / "cluster_definitions"
TEST_SUBSETS_DIR = DATA_DIR / "test_subsets"
STUDYDATA_DIR = PROCESSED_DATA_DIR / "studydata"

MODELS_DIR = PROJECT_ROOT / "models"
BERT_BASE_DIR = MODELS_DIR / "base" / "bert-base-uncased"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
HIERARCHY_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "hierarchy"
CATEGORY_KFOLD_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "category_kfold"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
HIERARCHY_RESULTS_DIR = RESULTS_DIR / "hierarchy"

ARCHIVE_DIR = PROJECT_ROOT / "archive"


def hierarchy_checkpoint_dir(layer: int, pattern: str | None = None) -> Path:
    base = HIERARCHY_CHECKPOINTS_DIR / f"layer{layer}"
    return base / pattern if pattern else base


def studydata_dir(layer: int, pattern: str | None = None) -> Path:
    base = STUDYDATA_DIR / f"layer{layer}"
    return base / pattern if pattern else base


def path_str(path: str | Path) -> str:
    return str(path)


def _legacy_weight_path(raw: str) -> Path | None:
    name = Path(raw).name
    if not name.endswith("_model_weights.pth"):
        return None
    if name == "mashup_softfill_api_all_layer1_model_weights.pth":
        return hierarchy_checkpoint_dir(1) / name

    match = re.match(r"^mashup_softfill_api_all_layer([2-4])_(.+)_model_weights\.pth$", name)
    if not match:
        return None

    layer = int(match.group(1))
    tokens = [token for token in re.split(r"[_-]", match.group(2).replace("a", "_")) if token]
    depth = {2: 1, 3: 3, 4: 4}[layer]
    if len(tokens) < depth or not all(token.isdigit() for token in tokens[:depth]):
        return None
    pattern = f"layer{layer}_" + "_".join(tokens[:depth])
    return hierarchy_checkpoint_dir(layer, pattern) / name



def _legacy_studied_model_path(raw: str) -> Path | None:
    name = Path(raw).name
    if not raw.startswith("/home/nakagawa/studied_model/"):
        return None

    if name == "mashup_softfill_api_big12_mid4_big12_model_weights.pth":
        return hierarchy_checkpoint_dir(3, "layer3_12_4_3") / "mashup_softfill_api_all_layer3_12_model_weights.pth"

    match = re.match(r"^mashup_softfill_api_big12_mid4_big12_mid(\d+)(?:-(\d+))?_model_weights\.pth$", name)
    if match:
        mid = match.group(1)
        small = match.group(2)
        if small is None:
            new_name = f"mashup_softfill_api_all_layer3_12_{mid}_model_weights.pth"
        else:
            new_name = f"mashup_softfill_api_all_layer3_12_{mid}_{mid}_{small}_model_weights.pth"
        return hierarchy_checkpoint_dir(3, "layer3_12_4_3") / new_name

    return None



def _legacy_named_weight_path(raw: str) -> Path | None:
    name = Path(raw).name

    if name == "mashup_softfill_api_big12_mid4_big12_model_weights.pth":
        return hierarchy_checkpoint_dir(3, "layer3_12_4_3") / "mashup_softfill_api_all_layer3_12_model_weights.pth"
    match = re.match(r"^mashup_softfill_api_big12_mid4_big12_mid(\d+)(?:-(\d+))?_model_weights\.pth$", name)
    if match:
        mid = match.group(1)
        small = match.group(2)
        if small is None:
            new_name = f"mashup_softfill_api_all_layer3_12_{mid}_model_weights.pth"
        else:
            new_name = f"mashup_softfill_api_all_layer3_12_{mid}_{mid}_{small}_model_weights.pth"
        return hierarchy_checkpoint_dir(3, "layer3_12_4_3") / new_name

    match = re.match(r"^mashup_softfill_api_8category_(all_big|mid(\d+)(?:-(\d+))?)_new_model_weights_test\.pth$", name)
    if match:
        if match.group(1) == "all_big":
            new_name = "mashup_softfill_api_all_layer3_8_model_weights.pth"
        else:
            mid = match.group(2)
            small = match.group(3)
            if small is None:
                new_name = f"mashup_softfill_api_all_layer3_8_{mid}_model_weights.pth"
            else:
                new_name = f"mashup_softfill_api_all_layer3_8_{mid}_{mid}_{small}_model_weights.pth"
        return hierarchy_checkpoint_dir(3, "layer3_8_4_4") / new_name


    match = re.match(r"^mashup_softfill_api_4category(?:_(\d+))?_new_model_weights_test\.pth$", name)
    if match:
        cat = match.group(1)
        if cat is None:
            new_name = "mashup_softfill_api_all_layer2_4_model_weights.pth"
        else:
            new_name = f"mashup_softfill_api_all_layer2_4_{cat}_model_weights.pth"
        return hierarchy_checkpoint_dir(2, "layer2_4") / new_name

    match = re.match(r"^mashup_softfill_api_4category_(bigclusters|mid(\d+)clusters|small(\d+)-(\d+)clusters)_new_model_weights(?:_test)?\.pth$", name)
    if match:
        kind = match.group(1)
        if kind == "bigclusters":
            new_name = "mashup_softfill_api_all_layer3_4_model_weights.pth"
        elif kind.startswith("mid"):
            mid = match.group(2)
            new_name = f"mashup_softfill_api_all_layer3_4_{mid}_model_weights.pth"
        else:
            mid = match.group(3)
            small = match.group(4)
            new_name = f"mashup_softfill_api_all_layer3_4_{mid}_{mid}_{small}_model_weights.pth"
        return hierarchy_checkpoint_dir(3, "layer3_4_4_8") / new_name

    return None


def legacy_path(path: str | Path) -> str:
    raw = str(path)
    exact_dirs = {
        "/home/nakagawa/lab-MLMLC": PROJECT_ROOT,
        "/home/nakagawa/lab-MLMLC/codex/output/by_pattern": HIERARCHY_RESULTS_DIR / "output" / "by_pattern",
        "/home/nakagawa/lab-MLMLC/create_study_data_and_model/GoogleNews-vectors-negative300.bin": EMBEDDINGS_DIR / "GoogleNews-vectors-negative300.bin",
        "/home/nakagawa/lab-MLMLC/codex/output/layer1_direct_api": HIERARCHY_RESULTS_DIR / "output" / "layer1_direct_api",
        "/home/nakagawa/lab-MLMLC/codex/results/hierarchy_existing_models_condition_2x2": HIERARCHY_RESULTS_DIR / "hierarchy_existing_models_condition_2x2",
        "/home/nakagawa/lab-MLMLC/csvs": TABLES_DIR / "csvs",
        "/home/nakagawa/lab-MLMLC/logs": LOGS_DIR,
        "/home/nakagawa/logs": LOGS_DIR,
        "/home/nakagawa/test_by_category_inc_negative_1_10": SPLITS_DIR / "test" / "test_by_category_inc_negative_1_10",
        "/home/nakagawa/test_by_category_inc_negative_1_10_new": SPLITS_DIR / "test" / "test_by_category_inc_negative_1_10_new",
        "/home/nakagawa/study_category_inc_negative_1_10model_only_train": CHECKPOINTS_DIR / "category" / "study_category_inc_negative_1_10model_only_train",
        "lab-MLMLC/csvs": TABLES_DIR / "csvs",
        "csvs": TABLES_DIR / "csvs",
        "logs": LOGS_DIR,
        "ga_logs": LOGS_DIR / "ga_logs",
        "codex/output/by_pattern": HIERARCHY_RESULTS_DIR / "output" / "by_pattern",
        "codex/output/layer1_direct_api": HIERARCHY_RESULTS_DIR / "output" / "layer1_direct_api",
    }
    if raw in exact_dirs:
        return str(exact_dirs[raw])

    replacements = (
        ("/home/nakagawa/lab-MLMLC/api_category_list/", API_CATEGORY_LIST_DIR),
        ("/home/nakagawa/api_category_list/", API_CATEGORY_LIST_DIR),
        ("api_category_list/", API_CATEGORY_LIST_DIR),
        ("/home/nakagawa/lab-MLMLC/studydata/", STUDYDATA_DIR),
        ("/home/nakagawa/studydata/", STUDYDATA_DIR),
        ("studydata/", STUDYDATA_DIR),
        ("/home/nakagawa/lab-MLMLC/mashup_data/", RAW_DATA_DIR / "mashup_data"),
        ("/home/nakagawa/mashup_data/", RAW_DATA_DIR / "mashup_data"),
        ("mashup_data/", RAW_DATA_DIR / "mashup_data"),
        ("/home/nakagawa/lab-MLMLC/new_api_data/", RAW_DATA_DIR / "new_api_data"),
        ("/home/nakagawa/new_api_data/", RAW_DATA_DIR / "new_api_data"),
        ("new_api_data/", RAW_DATA_DIR / "new_api_data"),
        (
            "/home/nakagawa/lab-MLMLC/softfill_mashup_data_new_api_inclu_nomash/",
            PROCESSED_DATA_DIR / "softfill_mashup_data_new_api_inclu_nomash",
        ),
        (
            "/home/nakagawa/softfill_mashup_data_new_api_inclu_nomash/",
            PROCESSED_DATA_DIR / "softfill_mashup_data_new_api_inclu_nomash",
        ),
        (
            "softfill_mashup_data_new_api_inclu_nomash/",
            PROCESSED_DATA_DIR / "softfill_mashup_data_new_api_inclu_nomash",
        ),
        (
            "/home/nakagawa/lab-MLMLC/softfill_mashup_data_new_api/",
            PROCESSED_DATA_DIR / "softfill_mashup_data_new_api",
        ),
        (
            "/home/nakagawa/softfill_mashup_data_new_api/",
            PROCESSED_DATA_DIR / "softfill_mashup_data_new_api",
        ),
        (
            "softfill_mashup_data_new_api/",
            PROCESSED_DATA_DIR / "softfill_mashup_data_new_api",
        ),
        ("/home/nakagawa/lab-MLMLC/cluster/", CLUSTER_DEFINITIONS_DIR),
        ("/home/nakagawa/cluster/", CLUSTER_DEFINITIONS_DIR),
        ("cluster/", CLUSTER_DEFINITIONS_DIR),
        ("/home/nakagawa/lab-MLMLC/create_study_data_and_model/", PROJECT_ROOT / "scripts" / "create_study_data_and_model"),
        ("/home/nakagawa/lab-MLMLC/models/bert-base-uncased/", BERT_BASE_DIR),
        ("/home/nakagawa/models/bert-base-uncased/", BERT_BASE_DIR),
        (
            "/home/nakagawa/lab-MLMLC/study_category_inc_nomashup_model_new_kfold/",
            CATEGORY_KFOLD_CHECKPOINTS_DIR / "study_category_inc_nomashup_model_new_kfold",
        ),
        (
            "/home/nakagawa/study_category_inc_nomashup_model_new_kfold/",
            CATEGORY_KFOLD_CHECKPOINTS_DIR / "study_category_inc_nomashup_model_new_kfold",
        ),
        ("/home/nakagawa/lab-MLMLC/csvs/", TABLES_DIR / "csvs"),
        ("/home/nakagawa/lab-MLMLC/logs/", LOGS_DIR),
        ("/home/nakagawa/logs/", LOGS_DIR),
        ("lab-MLMLC/csvs/", TABLES_DIR / "csvs"),
        ("csvs/", TABLES_DIR / "csvs"),
        ("logs/", LOGS_DIR),
        ("ga_logs/", LOGS_DIR / "ga_logs"),
        ("codex/output/by_pattern/", HIERARCHY_RESULTS_DIR / "output" / "by_pattern"),
        ("codex/output/layer1_direct_api/", HIERARCHY_RESULTS_DIR / "output" / "layer1_direct_api"),
        ("codex/baseline/results/", RESULTS_DIR / "baselines"),
        ("codex/results/", HIERARCHY_RESULTS_DIR),
        (
            "/home/nakagawa/lab-MLMLC/codex/results/hierarchy_existing_models_condition_2x2/",
            HIERARCHY_RESULTS_DIR / "hierarchy_existing_models_condition_2x2",
        ),
        (
            "/home/nakagawa/lab-MLMLC/codex/output/by_pattern/",
            HIERARCHY_RESULTS_DIR / "output" / "by_pattern",
        ),
        (
            "/home/nakagawa/lab-MLMLC/codex/output/layer1_direct_api/",
            HIERARCHY_RESULTS_DIR / "output" / "layer1_direct_api",
        ),
    )
    for old_prefix, new_base in replacements:
        if raw.startswith(old_prefix):
            rel = raw[len(old_prefix) :]
            return str(Path(new_base) / rel)


    named_weight_path = _legacy_named_weight_path(raw)
    if named_weight_path is not None:
        return str(named_weight_path)

    studied_model_path = _legacy_studied_model_path(raw)
    if studied_model_path is not None:
        return str(studied_model_path)

    weight_path = _legacy_weight_path(raw)
    if weight_path is not None:
        return str(weight_path)


    split_dir_match = re.match(r"^/home/nakagawa/(out_by_category[^/]*|train_by_category[^/]*|test_by_category[^/]*)(?:/(.*))?$", raw)
    if split_dir_match:
        directory, rest = split_dir_match.groups()
        if directory.startswith("out_by_category"):
            base = SPLITS_DIR / "by_category" / directory
        elif directory.startswith("train_by_category"):
            base = SPLITS_DIR / "train" / directory
        else:
            base = SPLITS_DIR / "test" / directory
        return str(base / rest) if rest else str(base)

    cv_dir_match = re.match(r"^/home/nakagawa/(study_category[^/]*_cv)(?:/(.*))?$", raw)
    if cv_dir_match:
        directory, rest = cv_dir_match.groups()
        base = CV_DIR / directory
        return str(base / rest) if rest else str(base)

    category_model_dir_match = re.match(r"^/home/nakagawa/(study_category[^/]*model[^/]*)(?:/(.*))?$", raw)
    if category_model_dir_match:
        directory, rest = category_model_dir_match.groups()
        base = CHECKPOINTS_DIR / "category" / directory
        return str(base / rest) if rest else str(base)

    exact_layer_match = re.match(r"^/home/nakagawa/lab-MLMLC/(layer([1-4])[^/]*)$", raw)
    if exact_layer_match:
        directory, layer = exact_layer_match.groups()
        return str(hierarchy_checkpoint_dir(int(layer), directory))

    relative_layer_match = re.match(r"^(layer([1-4])[^/]*)(?:/(.*))?$", raw)
    if relative_layer_match:
        directory, layer, rest = relative_layer_match.groups()
        base = hierarchy_checkpoint_dir(int(layer), directory)
        return str(base / rest) if rest else str(base)

    layer_match = re.match(r"^/home/nakagawa/lab-MLMLC/(layer([1-4])[^/]*)/(.*)$", raw)
    if layer_match:
        directory, layer, rest = layer_match.groups()
        return str(hierarchy_checkpoint_dir(int(layer), directory) / rest)

    layer_home_match = re.match(r"^/home/nakagawa/(layer([1-4])[^/]*)/(.*)$", raw)
    if layer_home_match:
        directory, layer, rest = layer_home_match.groups()
        return str(hierarchy_checkpoint_dir(int(layer), directory) / rest)

    root_files = {
        "/home/nakagawa/lab-MLMLC/test_data_mashup_apis.json": ROOT_JSON_DIR / "test_data_mashup_apis.json",
        "/home/nakagawa/test_data_mashup_apis.json": ROOT_JSON_DIR / "test_data_mashup_apis.json",
        "/home/nakagawa/collaboration_test_data.json": ROOT_JSON_DIR / "collaboration_test_data.json",
        "/home/nakagawa/lab-MLMLC/collaboration_test_data.json": ROOT_JSON_DIR / "collaboration_test_data.json",
        "/home/nakagawa/eval_results_summary_inc_negative_1_10.json": SUMMARIES_DIR / "eval_results_summary_inc_negative_1_10.json",
        "/home/nakagawa/8category_small_multi_log.txt": LOGS_DIR / "root" / "8category_small_multi_log.txt",
        "/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_4big_test_data.json": STUDYDATA_DIR / "layer3" / "layer3_4_4_8" / "softfill_mashup_data_new_api_category_inclu_nomash_all_inc_4big_test_data.json",
        "/home/nakagawa/softfill_mashup_data_new_api_category_inclu_nomash_all_inc_8big_test_data.json": STUDYDATA_DIR / "layer3" / "layer3_8_4_4" / "split" / "softfill_mashup_data_new_api_category_inclu_nomash_split_all_inc_8big_test_data.json",
        "/home/nakagawa/lab-MLMLC/GoogleNews-vectors-negative300.bin": EMBEDDINGS_DIR
        / "GoogleNews-vectors-negative300.bin",
        "/home/nakagawa/GoogleNews-vectors-negative300.bin": EMBEDDINGS_DIR
        / "GoogleNews-vectors-negative300.bin",
        "GoogleNews-vectors-negative300.bin": EMBEDDINGS_DIR / "GoogleNews-vectors-negative300.bin",
    }
    if raw in root_files:
        return str(root_files[raw])


    home_file_match = re.match(r"^/home/nakagawa/([^/]+)$", raw)
    if home_file_match:
        name = home_file_match.group(1)
        suffix = Path(name).suffix.lower()
        if suffix == ".json":
            return str(ROOT_JSON_DIR / name)
        if suffix == ".csv":
            return str(TABLES_DIR / name)
        if suffix == ".log":
            return str(LOGS_DIR / name)
        if suffix == ".pth":
            return str(CHECKPOINTS_DIR / "legacy" / name)
        if suffix == ".bin":
            return str(EMBEDDINGS_DIR / name)

    return raw
