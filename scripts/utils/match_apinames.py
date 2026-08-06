#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project_paths import legacy_path

# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Set, Generator

# ====== EDIT HERE: specify your file paths ======
FILE1 = Path(legacy_path("/home/nakagawa/softfill_mashup_data_new_api/nocategory/softfill_mashup_new_data.json"))
FILE2 = Path(legacy_path("/home/nakagawa/mashup_api_merged_softfill.json"))
OUT_FOUND = Path("study_data_mashup_apis2.json")
OUT_MISSING = Path("test_data_mashup_apis2.json")
# ================================================

def normalize_name(s: str) -> str:
    return s.strip().lower()

def _maybe_get_name_from_dict(d: Dict[str, Any]) -> str | None:
    """
    単一レコード(dict)からAPI名らしき値を取り出す。
    優先順: "apiname" > "name" > "api_name" > ["api"]["name"] > ["service"]["name"]
    """
    for k in ("apiname", "name", "api_name"):
        if k in d and isinstance(d[k], str):
            return normalize_name(d[k])
    # ネスト（api/name, service/nameなど）
    for parent in ("api", "service"):
        if parent in d and isinstance(d[parent], dict):
            v = d[parent].get("name")
            if isinstance(v, str):
                return normalize_name(v)
    return None

def iter_names_from_any(obj: Any) -> Generator[str, None, None]:
    """
    FILE1 用: どんな形でも API名のイテレータを返す。
    - リスト: 各要素から name を抽出
    - 辞書: 値から name を抽出 or キー自体がAPI名
    - ネスト配列/辞書もなるべく走査（浅め）
    """
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                name = _maybe_get_name_from_dict(item)
                if name: 
                    yield name
            elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                # 例: ["gmail", ...] のような最初の要素が名前
                yield normalize_name(item[0])
    elif isinstance(obj, dict):
        # まずは値から name を試す
        extracted = False
        for v in obj.values():
            if isinstance(v, dict):
                name = _maybe_get_name_from_dict(v)
                if name:
                    extracted = True
                    yield name
        if not extracted:
            # 値から取れなければ「キーがAPI名」とみなす
            for k in obj.keys():
                if isinstance(k, str):
                    yield normalize_name(k)

def load_file1_apiname_set(obj: Any) -> Set[str]:
    return set(iter_names_from_any(obj))

def iter_apimap_from_any(obj: Any) -> Generator[Tuple[str, Any], None, None]:
    """
    FILE2 用: どんな形でも (apiname, record) を返す。
    - 辞書: キーをAPI名とみなす or 値に name/apiname を持つ
    - リスト: 各要素(dict)の name/apiname をキーに
    - { "items": [...] } を含む場合も items を走査
    """
    if isinstance(obj, dict):
        # 1) まずキーがAPI名ケース
        # 文字列キーのみで構成されていそうなら、それを優先採用
        if all(isinstance(k, str) for k in obj.keys()) and len(obj) > 0:
            # ただし "items" 一発だけのコンテナなら次へ
            if not (len(obj) == 1 and "items" in obj and isinstance(obj["items"], list)):
                for k, v in obj.items():
                    yield normalize_name(k), v
                return

        # 2) 値（dict）から name/apiname を抽出
        matched = False
        for v in obj.values():
            if isinstance(v, dict):
                name = _maybe_get_name_from_dict(v)
                if name:
                    matched = True
                    yield name, v
        if matched:
            return

        # 3) items があれば items を走査
        if "items" in obj and isinstance(obj["items"], list):
            for rec in obj["items"]:
                if isinstance(rec, dict):
                    name = _maybe_get_name_from_dict(rec)
                    if name:
                        yield name, rec
            return

        # 4) その他: 浅く総当り
        for k, v in obj.items():
            if isinstance(v, dict):
                name = _maybe_get_name_from_dict(v)
                if name:
                    yield name, v

    elif isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                name = _maybe_get_name_from_dict(rec)
                if name:
                    yield name, rec
            elif isinstance(rec, (list, tuple)) and rec and isinstance(rec[0], str):
                # ["gmail", {...}] のような形
                yield normalize_name(rec[0]), rec
    # それ以外は何も返さない

def main():
    file1_obj = json.loads(FILE1.read_text(encoding="utf-8"))
    file2_obj = json.loads(FILE2.read_text(encoding="utf-8"))

    # FILE1: API名セットを作成
    apinames = load_file1_apiname_set(file1_obj)
    if not apinames:
        raise ValueError("FILE1 から API 名を抽出できませんでした。キー 'name' / 'apiname' / dictキーのいずれかが必要です。")

    # FILE2: (apiname, record) の反復子へ正規化
    found: Dict[str, Any] = {}
    missing: Dict[str, Any] = {}

    any_yield = False
    for name, rec in iter_apimap_from_any(file2_obj):
        any_yield = True
        if name in apinames:
            found[name] = rec
        else:
            missing[name] = rec

    if not any_yield:
        raise ValueError("FILE2 から (apiname, record) を抽出できませんでした。")

    OUT_FOUND.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MISSING.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print("FILE1 api names :", len(apinames))
    print("found           :", len(found), "->", OUT_FOUND)
    print("missing         :", len(missing), "->", OUT_MISSING)

if __name__ == "__main__":
    main()
