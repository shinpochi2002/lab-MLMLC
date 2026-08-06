#!/usr/bin/env python3
"""Mask official API-name mentions in test input text.

This script intentionally uses only official API names registered in the API
catalog.  It does not read or generate manual aliases, semantic expansions,
stemming rules, singular/plural variants, or sample-specific labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "hierarchy"
    / "output"
    / "layer1_direct_api"
    / "test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "hierarchy"
    / "output"
    / "layer1_direct_api"
    / "test_data_mashup_apis_layer1_drop_unknown_samples_categories_from_collaboration_api_name_masked.json"
)
DEFAULT_CATALOG = ROOT / "data" / "processed" / "root_json" / "api_dataset_like_manifest.json"
DEFAULT_AUDIT = Path(__file__).resolve().parent / "api_name_masking_audit.json"
DEFAULT_MASK_LOG_CSV = Path(__file__).resolve().parent / "api_name_masking_log.csv"
DEFAULT_RESOLVED_DICTIONARY = (
    Path(__file__).resolve().parent / "api_name_masking_resolved_dictionary.json"
)
PLACEHOLDER = "<API_NAME>"

ENTITY_RE = re.compile(r"&(?:#[xX][0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]+);")
TEXT_FIELD_CANDIDATES = (
    "description",
    "text",
    "input_text",
    "query",
    "utterance",
)
TEXT_ID_FIELD_CANDIDATES = (
    "text_id",
    "sample_id",
    "mashup_id",
    "id",
    "name",
)
API_NAME_FIELD_CANDIDATES = (
    "official_api_name",
    "api_name",
    "name",
    "title",
    "canonical_name",
    "api",
)
API_ID_FIELD_CANDIDATES = (
    "official_api_id",
    "api_id",
    "id",
    "identifier",
)
MASKED_DATA_FIELDS = ("original_text", "masked_text", "num_masked_api_mentions")
MASK_LOG_FIELDS = (
    "row_index",
    "text_id",
    "official_api_id",
    "official_api_name",
    "normalized_api_name",
    "matched_surface_form",
    "normalized_matched_surface_form",
    "start_char",
    "end_char",
)


@dataclass(frozen=True)
class Token:
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class CatalogRecord:
    official_api_name: str
    official_api_id: str = ""


@dataclass(frozen=True)
class ApiNameEntry:
    normalized_api_name: str
    tokens: tuple[str, ...]
    official_api_names: tuple[str, ...]
    official_api_ids: tuple[str, ...]


@dataclass(frozen=True)
class ApiNameMatch:
    entry: ApiNameEntry
    matched_surface_form: str
    normalized_matched_surface_form: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class TextRecord:
    row_index: int
    text_id: str
    text: str
    location_kind: str
    key: Any
    text_field: str | None = None


@dataclass
class InputBundle:
    data: Any
    input_format: str
    csv_fieldnames: list[str] | None = None


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def choose_field(
    fieldnames: Iterable[str],
    explicit: str | None,
    candidates: Sequence[str],
    *,
    required: bool,
    label: str,
) -> str | None:
    names = list(fieldnames)
    if explicit:
        for name in names:
            if name == explicit:
                return name
        explicit_key = canonical_field_name(explicit)
        for name in names:
            if canonical_field_name(name) == explicit_key:
                return name
        raise ValueError(f"{label} column not found: {explicit!r}")

    normalized_to_name = {canonical_field_name(name): name for name in names}
    for candidate in candidates:
        selected = normalized_to_name.get(canonical_field_name(candidate))
        if selected is not None:
            return selected

    if required:
        raise ValueError(
            f"Could not infer {label} column. Available columns: {', '.join(names)}"
        )
    return None


def decoded_units(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield HTML-unescaped characters with their original source span."""
    index = 0
    while index < len(text):
        if text[index] == "&":
            match = ENTITY_RE.match(text, index)
            if match:
                source = match.group(0)
                decoded = html.unescape(source)
                if decoded != source:
                    for character in decoded:
                        yield character, index, match.end()
                    index = match.end()
                    continue

        yield text[index], index, index + 1
        index += 1


def is_ascii_alnum(character: str) -> bool:
    return ("a" <= character <= "z") or ("0" <= character <= "9")


def normalized_tokens_with_spans(text: str) -> list[Token]:
    """Normalize text and return normalized tokens mapped to source spans.

    The deterministic normalization is:
    HTML unescape, Unicode NFKC, lowercase, replace non-ASCII alphanumerics with
    spaces, collapse whitespace, and strip leading/trailing whitespace.
    """
    tokens: list[Token] = []
    current: list[str] = []
    token_start: int | None = None
    token_end: int | None = None

    def flush() -> None:
        nonlocal current, token_start, token_end
        if current and token_start is not None and token_end is not None:
            tokens.append(Token("".join(current), token_start, token_end))
        current = []
        token_start = None
        token_end = None

    for decoded, source_start, source_end in decoded_units(text):
        normalized = unicodedata.normalize("NFKC", decoded).lower()
        for character in normalized:
            if is_ascii_alnum(character):
                if token_start is None:
                    token_start = source_start
                current.append(character)
                token_end = source_end
            else:
                flush()

    flush()
    return tokens


def normalize_text(text: str) -> str:
    return " ".join(token.text for token in normalized_tokens_with_spans(text))


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def load_input(path: Path) -> InputBundle:
    if path.suffix.lower() == ".csv":
        rows, fieldnames = load_csv_rows(path)
        return InputBundle(rows, "csv", fieldnames)

    data = load_json(path)
    if isinstance(data, dict):
        return InputBundle(data, "json_dict")
    if isinstance(data, list):
        return InputBundle(data, "json_list")
    raise ValueError("Input data must be a CSV, JSON object, or JSON array")


def stringify_optional(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def load_catalog(
    path: Path,
    *,
    api_name_column: str | None,
    api_id_column: str | None,
) -> list[CatalogRecord]:
    if path.suffix.lower() == ".csv":
        rows, fieldnames = load_csv_rows(path)
        name_field = choose_field(
            fieldnames,
            api_name_column,
            API_NAME_FIELD_CANDIDATES,
            required=True,
            label="official API name",
        )
        id_field = choose_field(
            fieldnames,
            api_id_column,
            API_ID_FIELD_CANDIDATES,
            required=False,
            label="official API ID",
        )
        return [
            CatalogRecord(
                official_api_name=stringify_optional(row.get(name_field)).strip(),
                official_api_id=stringify_optional(row.get(id_field)).strip()
                if id_field
                else "",
            )
            for row in rows
        ]

    data = load_json(path)
    records: list[CatalogRecord] = []

    if isinstance(data, list):
        if not data:
            return records
        if all(isinstance(item, dict) for item in data):
            all_fields = OrderedDict.fromkeys(
                field for item in data for field in item.keys()
            )
            name_field = choose_field(
                all_fields,
                api_name_column,
                API_NAME_FIELD_CANDIDATES,
                required=True,
                label="official API name",
            )
            id_field = choose_field(
                all_fields,
                api_id_column,
                API_ID_FIELD_CANDIDATES,
                required=False,
                label="official API ID",
            )
            for item in data:
                records.append(
                    CatalogRecord(
                        official_api_name=stringify_optional(item.get(name_field)).strip(),
                        official_api_id=stringify_optional(item.get(id_field)).strip()
                        if id_field
                        else "",
                    )
                )
            return records
        if all(isinstance(item, str) for item in data):
            return [CatalogRecord(official_api_name=item.strip()) for item in data]
        raise ValueError("JSON catalog arrays must contain either objects or strings")

    if isinstance(data, dict):
        values = list(data.values())
        value_dicts = [value for value in values if isinstance(value, dict)]
        use_value_name = bool(api_name_column)
        if value_dicts and not use_value_name:
            all_fields = OrderedDict.fromkeys(
                field for value in value_dicts for field in value.keys()
            )
            use_value_name = (
                choose_field(
                    all_fields,
                    None,
                    API_NAME_FIELD_CANDIDATES,
                    required=False,
                    label="official API name",
                )
                is not None
            )

        if use_value_name:
            if not value_dicts:
                raise ValueError(
                    "An official API name column was supplied, but JSON catalog "
                    "values are not objects"
                )
            all_fields = OrderedDict.fromkeys(
                field for value in value_dicts for field in value.keys()
            )
            name_field = choose_field(
                all_fields,
                api_name_column,
                API_NAME_FIELD_CANDIDATES,
                required=True,
                label="official API name",
            )
            id_field = choose_field(
                all_fields,
                api_id_column,
                API_ID_FIELD_CANDIDATES,
                required=False,
                label="official API ID",
            )
            for key, value in data.items():
                if not isinstance(value, dict):
                    raise ValueError(f"Mixed JSON catalog value type at {key!r}")
                records.append(
                    CatalogRecord(
                        official_api_name=stringify_optional(value.get(name_field)).strip(),
                        official_api_id=stringify_optional(value.get(id_field)).strip()
                        if id_field
                        else "",
                    )
                )
            return records

        if api_id_column:
            print(
                "WARNING: --api-id-column was ignored because this JSON catalog "
                "uses object keys as official API names.",
                file=sys.stderr,
            )
        return [CatalogRecord(official_api_name=str(key).strip()) for key in data]

    raise ValueError("API catalog must be a CSV, JSON object, or JSON array")


def build_api_name_entries(
    catalog_records: Iterable[CatalogRecord],
) -> tuple[list[ApiNameEntry], list[dict[str, Any]]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in catalog_records:
        if not record.official_api_name:
            continue
        normalized = normalize_text(record.official_api_name)
        if not normalized:
            continue
        bucket = grouped.setdefault(
            normalized,
            {"official_api_names": OrderedDict(), "official_api_ids": OrderedDict()},
        )
        bucket["official_api_names"].setdefault(record.official_api_name, None)
        if record.official_api_id:
            bucket["official_api_ids"].setdefault(record.official_api_id, None)

    entries: list[ApiNameEntry] = []
    duplicate_warnings: list[dict[str, Any]] = []
    for normalized, bucket in grouped.items():
        names = tuple(bucket["official_api_names"].keys())
        ids = tuple(bucket["official_api_ids"].keys())
        if len(names) > 1 or len(ids) > 1:
            warning = {
                "normalized_api_name": normalized,
                "official_api_names": list(names),
                "official_api_ids": list(ids),
            }
            duplicate_warnings.append(warning)
            print(
                "WARNING: duplicate normalized official API names: "
                f"normalized_api_name={normalized!r}, official_api_names={list(names)!r}, "
                f"official_api_ids={list(ids)!r}",
                file=sys.stderr,
            )
        entries.append(
            ApiNameEntry(
                normalized_api_name=normalized,
                tokens=tuple(normalized.split()),
                official_api_names=names,
                official_api_ids=ids,
            )
        )

    entries.sort(
        key=lambda entry: (
            -len(entry.tokens),
            entry.normalized_api_name,
            entry.official_api_names,
        )
    )
    return entries, duplicate_warnings


def build_first_token_index(
    entries: Iterable[ApiNameEntry],
) -> dict[str, list[ApiNameEntry]]:
    index: dict[str, list[ApiNameEntry]] = {}
    for entry in entries:
        if not entry.tokens:
            continue
        index.setdefault(entry.tokens[0], []).append(entry)

    for bucket in index.values():
        bucket.sort(
            key=lambda entry: (
                -len(entry.tokens),
                entry.normalized_api_name,
                entry.official_api_names,
            )
        )
    return index


def overlaps(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in ranges)


def locate_matches(
    text: str,
    first_token_index: dict[str, list[ApiNameEntry]],
    *,
    placeholder: str,
) -> list[ApiNameMatch]:
    tokens = normalized_tokens_with_spans(text)
    protected_ranges = [
        (match.start(), match.end())
        for match in re.finditer(re.escape(placeholder), text)
    ]
    raw_matches: list[ApiNameMatch] = []

    for token_index, token in enumerate(tokens):
        for entry in first_token_index.get(token.text, []):
            token_count = len(entry.tokens)
            end_token_index = token_index + token_count
            if end_token_index > len(tokens):
                continue
            if tuple(item.text for item in tokens[token_index:end_token_index]) != entry.tokens:
                continue

            start_char = tokens[token_index].start_char
            end_char = tokens[end_token_index - 1].end_char
            if overlaps(start_char, end_char, protected_ranges):
                continue

            surface = text[start_char:end_char]
            raw_matches.append(
                ApiNameMatch(
                    entry=entry,
                    matched_surface_form=surface,
                    normalized_matched_surface_form=normalize_text(surface),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

    selected: list[ApiNameMatch] = []
    for match in sorted(
        raw_matches,
        key=lambda item: (
            -len(item.entry.tokens),
            -(item.end_char - item.start_char),
            item.start_char,
            item.entry.normalized_api_name,
        ),
    ):
        if overlaps(
            match.start_char,
            match.end_char,
            ((item.start_char, item.end_char) for item in selected),
        ):
            continue
        selected.append(match)

    return sorted(selected, key=lambda item: (item.start_char, item.end_char))


def apply_matches(
    text: str,
    matches: Sequence[ApiNameMatch],
    *,
    placeholder: str,
) -> str:
    masked = text
    for match in sorted(matches, key=lambda item: item.start_char, reverse=True):
        masked = masked[: match.start_char] + placeholder + masked[match.end_char :]
    return masked


def match_to_log_row(
    record: TextRecord,
    match: ApiNameMatch,
) -> dict[str, Any]:
    return {
        "row_index": record.row_index,
        "text_id": record.text_id,
        "official_api_id": ";".join(match.entry.official_api_ids),
        "official_api_name": ";".join(match.entry.official_api_names),
        "normalized_api_name": match.entry.normalized_api_name,
        "matched_surface_form": match.matched_surface_form,
        "normalized_matched_surface_form": match.normalized_matched_surface_form,
        "start_char": match.start_char,
        "end_char": match.end_char,
    }


def text_from_list(value: list[Any], index: int, *, row_label: str) -> str:
    if index >= len(value) or index < -len(value):
        raise ValueError(f"Text index {index} is out of range for {row_label}")
    text = value[index]
    if not isinstance(text, str):
        raise ValueError(f"Text value is not a string for {row_label}")
    return text


def iter_text_records(
    bundle: InputBundle,
    *,
    text_column: str,
    text_id_column: str | None,
    list_text_index: int,
) -> list[TextRecord]:
    records: list[TextRecord] = []

    if bundle.input_format == "csv":
        assert bundle.csv_fieldnames is not None
        text_field = choose_field(
            bundle.csv_fieldnames,
            text_column,
            TEXT_FIELD_CANDIDATES,
            required=True,
            label="input text",
        )
        text_id_field = choose_field(
            bundle.csv_fieldnames,
            text_id_column,
            TEXT_ID_FIELD_CANDIDATES,
            required=False,
            label="text ID",
        )
        for row_index, row in enumerate(bundle.data):
            text = row.get(text_field, "")
            records.append(
                TextRecord(
                    row_index=row_index,
                    text_id=stringify_optional(row.get(text_id_field)).strip()
                    if text_id_field
                    else str(row_index),
                    text=text,
                    location_kind="csv",
                    key=row_index,
                    text_field=text_field,
                )
            )
        return records

    if bundle.input_format == "json_dict":
        for row_index, (key, value) in enumerate(bundle.data.items()):
            if isinstance(value, dict):
                text_field = choose_field(
                    value.keys(),
                    text_column,
                    TEXT_FIELD_CANDIDATES,
                    required=True,
                    label=f"input text for {key!r}",
                )
                text_id_field = choose_field(
                    value.keys(),
                    text_id_column,
                    TEXT_ID_FIELD_CANDIDATES,
                    required=False,
                    label=f"text ID for {key!r}",
                )
                text_value = value[text_field]
                if not isinstance(text_value, str):
                    raise ValueError(f"Text value is not a string for {key!r}")
                records.append(
                    TextRecord(
                        row_index=row_index,
                        text_id=stringify_optional(value.get(text_id_field)).strip()
                        if text_id_field
                        else str(key),
                        text=text_value,
                        location_kind="json_dict_value_dict",
                        key=key,
                        text_field=text_field,
                    )
                )
            elif isinstance(value, list):
                records.append(
                    TextRecord(
                        row_index=row_index,
                        text_id=str(key),
                        text=text_from_list(value, list_text_index, row_label=str(key)),
                        location_kind="json_dict_value_list",
                        key=key,
                    )
                )
            else:
                raise ValueError(f"Unsupported JSON object value for {key!r}")
        return records

    if bundle.input_format == "json_list":
        for row_index, value in enumerate(bundle.data):
            if isinstance(value, dict):
                text_field = choose_field(
                    value.keys(),
                    text_column,
                    TEXT_FIELD_CANDIDATES,
                    required=True,
                    label=f"input text for row {row_index}",
                )
                text_id_field = choose_field(
                    value.keys(),
                    text_id_column,
                    TEXT_ID_FIELD_CANDIDATES,
                    required=False,
                    label=f"text ID for row {row_index}",
                )
                text_value = value[text_field]
                if not isinstance(text_value, str):
                    raise ValueError(f"Text value is not a string for row {row_index}")
                records.append(
                    TextRecord(
                        row_index=row_index,
                        text_id=stringify_optional(value.get(text_id_field)).strip()
                        if text_id_field
                        else str(row_index),
                        text=text_value,
                        location_kind="json_list_dict",
                        key=row_index,
                        text_field=text_field,
                    )
                )
            elif isinstance(value, list):
                records.append(
                    TextRecord(
                        row_index=row_index,
                        text_id=str(row_index),
                        text=text_from_list(
                            value,
                            list_text_index,
                            row_label=f"row {row_index}",
                        ),
                        location_kind="json_list_list",
                        key=row_index,
                    )
                )
            elif isinstance(value, str):
                records.append(
                    TextRecord(
                        row_index=row_index,
                        text_id=str(row_index),
                        text=value,
                        location_kind="json_list_string",
                        key=row_index,
                    )
                )
            else:
                raise ValueError(f"Unsupported JSON array value at row {row_index}")
        return records

    raise ValueError(f"Unsupported input format: {bundle.input_format}")


def list_record_to_dict(value: list[Any]) -> dict[str, Any]:
    default_names = ("categories", "description", "apis", "id")
    converted: dict[str, Any] = {}
    for index, item in enumerate(value):
        field_name = default_names[index] if index < len(default_names) else f"field_{index}"
        converted[field_name] = item
    return converted


def add_mask_fields(
    row: dict[str, Any],
    *,
    original_text: str,
    masked_text: str,
    num_masked: int,
) -> None:
    row["original_text"] = original_text
    row["masked_text"] = masked_text
    row["num_masked_api_mentions"] = num_masked


def store_mask_result(
    output_data: Any,
    record: TextRecord,
    *,
    original_text: str,
    masked_text: str,
    num_masked: int,
    overwrite_text_column: bool,
    legacy_list_output: bool,
    list_text_index: int,
) -> None:
    if record.location_kind == "csv":
        row = output_data[record.key]
        if overwrite_text_column:
            row[record.text_field] = masked_text
        add_mask_fields(
            row,
            original_text=original_text,
            masked_text=masked_text,
            num_masked=num_masked,
        )
        return

    if record.location_kind in {"json_dict_value_dict", "json_list_dict"}:
        row = output_data[record.key]
        if overwrite_text_column:
            row[record.text_field] = masked_text
        add_mask_fields(
            row,
            original_text=original_text,
            masked_text=masked_text,
            num_masked=num_masked,
        )
        return

    if record.location_kind in {"json_dict_value_list", "json_list_list"}:
        value = output_data[record.key]
        if legacy_list_output:
            if not overwrite_text_column:
                raise ValueError("--legacy-list-output requires --overwrite-text-column")
            value[list_text_index] = masked_text
            return

        converted = list_record_to_dict(value)
        if overwrite_text_column:
            known_text_field = "description" if list_text_index == 1 else f"field_{list_text_index}"
            converted[known_text_field] = masked_text
        add_mask_fields(
            converted,
            original_text=original_text,
            masked_text=masked_text,
            num_masked=num_masked,
        )
        output_data[record.key] = converted
        return

    if record.location_kind == "json_list_string":
        if legacy_list_output:
            if not overwrite_text_column:
                raise ValueError("--legacy-list-output requires --overwrite-text-column")
            output_data[record.key] = masked_text
            return

        output_data[record.key] = {
            "text": masked_text if overwrite_text_column else original_text,
            "original_text": original_text,
            "masked_text": masked_text,
            "num_masked_api_mentions": num_masked,
        }
        return

    raise ValueError(f"Unsupported record location: {record.location_kind}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_output_data(path: Path, bundle: InputBundle, output_data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundle.input_format == "csv":
        assert bundle.csv_fieldnames is not None
        fieldnames = list(bundle.csv_fieldnames)
        for field in MASKED_DATA_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)
        write_csv(path, output_data, fieldnames)
        return
    write_json(path, output_data)


def normalization_steps() -> list[str]:
    return [
        "HTML unescape",
        "Unicode NFKC",
        "lowercase",
        "replace non-ASCII-alphanumeric characters with spaces",
        "collapse repeated spaces",
        "strip leading and trailing spaces",
    ]


def resolved_dictionary_json(entries: Sequence[ApiNameEntry], *, placeholder: str) -> dict[str, Any]:
    return {
        "version": "official-api-name-only-v1",
        "placeholder": placeholder,
        "normalization": normalization_steps(),
        "matching_rule": "exact match against consecutive normalized tokens",
        "manual_aliases_used": False,
        "semantic_completion_used": False,
        "entry_count": len(entries),
        "entries": [
            {
                "official_api_id": ";".join(entry.official_api_ids),
                "official_api_name": ";".join(entry.official_api_names),
                "normalized_api_name": entry.normalized_api_name,
            }
            for entry in entries
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mask official API names in test text using deterministic normalized "
            "token matching."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", "--api-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--text-column", default="description")
    parser.add_argument("--text-id-column")
    parser.add_argument("--api-name-column")
    parser.add_argument("--api-id-column")
    parser.add_argument("--list-text-index", type=int, default=1)
    parser.add_argument("--placeholder", default=PLACEHOLDER)
    parser.add_argument("--mask-log-csv", type=Path, default=DEFAULT_MASK_LOG_CSV)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--resolved-dictionary",
        type=Path,
        default=DEFAULT_RESOLVED_DICTIONARY,
    )
    parser.add_argument(
        "--overwrite-text-column",
        action="store_true",
        help="Also replace the original input text column/list slot with masked text.",
    )
    parser.add_argument(
        "--legacy-list-output",
        action="store_true",
        help=(
            "Keep JSON list records as lists. This requires --overwrite-text-column "
            "because list records cannot receive added fields."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dictionary",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--full-catalog",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dictionary:
        print(
            "WARNING: --dictionary is ignored; this implementation uses only "
            "official API names from --catalog.",
            file=sys.stderr,
        )
    if args.full_catalog:
        print(
            "WARNING: --full-catalog is ignored; this implementation uses only "
            "official API names from --catalog.",
            file=sys.stderr,
        )

    catalog_records = load_catalog(
        args.catalog,
        api_name_column=args.api_name_column,
        api_id_column=args.api_id_column,
    )
    entries, duplicate_warnings = build_api_name_entries(catalog_records)
    first_token_index = build_first_token_index(entries)
    bundle = load_input(args.input)
    records = iter_text_records(
        bundle,
        text_column=args.text_column,
        text_id_column=args.text_id_column,
        list_text_index=args.list_text_index,
    )
    output_data = copy.deepcopy(bundle.data)
    mask_log_rows: list[dict[str, Any]] = []
    masked_row_count = 0

    for record in records:
        matches = locate_matches(
            record.text,
            first_token_index,
            placeholder=args.placeholder,
        )
        masked_text = apply_matches(
            record.text,
            matches,
            placeholder=args.placeholder,
        )
        if matches:
            masked_row_count += 1

        store_mask_result(
            output_data,
            record,
            original_text=record.text,
            masked_text=masked_text,
            num_masked=len(matches),
            overwrite_text_column=args.overwrite_text_column,
            legacy_list_output=args.legacy_list_output,
            list_text_index=args.list_text_index,
        )
        mask_log_rows.extend(match_to_log_row(record, match) for match in matches)

    summary = {
        "total_rows": len(records),
        "masked_rows": masked_row_count,
        "unmasked_rows": len(records) - masked_row_count,
        "total_masked_api_mentions": len(mask_log_rows),
        "catalog_records": len(catalog_records),
        "normalized_api_name_entries": len(entries),
        "duplicate_normalized_api_name_count": len(duplicate_warnings),
        "manual_aliases_used": False,
        "semantic_completion_used": False,
        "text_column_overwritten": bool(args.overwrite_text_column),
        "legacy_list_output": bool(args.legacy_list_output),
    }
    audit = {
        "version": "official-api-name-only-v1",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "api_catalog": str(args.catalog.resolve()),
        "mask_log_csv": str(args.mask_log_csv.resolve()),
        "resolved_dictionary": str(args.resolved_dictionary.resolve()),
        "placeholder": args.placeholder,
        "normalization": normalization_steps(),
        "matching_rule": "exact match against consecutive normalized tokens",
        "summary": summary,
        "duplicate_normalized_api_name_warnings": duplicate_warnings,
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if duplicate_warnings:
            print(
                json.dumps(
                    {"duplicate_normalized_api_name_warnings": duplicate_warnings},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    write_output_data(args.output, bundle, output_data)
    write_csv(args.mask_log_csv, mask_log_rows, MASK_LOG_FIELDS)
    write_json(args.resolved_dictionary, resolved_dictionary_json(entries, placeholder=args.placeholder))
    audit["sha256"] = {
        "input": sha256(args.input),
        "output": sha256(args.output),
        "api_catalog": sha256(args.catalog),
        "mask_log_csv": sha256(args.mask_log_csv),
        "resolved_dictionary": sha256(args.resolved_dictionary),
    }
    write_json(args.audit, audit)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
