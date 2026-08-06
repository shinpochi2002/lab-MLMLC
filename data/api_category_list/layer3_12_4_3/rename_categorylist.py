#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path


PATTERN_PREFIX = re.compile(r"^softfill_inc_12big_4mid_categorylist_")
PATTERN_BIG_ALL = re.compile(r"^big12\.json$")
PATTERN_MID_ALL = re.compile(r"^big12_mid(\d+)\.json$")
PATTERN_MID_PART = re.compile(r"^big12_mid(\d+)-(\d+)\.json$")


def convert_name(src_name: str) -> str | None:
    """
    Convert:
      softfill_inc_12big_4mid_categorylist_big12.json
        -> softfill_inc_categorylist_layer3_12.json

      softfill_inc_12big_4mid_categorylist_big12_midX.json
        -> softfill_inc_categorylist_layer3_12_X.json

      softfill_inc_12big_4mid_categorylist_big12_midX-Y.json
        -> softfill_inc_categorylist_layer3_12_X_X_Y.json
    """
    base = PATTERN_PREFIX.sub("", src_name)

    if PATTERN_BIG_ALL.match(base):
        return "softfill_inc_categorylist_layer3_12.json"

    m = PATTERN_MID_ALL.match(base)
    if m:
        x = m.group(1)
        return f"softfill_inc_categorylist_layer3_12_{x}.json"

    m = PATTERN_MID_PART.match(base)
    if m:
        x, y = m.group(1), m.group(2)
        return f"softfill_inc_categorylist_layer3_12_{x}_{x}_{y}.json"

    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rename categorylist jsons from big12_midX(-Y) style to layer3_12 style."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations only (no changes).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination if it exists (default: skip).",
    )
    ap.add_argument(
        "--glob",
        default="softfill_inc_12big_4mid_categorylist_*.json",
        help="Source file glob (default: %(default)s)",
    )
    args = ap.parse_args()

    src_files = sorted(Path(".").glob(args.glob))
    if not src_files:
        print(f"[ERROR] No files matched: {args.glob}", file=sys.stderr)
        return 2

    planned = []
    skipped_unmatched = 0

    for src in src_files:
        new_name = convert_name(src.name)
        if new_name is None:
            skipped_unmatched += 1
            continue
        dst = src.with_name(new_name)
        planned.append((src, dst))

    if not planned:
        print("[ERROR] No rename targets found. Check patterns/prefix.", file=sys.stderr)
        return 2

    # Show plan
    for src, dst in planned:
        if dst.exists() and not args.force:
            print(f"SKIP (exists): {src.name} -> {dst.name}")
        else:
            print(f"RENAME: {src.name} -> {dst.name}")

    if skipped_unmatched:
        print(f"[INFO] Unmatched files skipped: {skipped_unmatched}")

    if args.dry_run:
        print("[INFO] dry-run: no changes made.")
        return 0

    # Execute
    renamed = 0
    skipped_exists = 0

    for src, dst in planned:
        if dst.exists() and not args.force:
            skipped_exists += 1
            continue
        try:
            if args.force and dst.exists():
                dst.unlink()
            os.rename(src, dst)
            renamed += 1
        except Exception as e:
            print(f"[ERROR] Failed: {src} -> {dst} ({e})", file=sys.stderr)
            return 1

    print(f"[DONE] renamed={renamed}, skipped_exists={skipped_exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
