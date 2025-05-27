#!/usr/bin/env python3
import sys
import json
import random
from pathlib import Path

def choose_file(prompt, files):
    print(prompt)
    for i, p in enumerate(files, start=1):
        print(f"  [{i}] {p.name}")
    choice = input(f"Select 1–{len(files)}: ").strip()
    if not choice.isdigit() or not (1 <= (idx := int(choice)) <= len(files)):
        print("Invalid selection.", file=sys.stderr)
        sys.exit(1)
    return files[idx - 1]

def main():
    # locate project root and Data subfolders
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir     = project_root / "Data"
    ft_dir       = data_dir / "finetuning"
    opus_dir     = data_dir / "Opus_opensubtitles"

    # collect JSON candidates
    ft_files   = sorted(ft_dir.glob("*.json"))
    opus_files = sorted(opus_dir.glob("*.json"))

    if not ft_files or not opus_files:
        print("Could not find JSON files in one of the folders.", file=sys.stderr)
        sys.exit(1)

    # let user pick
    ft_path   = choose_file("Finetuning JSON files:", ft_files)
    opus_path = choose_file("Opus JSON files:", opus_files)

    # load
    try:
        with ft_path.open(encoding="utf-8") as f:
            ft_data = json.load(f)
        with opus_path.open(encoding="utf-8") as f:
            opus_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(ft_data, list) or not isinstance(opus_data, list):
        print("One of the files does not contain a JSON list.", file=sys.stderr)
        sys.exit(1)

    # combine & shuffle
    combined = ft_data + opus_data
    random.shuffle(combined)

    # output
    default_name = "combined_shuffled.json"
    out_name = input(f"Output filename [{default_name}]: ").strip() or default_name
    out_path = data_dir / "combined" / out_name

    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Wrote {len(combined)} items to {out_path}")

if __name__ == "__main__":
    main()
