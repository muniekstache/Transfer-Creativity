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
    ft_dir       = data_dir / "dpc_json"
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
    
    # De-duplicate the combined list
    print(f"\nCombined list has {len(combined)} entries before de-duplication.")
    
    target_size = 10000
    
    # Get unique entries from the base finetuning file
    print(f"Base file has {len(ft_data)} entries. Finding unique entries...")
    seen_pairs = set()
    unique_ft_entries = []
    for entry in ft_data:
        en_text = entry.get('translation', {}).get('en', '')
        nl_text = entry.get('translation', {}).get('nl', '')
        text_pair = (en_text, nl_text)
        if text_pair not in seen_pairs:
            seen_pairs.add(text_pair)
            unique_ft_entries.append(entry)
    
    print(f"Found {len(unique_ft_entries)} unique entries in the base file.")

    # Calculate how many supplementary entries are needed
    needed = target_size - len(unique_ft_entries)

    if needed <= 0:
        print("Base file already has 10,000 or more unique entries. Taking a sample.")
        random.shuffle(unique_ft_entries)
        final_list = unique_ft_entries[:target_size]
    else:
        print(f"Need to add {needed} more unique entries.")
        
        # Gather unique candidates from the supplementary opus file
        opus_candidates = []
        for entry in opus_data:
            en_text = entry.get('translation', {}).get('en', '')
            nl_text = entry.get('translation', {}).get('nl', '')
            text_pair = (en_text, nl_text)
            # Add only if it's not a duplicate from the base file or within opus itself
            if text_pair not in seen_pairs:
                seen_pairs.add(text_pair)
                opus_candidates.append(entry)
        
        print(f"Found {len(opus_candidates)} unique candidates in the supplement file.")

        # Check if we have enough candidates and combine
        if len(opus_candidates) < needed:
            print(f"Warning: Not enough unique candidates ({len(opus_candidates)}) to reach {target_size}. The final file will be smaller.", file=sys.stderr)
        
        random.shuffle(opus_candidates)
        supplement = opus_candidates[:needed]
        final_list = unique_ft_entries + supplement

    # Final shuffle of the combined list
    random.shuffle(final_list)

    # output
    default_name = "combined_shuffled.json"
    out_name = input(f"Output filename [{default_name}]: ").strip() or default_name
    out_path = data_dir / "finetuning" / f"{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(final_list, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Wrote {len(final_list)} items to {out_path}")

if __name__ == "__main__":
    main()
