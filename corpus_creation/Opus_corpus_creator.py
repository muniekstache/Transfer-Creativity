import sys
import random
import json
from pathlib import Path
from opustools import OpusRead

def get_opensubtitles_pairs(num_pairs, fetch_multiplier=5):
    """
    Fetch up to num_pairs random English–Dutch sentence pairs from OpenSubtitles.
    """
    cache_dir = Path('opus_data')
    cache_dir.mkdir(parents=True, exist_ok=True)

    en_file = cache_dir / 'opensubs.en'
    nl_file = cache_dir / 'opensubs.nl'
    max_fetch = num_pairs * fetch_multiplier
    download_dir = str(cache_dir.resolve())

    # Extract aligned sentences in Moses format
    reader = OpusRead(
        directory='OpenSubtitles',
        source='en',
        target='nl',
        release='v2024',
        preprocess='xml',
        write_mode='moses',
        write=[str(en_file), str(nl_file)],
        src_range='1',
        tgt_range='1',
        maximum=max_fetch,
        leave_non_alignments_out=True,
        suppress_prompts=True,
        download_dir=download_dir,
        verbose=False
    )
    reader.printPairs()

    # Verify output files exist
    if not en_file.exists() or not nl_file.exists():
        en_file.unlink(missing_ok=True)
        nl_file.unlink(missing_ok=True)
        raise RuntimeError("Failed to produce temporary output files.")

    # Read and build pair list
    pairs = []
    with en_file.open(encoding='utf-8') as fe, nl_file.open(encoding='utf-8') as fn:
        for en_line, nl_line in zip(fe, fn):
            en_text, nl_text = en_line.strip(), nl_line.strip()
            if en_text and nl_text:
                pairs.append({'translation': {'en': en_text, 'nl': nl_text}})
                
    # De-duplicate the fetched pairs
    print(f"De-duplicating {len(pairs)} fetched pairs...")
    
    seen_pairs = set()
    unique_pairs = []
    for entry in pairs:
        # Create a hashable representation of the translation pair
        en_text = entry['translation']['en']
        nl_text = entry['translation']['nl']
        text_pair = (en_text, nl_text)

        # Add to list only if this exact text pair has not been seen
        if text_pair not in seen_pairs:
            seen_pairs.add(text_pair)
            unique_pairs.append(entry)

    duplicates_found = len(pairs) - len(unique_pairs)
    print(f"Removed {duplicates_found} duplicate pairs, {len(unique_pairs)} unique pairs remain.")

    # Clean up temp files immediately
    en_file.unlink(missing_ok=True)
    nl_file.unlink(missing_ok=True)

    random.shuffle(pairs)
    return pairs[:num_pairs]

def main():
    current_pairs = 5000  # Set to lower amount so that combine_n_shuffle has enough sentences to grab unique ones
    target_pairs = 10000
    to_add = target_pairs - current_pairs

    if to_add <= 0:
        print(f"You already have {current_pairs} pairs (target: {target_pairs}). Nothing to do.")
        return

    print(f"Fetching {to_add} new pairs from OpenSubtitles…")
    try:
        new_pairs = get_opensubtitles_pairs(to_add)
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

    if not new_pairs:
        print("No sentence pairs were extracted; exiting.")
        return

    # Determine project root and target data directory
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    out_dir = project_root / 'Data' / 'Opus_opensubtitles'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / 'opensubtitles_supplement_en_nl.json'
    try:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(new_pairs, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✔ Wrote {len(new_pairs)} pairs to {out_path}")

if __name__ == '__main__':
    main()
