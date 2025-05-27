from pathlib import Path
import xml.etree.ElementTree as ET
import json
import argparse
import random

# Allowed text types for filtering
ALLOWED_TEXT_TYPES = {
    'Fictional literature',
    'Non-fictional literature',
    'Journalistic texts',
    'Instructive texts',
    'Administrative texts',
    'External Communication'
}

# TEI namespace for parsing both alignment and monolingual files
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


def find_dpc_files(root_dir, text_type='Fictional literature'):
    """
    Traverse the DPC core directory and build a dictionary of document entries
    for metadata files matching the given text type.

    Args:
        root_dir (str or Path): Path to the root DPC data directory.
        text_type (str): One of the allowed text types (case-sensitive).

    Returns:
        dict: A mapping from document base names (e.g., 'dpc-xxx-000000') to
              a dict with keys:
                - 'alignment': Path to the alignment index file
                - 'dutch': Path to the Dutch monolingual TEI file
                - 'english': Path to the English monolingual TEI file
                - 'aligned_sents': an empty list to be populated later

    Raises:
        ValueError: If text_type is not in ALLOWED_TEXT_TYPES.
    """
    if text_type not in ALLOWED_TEXT_TYPES:
        raise ValueError(
            f"Invalid text_type '{text_type}'. Must be one of: {', '.join(sorted(ALLOWED_TEXT_TYPES))}."
        )

    root = Path(root_dir)
    result = {}

    # Find all English metadata files and filter by TextType
    for meta_path in root.rglob('*-en-mtd.xml'):
        try:
            tree = ET.parse(meta_path)
            xml_root = tree.getroot()
            text_type_elem = xml_root.find('.//TextType')

            if text_type_elem is not None and text_type_elem.text == text_type:
                base = meta_path.stem[:-len('-en-mtd')]
                result[base] = {
                    'alignment': meta_path.with_name(f"{base}-nl-en-tei.xml"),
                    'dutch': meta_path.with_name(f"{base}-nl-tei.xml"),
                    'english': meta_path.with_name(f"{base}-en-tei.xml"),
                    'aligned_sents': []
                }
        except ET.ParseError as e:
            print(f"Warning: failed to parse XML file {meta_path}: {e}")

    return result


def extract_alignments(dpc_dict):
    """
    Parse each document's alignment index file and extract
    'A:1-1' link targets into the 'aligned_sents' list.

    Args:
        dpc_dict (dict): The dictionary returned by find_dpc_files.

    Returns:
        dict: The same dict with 'aligned_sents' populated.
    """
    for base, info in dpc_dict.items():
        aln_path = info['alignment']

        if aln_path and aln_path.exists():
            try:
                tree = ET.parse(aln_path)
                root = tree.getroot()

                # Find all <link> elements in the TEI namespace
                for link in root.findall('.//tei:link', NS):
                    # Normalize and check type attribute
                    if link.get('type', '').replace(' ', '') == 'A:1-1':
                        targets = link.get('targets')
                        if targets:
                            info['aligned_sents'].append(targets)
            except ET.ParseError as e:
                print(f"Warning: failed to parse alignment file {aln_path}: {e}")

    return dpc_dict


def extract_sentences(dpc_dict, output_dir, text_type, max_entries=None):
    """
    For each document entry, extract aligned sentence pairs from the
    monolingual TEI files and write both per-document and aggregate JSON.

    Args:
        dpc_dict (dict): Dictionary with alignment and file paths.
        output_dir (str or Path): Directory to write JSON output.

    returns:
        list: A flat list of translation objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = []

    for base, info in dpc_dict.items():
        # Parse monolingual TEI files
        en_tree = ET.parse(info['english'])
        nl_tree = ET.parse(info['dutch'])
        en_root = en_tree.getroot()
        nl_root = nl_tree.getroot()

        translations = []
        for target_pair in info['aligned_sents']:
            # Split 'p1.s1; p1.s1' into NL and EN IDs
            parts = [p.strip() for p in target_pair.split(';')]
            if len(parts) != 2:
                continue
            nl_id, en_id = parts

            # Locate the sentence <seg> by its 'n' attribute
            nl_seg = nl_root.find(
                f".//tei:seg[@type='sent'][@n='seg.{nl_id}']", NS
            )
            en_seg = en_root.find(
                f".//tei:seg[@type='sent'][@n='seg.{en_id}']", NS
            )
            if nl_seg is None or en_seg is None:
                continue

            # Extract the original text from the child <seg type="original">
            nl_text_elem = nl_seg.find("tei:seg[@type='original']", NS)
            en_text_elem = en_seg.find("tei:seg[@type='original']", NS)
            if nl_text_elem is None or en_text_elem is None:
                continue

            nl_text = (nl_text_elem.text or '').strip()
            en_text = (en_text_elem.text or '').strip()

            translation_obj = {'translation': {'en': en_text, 'nl': nl_text}}
            translations.append(translation_obj)
            aggregate.append(translation_obj)

        # Write per-document JSON file
        doc_file = output_dir / f"{text_type}-{base}.json"
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(translations)} sentence pairs to {doc_file}")

    # Write aggregate JSON file
    agg_file = output_dir / f"{text_type}-aggregate.json"
    with open(agg_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"Wrote aggregate file with {len(aggregate)} total translations to {agg_file}")
    # If requested, randomly sample up to max_entries
    if max_entries is not None and len(aggregate) > max_entries:
        aggregate = random.sample(aggregate, max_entries)
 
    with open(agg_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"Wrote aggregate file with {len(aggregate)} total translations to {agg_file}")

    return aggregate


def main():
    """
    Entry point: parse command-line arguments, run extraction pipeline,
    and handle errors.
    """
    parser = argparse.ArgumentParser(
        description='Extract aligned sentences to JSON'
    )
    parser.add_argument(
        '--text_type', type=str, choices=sorted(ALLOWED_TEXT_TYPES),
        default='Fictional literature',
        help='One of the allowed text types'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to write JSON files (default: project_root/output_json)'
    )
    parser.add_argument(
        '--max_entries', type=int, default=None,
        help='Maximum number of sentence-pairs to include in the aggregate JSON'
    )
    
    args = parser.parse_args()

    script_dir = Path(__file__).parent  # directory containing this script
    project_root = script_dir.parent     # parent of corpus_creator
    core_dir = project_root / 'data' / 'DPC' / 'data' / 'core'
    out_dir = Path(args.output_dir) if args.output_dir else project_root / 'Data' / 'finetuning'
    text_type = args.text_type

    try:
        # Build dict of relevant DPC files
        dpc_dict = find_dpc_files(core_dir, text_type=text_type)
        # Extract alignment indices
        dpc_dict = extract_alignments(dpc_dict)
        # Extract sentences and write JSON
        extract_sentences(dpc_dict, out_dir, text_type = text_type, max_entries= args.max_entries)
    except ValueError as err:
        print(err)


if __name__ == '__main__':
    main()
