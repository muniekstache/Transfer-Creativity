# Transferring Creativity Thesis

For now only a lightweight set of Python utilities for collecting, cleaning and assembling **English ↔ Dutch** sentence pairs. Eventually this will include a finetuning pipeline.

---

## Repository layout

```text
.
├── corpus_creation/
│   ├── combine_n_shuffle.py
│   ├── dpc_corpus_creator.py
│   └── Opus_corpus_creator.py
├── Data/
│   ├── DPC                 # local version of DPC corpus
│   ├── finetuning/         # output from dpc_corpus_creator
│   ├── Opus_opensubtitles/ # output from Opus_corpus_creator
│   └── combined/           # output from combine_n_shuffle
├── requirements.txt
└── README.md
```


## Requirements

* Python ≥ 3.9  
* **opustools** ≥ 1 .0 (required only by `Opus_corpus_creator.py`)
* The **DPC core dataset** (Dutch Parallel Corpus) extracted to `data/DPC/data/core/` for `dpc_corpus_creator.py`.
* NLTK **punkt_tab** sentence tokenizer (`nltk.download('punkt_tab')` within python interpreter)

Install everything in one go:

```bash
python -m pip install -r requirements.txt
```

---

## Quick-start workflow

| Step | Command | What it does |
|------|---------|--------------|
| 1. Extract DPC pairs | `python corpus_creation/dpc_corpus_creator.py --text_type "Fictional literature" --max_entries 10000` | Filters the DPC metadata, pulls **1-to-1** sentence links and writes JSON files to `Data/finetuning/`. |
| 2. Fetch OpenSubtitles pairs | `python corpus_creation/Opus_corpus_creator.py` | Downloads extra English–Dutch pairs (up to 10 k) via **OpusTools** into `Data/Opus_opensubtitles/opensubtitles_supplement_en_nl.json`. |
| 3. Merge & shuffle | `python corpus_creation/combine_n_shuffle.py` | Lets you pick one JSON file per source, concatenates and shuffles them, then writes `Data/combined/combined_shuffled.json`. |
| 4. Fine-tune | Point your training script at `Data/combined/combined_shuffled.json` | File uses the standard *translation* schema: see example below. |

Example JSON entry:

```json
{
  "translation": {
    "en": "Good morning.",
    "nl": "Goedemorgen."
  }
}
```

---

## Command-line reference

| Script | Key options (`-h/--help`) |
|--------|---------------------------|
| **dpc_corpus_creator.py** | `--text_type {Fictional literature,…}` · `--output_dir DIR` · `--max_entries N` |
| **Opus_corpus_creator.py** | _(no flags)_ |
| **combine_n_shuffle.py** | _(interactive prompts only)_ |


