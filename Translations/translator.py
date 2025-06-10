#!/usr/bin/env python

import pathlib, re, torch, pandas as pd, nltk
from transformers import MarianTokenizer, MarianMTModel


# Paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
RUNS_DIR   = ROOT_DIR / "Finetuning_experiments" / "results"
STORY_FILE = ROOT_DIR / "Data" / "2B0R2B" / "2B0R2B.txt"

RUN_FOLDERS = [
    "en-de_v_prime_finetuned_on_creative",
    "en-de_v_prime_finetuned_on_instructive",
    "en-fr_v_prime_finetuned_on_creative",
    "en-fr_v_prime_finetuned_on_instructive",
]


# Prepare source: paragraph list & per-paragraph sentence lists

raw = STORY_FILE.read_text(encoding="utf-8")
paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

# sentence tokeniser for English
sent_tok = nltk.data.load("tokenizers/punkt/english.pickle")
para_sent_src = [sent_tok.tokenize(p) for p in paragraphs]
flat_src_sents = [s for chunk in para_sent_src for s in chunk]


# Translation loop (sentence-level batches)
for run in RUN_FOLDERS:
    model_dir = RUNS_DIR / run
    print(f"\nTranslating with {run}")

    tok = MarianTokenizer.from_pretrained(model_dir)
    mt  = MarianMTModel.from_pretrained(model_dir).to("cuda")

    tgt_sents, batch_size = [], 16
    for i in range(0, len(flat_src_sents), batch_size):
        batch_src = flat_src_sents[i : i + batch_size]
        batch = tok(batch_src,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256).to("cuda")
        with torch.no_grad():
            out = mt.generate(
                **batch,
                num_beams=5,
                length_penalty=1.0,
                max_length=256,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
            )
        tgt_sents.extend(tok.batch_decode(out, skip_special_tokens=True))

    # Excel (1 sentence per row)
    df = pd.DataFrame({"Source sentence": flat_src_sents,
                       "Target sentence": tgt_sents})
    xlsx = SCRIPT_DIR / f"{run}_2B0R2B.xlsx"
    df.to_excel(xlsx, index=False)
    print(f"  • {xlsx.name}")

    # Plain-text story (paragraph grouping restored)
    idx, tgt_paragraphs = 0, []
    for chunk in para_sent_src:
        tgt_paragraphs.append(" ".join(tgt_sents[idx : idx + len(chunk)]))
        idx += len(chunk)
    txt = SCRIPT_DIR / f"{run}_2B0R2B_nl.txt"
    txt.write_text("\n\n".join(tgt_paragraphs), encoding="utf-8")
    print(f"  - {txt.name}")

print("\nDone.")