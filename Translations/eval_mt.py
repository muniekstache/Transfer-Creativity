#!/usr/bin/env python3

import pathlib
import pandas as pd
import sacrebleu
from comet import download_model, load_from_checkpoint

WORKBOOK_PATH = pathlib.Path("aggregate_evaluations.xlsx")

ckpt = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(ckpt)


def bleu(sys_out, refs):
    return sacrebleu.corpus_bleu(sys_out, [refs]).score


def comet(sys_out, refs, src):
    """Return system-level COMET score as float, robust to COMET version."""
    data = [{"src": s, "mt": mt, "ref": ref} for s, mt, ref in zip(src, sys_out, refs)]
    out = comet_model.predict(data, batch_size=8)
    sys_score = out["system_score"]

    return float(sys_score)


def main():
    wb = pd.ExcelFile(WORKBOOK_PATH)
    results = []

    for sheet in wb.sheet_names:
        df = wb.parse(sheet)

        src = df["Source sentence"].astype(str).tolist()
        transl = df["Translation sentence"].astype(str).tolist()
        refs = df["Human Translation"].astype(str).tolist()

        results.append(
            {
                "Sheet": sheet,
                "BLEU": bleu(transl, refs),
                "COMET": comet(transl, refs, src),
            }
        )

    print("\n=== Evaluation summary ===")
    for r in results:
        print(f"{r['Sheet']}  BLEU: {r['BLEU']:.3f}   COMET: {r['COMET']:.3f}")


if __name__ == "__main__":
    main()
