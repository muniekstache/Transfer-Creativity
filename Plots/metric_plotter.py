#!/usr/bin/env python

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt



# Which model folders to plot
RUN_FOLDERS = [
    "en-de_v_prime_finetuned_on_creative",
    "en-de_v_prime_finetuned_on_less-creative",
    "en-fr_v_prime_finetuned_on_creative",
    "en-fr_v_prime_finetuned_on_less-creative",
]


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
RUNS_BASE  = ROOT_DIR / "Finetuning_experiments" / "results"

CKPT_RE = re.compile(r"checkpoint-(\d+)")  # simple regex that captures all checkpoints


def latest_checkpoint(run_dir: Path) -> Path | None:
    best, step_best = None, -1
    for child in run_dir.iterdir():
        m = CKPT_RE.fullmatch(child.name)
        if m and child.is_dir():
            step = int(m.group(1))
            if step > step_best:
                step_best, best = step, child
    return best


def extract_metric_history(ts_path: Path):
    """Return {'BLEU': [(epoch,val)…], 'chrF': …, 'COMET': …}."""
    with ts_path.open(encoding="utf-8") as fh:
        state = json.load(fh)

    mapping = {"eval_bleu": "BLEU", "eval_chrf": "chrF", "eval_comet": "COMET"}
    history = {v: [] for v in mapping.values()}

    for log in state["log_history"]:
        if "epoch" not in log:
            continue
        for raw, pretty in mapping.items():
            if raw in log:
                history[pretty].append((log["epoch"], log[raw]))
    return history


def plot_group(metric, runs_dict):
    """Draw one figure with all models for a given metric."""
    colors = ["blue", "red", "yellow", "green"]
    plt.figure()

    for idx, (run, series) in enumerate(sorted(runs_dict.items())):
        epochs, vals = zip(*series)
        plt.plot(
            epochs,
            vals,
            marker="o",
            label=run.replace("_v_prime_finetuned_on_", "\n"),
            color=colors[idx],
        )

    plt.title(f"{metric} per epoch")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True, linewidth=0.3)
    plt.legend(fontsize=8)
    out_file = SCRIPT_DIR / f"{metric.lower()}_all_runs.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()
    print(f"saved -> {out_file}")



# main
def main() -> None:
    print(f"Plots directory  : {SCRIPT_DIR}")
    print(f"Models base folder : {RUNS_BASE}\n")

    all_runs = {m: {} for m in ("BLEU", "chrF", "COMET")}

    for run_name in RUN_FOLDERS:
        run_dir = RUNS_BASE / run_name
        if not run_dir.is_dir():
            print(f"[skip] model folder not found: {run_dir}")
            continue

        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoints in {run_dir}")
            continue

        ts_file = ckpt / "trainer_state.json"
        if not ts_file.is_file():
            print(f"[skip] trainer_state.json missing in {ckpt}")
            continue

        hist = extract_metric_history(ts_file)
        for metric, series in hist.items():
            if series:
                all_runs[metric][run_name] = series
            else:
                print(f"[warn] {metric} missing in {run_name}")

    # create one plot per metric
    for metric, runs in all_runs.items():
        if runs:
            plot_group(metric, runs)

    print("\nDone.")


if __name__ == "__main__":
    main()
