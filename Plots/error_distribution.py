#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ERROR_TYPES = [
    "No error",
    "Minor error",
    "Major error",
    "Critical error",
]

BIN_EDGES = [-np.inf,5, 10, 15, 20,
             25, 30, 35, 40, np.inf,
             ]
BIN_LABELS = [
    "≤5", "6-10", "11-15", "16-20", "21-25",
    "26-30", "31-35", "36-40", "41+",
]

BAR_WIDTH = 0.18


def parse_error_type(value):
    """Extract the part before the colon in an 'Error points' cell."""
    return str(value).split(":")[0].strip()


def read_and_prepare(excel_path):
    """Return a dict {sheet_name: cleaned DataFrame}."""
    xls = pd.ExcelFile(excel_path)
    dataframes = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df["ErrorType"] = df["Error points"].apply(parse_error_type)
        df["WordCount"] = df["Source sentence"].astype(str).str.split().str.len()
        df = df[df["ErrorType"].isin(ERROR_TYPES)]

        df["Bin"] = pd.cut(
            df["WordCount"],
            bins=BIN_EDGES,
            labels=BIN_LABELS,
            include_lowest=True,
        )
        dataframes[sheet] = df

    return dataframes


def compute_counts(dataframes):
    """Build nested dict error_type → sheet → list[count per bin]."""
    counts = {
        err: {sheet: [0] * len(BIN_LABELS) for sheet in dataframes}
        for err in ERROR_TYPES
    }

    for sheet, df in dataframes.items():
        for err in ERROR_TYPES:
            freq = (
                df[df["ErrorType"] == err]["Bin"]
                .value_counts()
                .reindex(BIN_LABELS, fill_value=0)
            )
            counts[err][sheet] = freq.tolist()
    return counts


def plot_counts(counts, sheet_names, bar_width=BAR_WIDTH):
    """Render the four‑subplot bar chart."""
    
    cmap = plt.get_cmap("Set2") #  best color scheme imo
    colours = [cmap(i) for i in range(len(sheet_names))]
    
    num_bins = len(BIN_LABELS)
    x = np.arange(num_bins)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="white")
    axes = axes.flatten()

    for idx, err in enumerate(ERROR_TYPES):
        ax = axes[idx]
        for i, sheet in enumerate(sheet_names):
            offset = (i - len(sheet_names) / 2) * bar_width + bar_width / 2
            ax.bar(
                x + offset,
                counts[err][sheet],
                bar_width,
                color=colours[i],
                edgecolor="black",
                linewidth=0.5,
                label=sheet if idx == 0 else None,
            )
        ax.set_title(err)
        ax.set_xticks(x, BIN_LABELS, rotation=45, ha="right")
        ax.set_ylabel("Frequency")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Frequency of Error Types by Source Sentence Length", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()



def main():

    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent
    excel_path = project_root / "Translations" / "aggregate_evaluations.xlsx"

    dataframes = read_and_prepare(excel_path)
    counts = compute_counts(dataframes)
    plot_counts(counts, list(dataframes.keys()))


if __name__ == "__main__":
    main()
