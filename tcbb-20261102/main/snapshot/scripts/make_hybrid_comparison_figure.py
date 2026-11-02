"""Fig: ASEM (no recursion) vs ASEM-Hybrid, F1 by sequencing depth, same-order
pairs only (the divergence level where the hybrid gap-fill step actually
does anything -- see run_grid_baseline_hybrid.py and asem_hybrid.py).

Uses each job's own final iteration (matches make_report.py/make_algo_figures.py
convention), not a naive per-iteration mean, to avoid the survivorship-bias
artifact documented in make_algo_figures.py's _forward_fill_iterations().
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(__file__)
PLAIN_CSV = os.path.join(SCRIPT_DIR, "results", "baseline1_ieee_access", "grid_results.csv")
HYBRID_CSV = os.path.join(SCRIPT_DIR, "results", "baseline_hybrid_boundary", "grid_results.csv")
OUT_PATH = os.path.join(
    SCRIPT_DIR, "results", "baseline_hybrid_boundary", "figures", "hybrid_vs_plain_depth.png"
)

# dataviz convention: categorical slot 1 = blue (baseline), slot 2 = orange (new)
COLOR_PLAIN = "#2a78d6"
COLOR_HYBRID = "#eb6834"


def final_iter(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["reference", "target", "depth", "replicate"])["iteration"].idxmax()]


def main() -> None:
    plain = final_iter(pd.read_csv(PLAIN_CSV))
    hybrid = final_iter(pd.read_csv(HYBRID_CSV))

    plain_so = plain[plain.divergence == "same-order"].groupby("depth")["f1"].mean()
    hybrid_so = hybrid[hybrid.divergence == "same-order"].groupby("depth")["f1"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(plain_so.index, plain_so.values, marker="o", color=COLOR_PLAIN, label="ASEM (no recursion)")
    ax.plot(hybrid_so.index, hybrid_so.values, marker="o", color=COLOR_HYBRID, label="ASEM-Hybrid")
    ax.set_xlabel("sequencing depth (X)")
    ax.set_ylabel("F1 (mean over same-order pairs/replicates, final iteration)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
