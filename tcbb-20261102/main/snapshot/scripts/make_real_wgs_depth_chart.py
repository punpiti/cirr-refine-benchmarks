"""Plot F1 vs. sequencing depth for chimpanzee-reference real-WGS jobs.

Values are derived directly from the selected benchmark CSV and its
initial-state summary so the chart cannot drift from the results.

Usage: python3 make_real_wgs_depth_chart.py
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
VARIANT = os.environ.get("ASEM_HYBRID_VARIANT", "boundary").strip().lower()
if VARIANT not in {"legacy", "boundary"}:
    raise ValueError("ASEM_HYBRID_VARIANT must be 'legacy' or 'boundary'")
RESULTS_NAME = "real_wgs_hybrid" if VARIANT == "legacy" else "real_wgs_hybrid_boundary"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", RESULTS_NAME)
RESULTS_CSV = os.path.join(RESULTS_DIR, "final_results.csv")
INITIAL_CSV = os.path.join(RESULTS_DIR, "initial_states.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "figures", "real_wgs_depth.png")

COLOR = "#2a78d6"  # dataviz skill: categorical slot 1 -- single series, no legend needed.


def main() -> None:
    results = pd.read_csv(RESULTS_CSV)
    chimp = results[
        (results.reference == "Pan_troglodytes_NC_001643")
        & (results.method == "ASEM_no_recursion")
    ]
    depth_means = chimp.groupby("depth")["f1"].mean().sort_index()
    initial = pd.read_csv(INITIAL_CSV).set_index("reference")
    depths = [0, *depth_means.index.astype(int).tolist()]
    f1 = [float(initial.loc["Pan_troglodytes_NC_001643", "f1"]), *depth_means.tolist()]

    fig, ax = plt.subplots(figsize=(7, 5))

    # 0X (initial, unrefined) drawn as an open marker to distinguish it from
    # the six refined ASEM measurements, same convention as the 1-8X point
    # in fig_external_tools_depth.png.
    ax.plot(depths[1:], f1[1:], marker="o", markersize=8, linewidth=2, color=COLOR)
    ax.plot(depths[:2], f1[:2], linewidth=2, color=COLOR)
    ax.scatter([0], [f1[0]], s=140, facecolors="none", edgecolors=COLOR, linewidths=1.5, zorder=5)
    ax.annotate("0X: initial\n(unrefined)", xy=(0, f1[0]), xytext=(1.15, f1[0] + 0.006),
                fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8))

    ax.set_xticks(depths)
    ax.set_xticklabels(["0X", "1X", "2X", "3X", "4X", "5X", "6X", "7X", "8X"])
    ax.set_xlabel("sequencing depth")
    ax.set_ylabel("F1")
    ax.set_ylim(0.930, 0.985)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
