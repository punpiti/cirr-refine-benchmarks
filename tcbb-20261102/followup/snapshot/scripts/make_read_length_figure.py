"""Line plot of F1 vs. read length for the read-length experiment (Section
3.4 in the manuscript, Table 4's data). One panel per algorithm -- (a) ASEM
and (b) SRSC -- so each algorithm's own read-length behavior is legible on
its own axis, instead of six overlapping lines competing in one panel.

ASEM's two variants (ieee_access = no recursion, ojemb = with recursion)
are pooled into one "ASEM" panel. In the insertion-aware rerun, F1 differs
in only 2/63 matched jobs (maximum 0.000071), and every pooled mean is
unchanged at the four-decimal precision reported in the manuscript.
Identity differs in 5/63 jobs (maximum 0.29 percentage points).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(SCRIPT_DIR, "results/read_length_experiment/results.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "results/read_length_experiment/figures")
OUT_PATH = os.path.join(OUT_DIR, "read_length_effect.png")

NUMERIC_LENGTHS = [50, 75, 100, 150, 250, 300]
DIVERGENCE_ORDER = ["same-order", "same-family", "same-genus"]  # most- to least-divergent, left to right in the legend
DIVERGENCE_COLOR = {"same-genus": "tab:green", "same-family": "tab:orange", "same-order": "tab:red"}

# (panel label, set of `baseline` values pooled into this panel).
#
# bmc/english-sections/03d-read-length-displays.tex's figure caption and
# Table 4 (tbl:read-length) currently describe a 2-panel (a)/(b) figure.
# ASEM-Hybrid ("hybrid") is deliberately NOT in this list -- see main()'s
# --include-hybrid guard below, which fails loudly instead of silently
# growing a 3rd panel if hybrid rows ever appear in results.csv.
PANELS = [
    ("(a) ASEM", {"ieee_access", "ojemb"}),
    ("(b) SRSC", {"ecticon"}),
]
HYBRID_PANEL = ("(c) ASEM-Hybrid", {"hybrid"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-hybrid", action="store_true",
        help="Add the ASEM-Hybrid panel. Only pass this after updating "
             "03d-read-length-displays.tex's caption and Table 4 to describe "
             "3 panels -- otherwise the figure and the text will disagree.",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(CSV_PATH)))
    present_baselines = {r["baseline"] for r in rows}

    if HYBRID_PANEL[1] & present_baselines and not args.include_hybrid:
        sys.exit(
            "results.csv now contains 'hybrid' rows, but the manuscript "
            "caption/table still describe a 2-panel (a)/(b) figure. Update "
            "03d-read-length-displays.tex's caption and Table 4 to describe "
            "3 panels first, then rerun with --include-hybrid."
        )
    panel_defs = PANELS + ([HYBRID_PANEL] if args.include_hybrid else [])
    panels = [(label, baselines) for label, baselines in panel_defs if baselines & present_baselines]

    means = defaultdict(list)  # (panel_label, divergence, length) -> [f1, ...]
    variable_means = defaultdict(list)  # (panel_label, divergence) -> [f1, ...] for the "variable" condition
    for r in rows:
        for label, baselines in panels:
            if r["baseline"] not in baselines:
                continue
            div = r["divergence"]
            lc = r["length_condition"]
            f1 = float(r["f1"])
            if lc == "variable":
                variable_means[(label, div)].append(f1)
            else:
                means[(label, div, int(lc))].append(f1)

    ymin, ymax = 0.65, 1.03
    var_x, box_left = 340, 320
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    lines = {}  # divergence -> Line2D, for one shared legend built from the last panel

    for ax, (label, _baselines) in zip(axes, panels):
        for div in DIVERGENCE_ORDER:
            groups = [means[(label, div, length)] for length in NUMERIC_LENGTHS]
            ys = [sum(g) / len(g) for g in groups]
            # Per-length min/max as an error bar, not just the mean line --
            # a mean-only line hides real replicate-level failures (e.g. SRSC
            # same-family at 250/300bp: one replicate near 0.73/0.80 while the
            # other two sit near 0.93, averaging out to a deceptively smooth
            # ~0.85). A mean-only line would make SRSC's degradation look
            # uniformly gradual when it is actually a mix of mostly-fine and
            # occasionally-bad runs.
            # max(0, ...): mean-of-group can round a hair below/above the
            # true min/max via float summation error when all values in a
            # group are near-identical (e.g. same-genus, all ~0.9999).
            yerr_lo = [max(0.0, y - min(g)) for y, g in zip(ys, groups)]
            yerr_hi = [max(0.0, max(g) - y) for y, g in zip(ys, groups)]
            container = ax.errorbar(
                NUMERIC_LENGTHS, ys, yerr=[yerr_lo, yerr_hi],
                linestyle="-", marker="o", color=DIVERGENCE_COLOR[div],
                label=div, capsize=3, elinewidth=1, capthick=1,
            )
            lines[div] = container.lines[0]

        # "variable"-length condition: a separate column of points inside a
        # lightly shaded box, since it is not a single fixed length and does
        # not belong on the same x-axis scale as the numeric conditions.
        ax.axvspan(box_left, var_x + 20, color="0.93", zorder=0)
        ax.axvline(box_left, color="gray", linewidth=0.7, linestyle=":")
        for div in DIVERGENCE_ORDER:
            key = (label, div)
            if key in variable_means:
                y = sum(variable_means[key]) / len(variable_means[key])
                ax.scatter([var_x], [y], color=DIVERGENCE_COLOR[div], marker="o", edgecolors="black", zorder=5)
        ax.text((box_left + var_x + 20) / 2, ymax - 0.015, "Variable\nlength", ha="center", va="top",
                fontsize=7, color="0.3", fontweight="bold", linespacing=1.3)

        ax.set_xlim(40, var_x + 20)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(NUMERIC_LENGTHS)
        ax.set_xlabel("Read length (bp)")
        ax.set_title(label, fontsize=10)

    axes[0].set_ylabel("F1")

    # One shared legend (divergence color only -- each panel is already one
    # algorithm, so no method dimension is needed here), ordered most- to
    # least-divergent to match DIVERGENCE_ORDER.
    ordered_handles = [lines[div] for div in DIVERGENCE_ORDER]
    fig.legend(ordered_handles, DIVERGENCE_ORDER, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH} ({len(panels)} panel(s): {', '.join(label for label, _ in panels)})")


if __name__ == "__main__":
    main()
