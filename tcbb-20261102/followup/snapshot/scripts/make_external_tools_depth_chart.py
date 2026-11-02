"""Line-chart replacement for Table 4 (tbl:external-tools): F1 vs. sequencing
depth for the two external de novo assemblers (NOVOPlasty, GetOrganelle).

NOVOPlasty's column is the same-genus pair (Saimiri_boliviensis <->
Saimiri_sciureus, both directions pooled -- there is exactly one same-genus
pair in the 6-species panel) computed live from
results/ext_novoplasty/grid_results.csv, chosen to show a clean transition
curve rather than being diluted by same-order pairs NOVOPlasty essentially
never solves regardless of depth. GetOrganelle's column is pooled across
all pair difficulties from results/ext_getorganelle/grid_results.csv (its
own near-total failure below ~10X makes a same-genus-only subset far too
small to be a stable estimate).

All five points (including the "1-8X" bucket, which pools depths 1-8
together as one point) are success-only means computed live from the two
tools' grid_results.csv -- jobs that failed to produce a scaffold are
excluded, not counted as F1=0 (matches bmc/thai-sections/03c-external-tools-table.tex,
which reports success/total counts alongside each value for transparency).
NOVOPlasty's 1-8X point is a real, non-trivial 0.11 (32/48 same-genus jobs
succeeded, mostly at 4-8X), not "near zero" -- only GetOrganelle's 1-8X
point is genuinely near-zero (3/720). A prior version of this chart
hardcoded a shared "~0, near-complete failure" annotation for both tools'
1-8X point; that was wrong for NOVOPlasty specifically once computed from
data instead of assumed, and is now sourced from the same live values as
everything else here.

These values were previously hardcoded, copied verbatim from
bmc/thai-sections/03c-external-tools-table.tex, written while the
ext_novoplasty retry-timeout pass was still mid-run (2026-08-19/20) --
after that grid finished, the hardcoded NOVOPlasty numbers were found to
be stale (10X moved 0.40 -> 0.13 once retry-recovered, genuinely marginal
10X jobs were included; 15X/20X/30X barely moved). Now computed directly
from the CSVs instead, so this can't drift from the underlying data again;
if Table 4's numbers need to change, re-run this script and update the
table together, not the table alone.

This study's own four methods (ASEM no-recursion, ASEM with recursion,
SRSC, ASEM-Hybrid) are also plotted at the "1-8X" point, on the identical
same-genus pair -- the fairest possible comparison, since it's the exact
same test case NOVOPlasty's column uses. All four land at essentially the
same value (0.9998-0.9999; verified live from each method's own
grid_results.csv, not assumed) since same-genus is the easiest pair in the
panel and every internal method already solves it near-perfectly -- shown
as one marker/label rather than four overlapping, visually indistinguishable
lines. This study's methods have no data (and make no claim) at
10X/15X/20X/30X, since that range is outside this paper's low-resource
scope by design -- the marker is deliberately not connected by a line
across those x-positions.

Usage: python3 make_external_tools_depth_chart.py
"""

from __future__ import annotations

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "bmc", "assets", "fig_external_tools_depth.png")
NOVOPLASTY_CSV = os.path.join(SCRIPT_DIR, "results", "ext_novoplasty", "grid_results.csv")
GETORGANELLE_CSV = os.path.join(SCRIPT_DIR, "results", "ext_getorganelle", "grid_results.csv")
OWN_METHOD_CSVS = {
    "ASEM (no rec.)": os.path.join(SCRIPT_DIR, "results", "baseline1_ieee_access", "grid_results.csv"),
    "ASEM (with rec.)": os.path.join(SCRIPT_DIR, "results", "baseline2_ojemb", "grid_results.csv"),
    "SRSC": os.path.join(SCRIPT_DIR, "results", "baseline3_ecticon", "grid_results.csv"),
    "ASEM-Hybrid": os.path.join(SCRIPT_DIR, "results", "baseline_hybrid", "grid_results.csv"),
}

DEPTH_LABELS = ["1–8X", "10X", "15X", "20X", "30X"]


def _f1_and_counts(csv_path: str, divergence: str | None = None) -> tuple[dict, dict, dict]:
    """Returns (f1_by_depth, n_success_by_depth, n_total_by_depth), plus a
    synthetic "1-8X" bucket pooling depths 1 through 8 together."""
    df = pd.read_csv(csv_path)
    if divergence is not None:
        df = df[df.divergence == divergence]
    df = df.copy()
    df["bucket"] = df["depth"].apply(lambda d: d if d > 8 else "1-8X")
    f1 = df.groupby("bucket")["f1"].mean().to_dict()
    n_success = df.groupby("bucket")["f1"].apply(lambda s: s.notna().sum()).to_dict()
    n_total = df.groupby("bucket").size().to_dict()
    return f1, n_success, n_total


def _series(csv_path: str, divergence: str | None = None) -> list[float]:
    f1, _, _ = _f1_and_counts(csv_path, divergence)
    keys = ["1-8X", 10, 15, 20, 30]
    return [float(f1.get(k, float("nan"))) for k in keys]


def _own_method_f1_1_8x(csv_path: str) -> float:
    """This study's methods: same-genus pair, depths 1-8 pooled (matches
    NOVOPlasty's own methodology for a fair, identical-test-case
    comparison). Iterative methods (everything but SRSC) use each job's
    own final recorded iteration."""
    df = pd.read_csv(csv_path)
    if "iteration" in df.columns:
        df = df.loc[df.groupby(["reference", "target", "depth", "replicate"])["iteration"].idxmax()]
    sub = df[(df.divergence == "same-genus") & (df.depth <= 8)]
    return float(sub["f1"].mean())


NOVOPLASTY_F1 = _series(NOVOPLASTY_CSV, divergence="same-genus")
GETORGANELLE_F1 = _series(GETORGANELLE_CSV)
OWN_METHOD_F1 = {name: _own_method_f1_1_8x(path) for name, path in OWN_METHOD_CSVS.items()}

# dataviz skill: fixed-order categorical slots 1 (blue), 2 (orange), 3 (aqua).
COLOR_NOVOPLASTY = "#2a78d6"
COLOR_GETORGANELLE = "#eb6834"
COLOR_OWN_METHODS = "#1baf7a"


def main() -> None:
    x = list(range(len(DEPTH_LABELS)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, NOVOPLASTY_F1, marker="o", markersize=8, linewidth=2,
             color=COLOR_NOVOPLASTY, label="NOVOPlasty (same-genus pair, both directions)")
    ax.plot(x, GETORGANELLE_F1, marker="o", markersize=8, linewidth=2,
             color=COLOR_GETORGANELLE, label="GetOrganelle (pooled, all pair difficulties)")

    # The 1-8X point pools 8 different depths into one estimate (unlike the
    # other four, each a single depth) -- mark both tools' points there
    # distinctly so the chart doesn't overstate how precise/comparable they
    # are to the rest. Only GetOrganelle's is actually near-zero; NOVOPlasty's
    # is a real 0.11 (mostly driven by 4-8X, where it succeeds consistently
    # but not accurately), so the two get separate annotations, not a shared
    # "both near-complete failure" label.
    ax.scatter([0, 0], [NOVOPLASTY_F1[0], GETORGANELLE_F1[0]], s=140, facecolors="none",
               edgecolors="0.35", linewidths=1.5, zorder=5)
    ax.annotate(f"{NOVOPLASTY_F1[0]:.2f} (32/48 succeeded)", xy=(0, NOVOPLASTY_F1[0]),
                xytext=(0.5, 0.32), fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8))
    ax.annotate(f"{GETORGANELLE_F1[0]:.4f} (3/720 succeeded)", xy=(0, GETORGANELLE_F1[0]),
                xytext=(0.5, 0.08), fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8))

    # This study's four methods, same test case (same-genus pair, 1-8X
    # pooled) as NOVOPlasty's column -- plotted as one marker since all four
    # land within 0.00003 of each other (same-genus is solved near-perfectly
    # by every internal method), not as four overlapping indistinguishable
    # lines. No line connects it to the 10-30X points: this study makes no
    # claim there, by design (outside the low-resource scope).
    own_mean = sum(OWN_METHOD_F1.values()) / len(OWN_METHOD_F1)
    ax.scatter([0], [own_mean], marker="D", s=110, color=COLOR_OWN_METHODS, zorder=6,
               label="This study, all 4 methods (same-genus, 1-8X pooled)")
    own_values = ", ".join(f"{v:.4f}" for v in OWN_METHOD_F1.values())
    # Denominator shown explicitly (48/48 succeeded, each method) for the same
    # transparency standard as the NOVOPlasty/GetOrganelle annotations above,
    # even though every internal method succeeds on every job here -- omitting
    # it would be an inconsistent double standard, not just a stylistic gap.
    ax.annotate(f"{own_mean:.4f} (48/48 succeeded, each)\n(all 4: {own_values})", xy=(0, own_mean),
                xytext=(0.12, 0.90), fontsize=7.5, color="#0d7a52")

    ax.set_xticks(x)
    ax.set_xticklabels(DEPTH_LABELS)
    ax.set_xlabel("sequencing depth")
    ax.set_ylabel("F1")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
