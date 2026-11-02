#!/usr/bin/env python3
"""Generate a position-linked draft replacement for manuscript Figure 2.

This is deliberately written to separate the three deterministic operations:
alignment-column evidence, retention, and base calling.  It writes candidate
files (manuscript Figure 2) into MANUSCRIPT_FIGURE_DIR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


import os

OUT_DIR = Path(os.environ.get("MANUSCRIPT_FIGURE_DIR",
                              str(Path(__file__).resolve().parents[1] / "results" / "manuscript_figures")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUT = OUT_DIR / "phisa2-candidate-v2.pdf"
PNG_OUT = OUT_DIR / "phisa2-candidate-v2.png"

plt.rcParams.update(
    {
        "font.family": "Inter",
        "font.size": 10.5,
        "pdf.fonttype": 42,
    }
)

INK = "#172033"
MUTED = "#526071"
GRID = "#CBD5E1"
PANEL_BG = "#F8FAFC"
ORANGE = "#C2410C"
ORANGE_BG = "#FFF1E8"
BLUE = "#1D4ED8"
BLUE_BG = "#EAF2FF"
GREEN = "#16794A"


def setup_panel(ax, label: str, title: str, subtitle: str | None = None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.002, 0.015),
            0.996,
            0.965,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=PANEL_BG,
            edgecolor=GRID,
            linewidth=0.9,
            transform=ax.transAxes,
        )
    )
    ax.text(0.020, 0.925, label, fontsize=14, weight="bold", color=INK, va="top")
    ax.text(0.058, 0.925, title, fontsize=12.5, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.058, 0.810, subtitle, fontsize=9.6, color=MUTED, va="top")


def cell(ax, x: float, y: float, text: str, *, color=INK, face="white", bold=False) -> None:
    width, height = 0.071, 0.098
    ax.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor=face,
            edgecolor=GRID,
            linewidth=0.8,
            transform=ax.transAxes,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color=color,
        weight="bold" if bold else "normal",
        family="DejaVu Sans Mono",
    )


# Smaller canvas so the unchanged point sizes print larger at text width.
fig = plt.figure(figsize=(8.0, 8.4), facecolor="white")
grid = fig.add_gridspec(4, 1, height_ratios=[2.65, 2.05, 1.65, 1.85], hspace=0.17)
axes = [fig.add_subplot(grid[i, 0]) for i in range(4)]

# Panel A: crop to exactly the seven columns used below, and use the same labels.
setup_panel(
    axes[0],
    "a",
    "The same alignment columns are followed through every panel",
    "Orange marks the reference-gap column; blue marks M16, used to show a deterministic 4:1 base call.",
)
states = ["M10", "M11", "M12", "M13", "M14", "M15", "M16"]
rows = [
    (r"$\Theta^{(t)}$ in alignment", ["–", "T", "A", "C", "G", "T", "A"]),
    ("read 1", ["G", "T", "A", "C", "G", "T", "A"]),
    ("read 2", ["G", "T", "A", "C", "G", "T", "A"]),
    ("read 3", ["G", "T", "A", "C", "G", "T", "A"]),
    ("read 4", ["G", "T", "A", "C", "G", "T", "T"]),
    ("read 5", ["G", "T", "A", "C", "G", "T", "A"]),
]
xs = [0.305 + 0.082 * i for i in range(7)]
header_y = 0.650
row_ys = [0.535, 0.445, 0.355, 0.265, 0.175, 0.085]

for j, (x, state) in enumerate(zip(xs, states)):
    face = ORANGE_BG if j == 0 else BLUE_BG if j == 6 else "#EEF2F7"
    edge = ORANGE if j == 0 else BLUE if j == 6 else GRID
    axes[0].add_patch(
        FancyBboxPatch(
            (x - 0.0355, header_y - 0.047),
            0.071,
            0.094,
            boxstyle="round,pad=0.004",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5 if j in (0, 6) else 0.8,
        )
    )
    axes[0].text(x, header_y, state, ha="center", va="center", weight="bold", color=edge if j in (0, 6) else INK)

for y, (name, symbols) in zip(row_ys, rows):
    axes[0].text(0.245, y, name, ha="right", va="center", color=INK if name.startswith("$") else MUTED, weight="bold" if name.startswith("$") else "normal")
    for j, (x, symbol) in enumerate(zip(xs, symbols)):
        face = ORANGE_BG if j == 0 else BLUE_BG if j == 6 else "white"
        color = ORANGE if j == 0 else BLUE if j == 6 else INK
        cell(axes[0], x, y, symbol, color=color, face=face, bold=j in (0, 6))

# Panel B: make the provenance and retention decision explicit; no state-chain arrows.
setup_panel(
    axes[1],
    "b",
    "Column retention uses the observations directly above",
    "Both highlighted columns have depth 5 and no read-supported gap majority, so both are retained.",
)

cards = [
    (0.075, 0.545, 0.395, ORANGE, ORANGE_BG, "M10 from panel a", "G, G, G, G, G", "depth = 5; read gaps = 0/5", "retain column"),
    (0.530, 0.545, 0.395, BLUE, BLUE_BG, "M16 from panel a", "A, A, A, T, A", "depth = 5; read gaps = 0/5", "retain column"),
]
for x, y, width, color, face, heading, observations, test, decision in cards:
    axes[1].add_patch(
        FancyBboxPatch(
            (x, y - 0.30),
            width,
            0.34,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=face,
            edgecolor=color,
            linewidth=1.5,
        )
    )
    axes[1].text(x + 0.018, y, heading, color=color, weight="bold", va="top", fontsize=11.2)
    axes[1].text(x + 0.018, y - 0.095, f"read symbols:  {observations}", color=INK, va="top", family="DejaVu Sans Mono")
    axes[1].text(
        x + 0.018,
        y - 0.220,
        f"{test}; {decision}",
        color=GREEN,
        va="center",
        weight="bold",
        fontsize=9.4,
    )

# Panel C: raw counts first, then the simple denominator and deterministic call.
setup_panel(
    axes[2],
    "c",
    "Base calls come from raw counts divided by column depth",
    "The proportions are descriptive counts, not fitted HMM emissions; the unique maximum is selected.",
)
columns = ["column", "raw nucleotide counts", "normalization", "deterministic call"]
col_x = [0.06, 0.19, 0.45, 0.81]
for x, heading in zip(col_x, columns):
    axes[2].text(x, 0.600, heading, color=MUTED, weight="bold", va="center", ha="left" if heading != "deterministic call" else "center")
axes[2].plot([0.05, 0.955], [0.535, 0.535], color=GRID, lw=1)

count_rows = [
    (0.405, ORANGE, "M10", "A=0, C=0, G=5, T=0", "G: 5/5 = 1.00", "G"),
    (0.190, BLUE, "M16", "A=4, C=0, G=0, T=1", "A: 4/5 = 0.80;  T: 1/5 = 0.20", "A"),
]
for y, color, state, counts, normalization, call in count_rows:
    axes[2].text(col_x[0], y, state, color=color, weight="bold", va="center", fontsize=11.4)
    axes[2].text(col_x[1], y, counts, color=INK, va="center", family="DejaVu Sans Mono", fontsize=10.2)
    axes[2].text(col_x[2], y, normalization, color=INK, va="center", fontsize=10.0)
    axes[2].add_patch(
        FancyBboxPatch(
            (0.780, y - 0.067),
            0.060,
            0.134,
            boxstyle="round,pad=0.008",
            facecolor=ORANGE_BG if color == ORANGE else BLUE_BG,
            edgecolor=color,
            linewidth=1.5,
        )
    )
    axes[2].text(0.810, y, call, color=color, weight="bold", ha="center", va="center", fontsize=12)
axes[2].text(0.855, 0.190, "not sampled", color=MUTED, style="italic", va="center", fontsize=9.5)

# Panel D: align old representation, calls, and new reference vertically.
setup_panel(
    axes[3],
    "d",
    r"Compare $\Theta^{(t)}$ and $\Theta^{(t+1)}$ at the same coordinates",
    "The aligned rows show the column-by-column change; the box at right removes the alignment gap.",
)
d_header_y = 0.620
d_row_y = [0.475, 0.305, 0.135]
d_labels = [r"$\Theta^{(t)}$ projected into $H$", "selected calls", r"$\Theta^{(t+1)}$ in $H$"]
d_xs = [0.290 + 0.068 * i for i in range(7)]
d_symbols = [
    ["–", "T", "A", "C", "G", "T", "A"],
    ["G", "T", "A", "C", "G", "T", "A"],
    ["G", "T", "A", "C", "G", "T", "A"],
]

for j, (x, state) in enumerate(zip(d_xs, states)):
    color = ORANGE if j == 0 else BLUE if j == 6 else MUTED
    axes[3].text(x, d_header_y, state, ha="center", va="center", color=color, weight="bold")

for i, (y, label, symbols) in enumerate(zip(d_row_y, d_labels, d_symbols)):
    axes[3].text(0.245, y, label, ha="right", va="center", color=INK, weight="bold" if i == 2 else "normal")
    for j, (x, symbol) in enumerate(zip(d_xs, symbols)):
        face = ORANGE_BG if j == 0 else BLUE_BG if j == 6 else "white"
        color = ORANGE if j == 0 else BLUE if j == 6 else INK
        cell(axes[3], x, y, symbol, color=color, face=face, bold=j in (0, 6) or i == 2)

axes[3].add_patch(
    FancyBboxPatch(
        (0.765, 0.105),
        0.215,
        0.435,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="white",
        edgecolor=GRID,
        linewidth=1.0,
    )
)
axes[3].text(0.785, 0.505, "Ungapped sequence", color=MUTED, weight="bold", va="top", fontsize=9.4)
axes[3].text(0.785, 0.310, r"$\Theta^{(t)}$", color=INK, va="center", fontsize=10.0)
axes[3].text(0.968, 0.310, "TACGTA", color=INK, va="center", ha="right", family="DejaVu Sans Mono", fontsize=10.2)
axes[3].text(0.785, 0.160, r"$\Theta^{(t+1)}$", color=INK, va="center", fontsize=10.0)
axes[3].text(0.968, 0.160, "GTACGTA", color=ORANGE, va="center", ha="right", family="DejaVu Sans Mono", fontsize=10.2, weight="bold")
axes[3].annotate("", xy=(0.876, 0.195), xytext=(0.876, 0.260), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.2})

fig.subplots_adjust(left=0.035, right=0.985, top=0.988, bottom=0.035)
fig.savefig(PDF_OUT, bbox_inches="tight")
fig.savefig(PNG_OUT, dpi=180, bbox_inches="tight")
print(PDF_OUT)
print(PNG_OUT)
