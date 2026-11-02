"""Redo Fig. 6/7 (depth x iteration heatmap grid) to match OJEMB's original
design: the initial-state heatmap (with a dendrogram giving the row/column
order by hierarchical clustering on genetic similarity) is shrunk to the
same panel size and placed alongside the post-refinement heatmaps, all
sharing one species order (from the dendrogram) and one colorbar.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from make_algo_figures import _forward_fill_iterations

SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results/baseline2_ojemb")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

SPECIES_ORDER_DEFAULT = ["Homo_sapiens", "Gorilla_gorilla", "Saimiri_boliviensis",
                          "Saimiri_sciureus", "Aotus_azarai", "Varecia_variegata"]
DEPTHS = (2, 4, 6, 8)
ITERATIONS = (1, 2, 3)


def _cluster_order(initial: pd.DataFrame, order: list[str]):
    mat = initial.pivot_table(index="reference", columns="target", values="f1", aggfunc="mean")
    mat = mat.reindex(index=order, columns=order)
    sym = (mat.values + mat.values.T) / 2.0
    dist = sym.max() - sym
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    link = linkage(squareform(dist, checks=False), method="average")
    dendro = dendrogram(link, labels=order, no_plot=True)
    leaf_order = dendro["ivl"]
    return leaf_order, link, mat.reindex(index=leaf_order, columns=leaf_order)


def make_grid(metric: str, metric_label: str, out_path: str) -> None:
    initial = pd.read_csv(os.path.join(RESULTS_DIR, "initial_state.csv"))
    grid = pd.read_csv(os.path.join(RESULTS_DIR, "grid_results.csv"))
    # Without this, a per-(depth, iteration) cell's pivot_table mean is
    # computed over however many of the 3 replicates hadn't yet converged
    # and dropped out by that iteration -- the same survivorship-bias
    # mechanism as the iteration-effect line charts (make_algo_figures.py),
    # just landing per-cell instead of per-aggregate-line.
    grid = _forward_fill_iterations(grid, [metric])

    leaf_order, link, initial_mat_f1 = _cluster_order(initial, SPECIES_ORDER_DEFAULT)
    initial_metric_mat = initial.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
    initial_metric_mat = initial_metric_mat.reindex(index=leaf_order, columns=leaf_order)

    vmin = min(initial_metric_mat.values.min(), grid[metric].min())
    vmax = max(initial_metric_mat.values.max(), grid[metric].max())

    n_rows, n_cols = len(DEPTHS), 1 + len(ITERATIONS)  # +1 column for "Initial"
    fig = plt.figure(figsize=(3.2 * n_cols + 1.8, 3.0 * n_rows + 1.6))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.5] + [3] * n_rows,
                           hspace=0.15, wspace=0.1)

    # Dendrogram above the "Initial" column only.
    ax_dendro = fig.add_subplot(gs[0, 0])
    dendrogram(link, labels=leaf_order, ax=ax_dendro, no_labels=True, color_threshold=0, above_threshold_color="black")
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    ax_dendro.set_title("Initial state\n(no refinement)", fontsize=9)

    im = None
    for i, depth in enumerate(DEPTHS):
        # Column 0: initial-state heatmap (identical every row, same species order/scale).
        ax0 = fig.add_subplot(gs[i + 1, 0])
        im = ax0.imshow(initial_metric_mat.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax0.set_xticks(range(len(leaf_order)))
        ax0.set_yticks(range(len(leaf_order)))
        ax0.set_yticklabels(leaf_order, fontsize=6)
        ax0.set_ylabel(f"{depth}X", fontsize=9)
        if i == n_rows - 1:
            ax0.set_xticklabels(leaf_order, rotation=90, fontsize=6)
        else:
            ax0.set_xticklabels([])

        for j, it in enumerate(ITERATIONS):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            cell = grid[(grid.depth == depth) & (grid.iteration == it)]
            mat = cell.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
            mat = mat.reindex(index=leaf_order, columns=leaf_order)
            im = ax.imshow(mat.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(leaf_order)))
            ax.set_yticks(range(len(leaf_order)))
            ax.set_yticklabels([])
            if i == n_rows - 1:
                ax.set_xticklabels(leaf_order, rotation=90, fontsize=6)
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(f"iteration {it}", fontsize=9)

    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=metric_label)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    make_grid("f1", "F1", os.path.join(FIG_DIR, "depth_iteration_grid_f1_v2.png"))
    make_grid("identity_pct", "Identity (%)", os.path.join(FIG_DIR, "depth_iteration_grid_identity_pct_v2.png"))
