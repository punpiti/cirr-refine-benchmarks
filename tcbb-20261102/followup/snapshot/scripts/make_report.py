"""Generate the paper-style figures/tables from a completed grid run:

- initial_state_heatmap_<metric>.png : heatmap of a metric between every
  reference/target pair with NO refinement (matches the paper's Fig.
  "heatmap of F1 between each sequence pair at the initial state").
- depth_effect.png   : metric vs sequencing depth, averaged over pairs/reps,
  one line per iteration (matches "effects of sequencing depth" figure).
- iteration_effect.png : metric vs iteration, one line per depth.
- results_heatmap_<metric>.png : post-refinement heatmap per pair at the
  final iteration, averaged over depth/replicate (matches
  heatmap_results-F1 / heatmap_results-identity).
- depth_iteration_grid_<metric>.png : IEEE Access Figure 8 style -- a grid
  of heatmaps, one per (depth, iteration) combination (rows=depth in
  {2,4,6,8}X, columns=iteration in {1,2,3}), each a full reference/target
  heatmap of the metric averaged over replicates. Matches "Heatmap of F1 at
  depth 2X, 4X, 6X, and 8X after 1 to 3 iterations."
- summary_table.csv  : one row per (pair, divergence, depth) averaged over
  replicates at the final iteration -- the paper-style results table.

Run after scripts/refine_reference/run_grid_baseline1.py finishes.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from make_algo_figures import _forward_fill_iterations
from phase1_species import SPECIES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "baseline1_ieee_access")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

METRICS = ["f1", "identity_pct", "recall", "precision"]


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    initial = pd.read_csv(os.path.join(RESULTS_DIR, "initial_state.csv"))
    grid = pd.read_csv(os.path.join(RESULTS_DIR, "grid_results.csv"))

    species_order = list(SPECIES.keys())

    for metric in METRICS:
        _heatmap(
            initial, metric, species_order,
            title=f"Initial state (no refinement): {metric}",
            out_path=os.path.join(FIG_DIR, f"initial_state_heatmap_{metric}.png"),
        )
        _clustermap(
            initial, metric, species_order,
            title=f"Initial state (no refinement): {metric}, phylogeny-clustered",
            out_path=os.path.join(FIG_DIR, f"initial_state_clustermap_{metric}.png"),
        )

    final_iter = grid.loc[grid.groupby(["reference", "target", "depth", "replicate"])["iteration"].idxmax()]

    for metric in METRICS:
        final_avg = final_iter.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
        final_avg = final_avg.reindex(index=species_order, columns=species_order)
        _heatmap_from_matrix(
            final_avg, title=f"After refinement (avg over depth/replicate): {metric}",
            out_path=os.path.join(FIG_DIR, f"results_heatmap_{metric}.png"),
        )

    _depth_effect_plot(grid, os.path.join(FIG_DIR, "depth_effect.png"))
    _iteration_effect_plot(grid, os.path.join(FIG_DIR, "iteration_effect.png"))

    for metric in METRICS:
        _depth_iteration_grid(
            grid, metric, species_order,
            depths=(2, 4, 6, 8), iterations=(1, 2, 3),
            out_path=os.path.join(FIG_DIR, f"depth_iteration_grid_{metric}.png"),
        )

    summary = (
        final_iter.groupby(["reference", "target", "divergence", "depth"])[METRICS]
        .mean()
        .reset_index()
        .sort_values(["divergence", "reference", "target", "depth"])
    )
    summary_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    summary.to_csv(summary_path, index=False)

    print(f"wrote figures to {FIG_DIR}")
    print(f"wrote {summary_path} ({len(summary)} rows)")


def _heatmap(df: pd.DataFrame, metric: str, order: list[str], title: str, out_path: str) -> None:
    mat = df.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
    mat = mat.reindex(index=order, columns=order)
    _heatmap_from_matrix(mat, title, out_path)


def _heatmap_from_matrix(mat: pd.DataFrame, title: str, out_path: str) -> None:
    side = max(6.0, 0.28 * len(mat.columns))
    fig, ax = plt.subplots(figsize=(side + 1, side))
    im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("target")
    ax.set_ylabel("reference")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _depth_effect_plot(grid: pd.DataFrame, out_path: str) -> None:
    """Plot Figure 5 using the manuscript's Seaborn visual language.

    The data are first pooled at each depth/iteration cell, preserving the
    original figure's mean-over-pairs-and-replicates interpretation.  The
    restrained colourblind palette, white grid, outlined markers, and compact
    legend match the other analytic figures while the caption supplies the
    figure title.
    """
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    # Forward-fill each job's last-recorded F1 through iterations it never
    # reached (the EM loop stops once theta converges) -- otherwise this
    # groupby-mean is computed over a shrinking, compositionally-biased
    # subset of jobs at later iterations. See make_algo_figures.py's
    # _forward_fill_iterations docstring for the full mechanism.
    filled = _forward_fill_iterations(grid, ["f1"])
    summary = (
        filled.groupby(["depth", "iteration"], as_index=False)["f1"]
        .mean()
        .sort_values(["iteration", "depth"])
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.lineplot(
        data=summary,
        x="depth",
        y="f1",
        hue="iteration",
        hue_order=sorted(summary["iteration"].unique()),
        palette="viridis",
        marker="o",
        markersize=6.5,
        linewidth=2.0,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Sequencing depth (X)")
    ax.set_ylabel("Mean F1")
    ax.set_xticks(sorted(summary["depth"].unique()))
    # A focused scale keeps the depth-dependent differences legible and is
    # consistent with the neighbouring iteration-effect figure.
    lower = max(0, summary["f1"].min() - 0.01)
    upper = min(1, summary["f1"].max() + 0.005)
    ax.set_ylim(lower, upper)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    legend = ax.legend(title="EM iteration", frameon=True, loc="lower right")
    legend.get_frame().set_edgecolor("0.85")
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _clustermap(df: pd.DataFrame, metric: str, order: list[str], title: str, out_path: str) -> None:
    """Same heatmap as _heatmap, but with a dendrogram on each axis built
    from the pairwise (dis)similarity itself -- since the metric here is
    genetic similarity (F1/identity/etc. between sequences), the resulting
    clustering approximates the true phylogeny of the Phase-1 species set
    (matches the original paper's Figure 4+5 pairing: a phylogenetic tree
    alongside the pairwise heatmap)."""
    mat = df.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
    mat = mat.reindex(index=order, columns=order)

    sym = (mat.values + mat.values.T) / 2.0
    dist = sym.max() - sym
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry against float round-off
    link = linkage(squareform(dist, checks=False), method="average")

    side = max(7.0, 0.3 * len(order))
    g = sns.clustermap(
        mat, row_linkage=link, col_linkage=link, cmap="viridis",
        xticklabels=order, yticklabels=order, figsize=(side + 1, side),
        cbar_kws={"label": metric},
    )
    g.ax_heatmap.set_xlabel("target")
    g.ax_heatmap.set_ylabel("reference")
    g.savefig(out_path, dpi=150)
    plt.close(g.fig)


def _depth_iteration_grid(
    grid: pd.DataFrame,
    metric: str,
    order: list[str],
    depths: tuple[int, ...],
    iterations: tuple[int, ...],
    out_path: str,
) -> None:
    """IEEE Access Figure 8 style: a grid of reference/target heatmaps, one
    per (depth, iteration) cell, each averaged over replicates."""
    # Same forward-fill as _depth_effect_plot/_iteration_effect_plot: a
    # per-cell replicate mean would otherwise silently average over fewer
    # than 3 replicates once some have converged and dropped out by that
    # iteration.
    grid = _forward_fill_iterations(grid, [metric])
    n_rows, n_cols = len(depths), len(iterations)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols + 1.5, 3.0 * n_rows + 1), squeeze=False,
        layout="constrained",
    )
    vmin, vmax = grid[metric].min(), grid[metric].max()
    im = None
    for i, depth in enumerate(depths):
        for j, it in enumerate(iterations):
            ax = axes[i][j]
            cell = grid[(grid.depth == depth) & (grid.iteration == it)]
            mat = cell.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
            mat = mat.reindex(index=order, columns=order)
            im = ax.imshow(mat.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(order)))
            ax.set_yticks(range(len(order)))
            if i == n_rows - 1:
                ax.set_xticklabels(order, rotation=90, fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels(order, fontsize=6)
                ax.set_ylabel(f"{depth}X", fontsize=9)
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(f"iteration {it}", fontsize=9)
    fig.suptitle(f"Heatmap of {metric} at depth {'/'.join(str(d) for d in depths)}X "
                 f"after {iterations[0]} to {iterations[-1]} iterations")
    fig.colorbar(im, ax=axes, shrink=0.6, label=metric)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _iteration_effect_plot(grid: pd.DataFrame, out_path: str) -> None:
    # See _depth_effect_plot: without forward-filling, jobs that converge
    # early drop out of later-iteration means, biasing the aggregate.
    grid = _forward_fill_iterations(grid, ["f1"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for depth, sub in grid.groupby("depth"):
        agg = sub.groupby("iteration")["f1"].mean()
        ax.plot(agg.index, agg.values, marker="o", label=f"{depth}X")
    ax.set_xlabel("iteration")
    ax.set_ylabel("F1 (mean over pairs/replicates)")
    ax.set_title("Effect of iteration count on performance")
    ax.legend(title="depth", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
