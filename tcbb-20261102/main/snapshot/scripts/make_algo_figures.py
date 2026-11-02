"""Generate OJEMB-style Fig. 3-8 report figures for every algorithm compared
in the BMC rebuild: ASEM without recursion, ASEM with recursion, SRSC, and
the two external de novo assemblers (NOVOPlasty, GetOrganelle).

Fig. 3 (initial-state heatmap) is a property of the dataset, not the
algorithm, and is only generated once (already present in
results/baseline1_ieee_access/figures/initial_state_heatmap_*.png).

For iterative methods (ASEM variants -- have an "iteration" column):
  - results_heatmap_<metric>.png       (Fig. 3-equivalent, post-refinement)
  - depth_effect.png / iteration_effect.png   (Fig. 4/5)
  - depth_iteration_grid_<metric>.png  (Fig. 6/7)
  - divergence_boxplot.png             (Fig. 8, new)

For non-iterative methods (SRSC, NOVOPlasty, GetOrganelle -- no "iteration"
column): the iteration-indexed figures (4/6/7) do not apply by construction;
only results_heatmap, depth_effect (single line), and divergence_boxplot are
produced, each at that method's own sensible depth range.

Usage: python3 make_algo_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from phase1_species import SPECIES

sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(__file__)
METRICS = ["f1", "identity_pct"]

# (label, results_dir, has_iteration, boxplot_depths)
DATASETS = [
    ("asem_no_recursion", "results/baseline1_ieee_access", True, (4, 8)),
    ("asem_recursion", "results/baseline2_ojemb", True, (4, 8)),
    ("srsc", "results/baseline3_ecticon", False, (4, 8)),
    ("novoplasty", "results/ext_novoplasty", False, (20, 30)),
    ("getorganelle", "results/ext_getorganelle", False, (20, 30)),
]


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
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _results_heatmap(grid: pd.DataFrame, metric: str, order: list[str], has_iteration: bool,
                      label: str, out_path: str) -> None:
    if has_iteration:
        final = grid.loc[grid.groupby(["reference", "target", "depth", "replicate"])["iteration"].idxmax()]
    else:
        final = grid
    mat = final.pivot_table(index="reference", columns="target", values=metric, aggfunc="mean")
    mat = mat.reindex(index=order, columns=order)
    _heatmap_from_matrix(mat, f"{label}: {metric} (avg over depth/replicate{'/final iteration' if has_iteration else ''})", out_path)


def _forward_fill_iterations(grid: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Carry each job's last-recorded metric values forward through
    iterations it never reached, because the EM loop (asem_core.run_asem_em_loop)
    stops as soon as theta stabilizes rather than always running to
    max_iterations -- a job absent from a later iteration means its theta
    (and metrics) held steady at its last recorded value, not that it
    "vanished".

    Without this, a naive groupby("iteration").mean() is computed over a
    shrinking, compositionally-biased subset: easy jobs converge in 2-3
    iterations (usually at a *good* F1) and then drop out of every later
    iteration's average, leaving only the hardest, still-oscillating jobs
    -- so the aggregate curve looks like it gets *worse* with more
    iterations even though almost every individual job's own F1
    trajectory is monotonically non-decreasing (verified directly against
    results/baseline1_ieee_access/grid_results.csv: e.g. Homo_sapiens vs
    Saimiri_boliviensis at 1X goes 0.871 -> 0.876 -> 0.877 -> 0.878 across
    its own 4 iterations before converging, never down). "Almost every":
    of 720 jobs, exactly 1 (Gorilla_gorilla vs Homo_sapiens, 7X, rep 1)
    has a real, tiny per-job dip (0.9781 -> 0.9779 -> 0.9780) -- the
    algorithm has no monotonicity guarantee (each M-step majority-votes
    against the *current*, possibly still-imperfect theta, not the true
    target), this dip is just empirically rare and small enough to vanish
    in a 720-job mean."""
    max_it = int(grid["iteration"].max())
    job_cols = ["reference", "target", "depth", "replicate"]
    filled = [
        grid.pivot_table(index=job_cols, columns="iteration", values=col)
        .reindex(columns=range(1, max_it + 1))
        .ffill(axis=1)
        .stack()
        .rename(col)
        for col in value_cols
    ]
    return pd.concat(filled, axis=1).reset_index()


def _depth_effect_plot(grid: pd.DataFrame, has_iteration: bool, label: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if has_iteration:
        filled = _forward_fill_iterations(grid, ["f1"])
        for it, sub in filled.groupby("iteration"):
            agg = sub.groupby("depth")["f1"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=f"iteration {it}")
        ax.legend()
    else:
        agg = grid.groupby("depth")["f1"].mean()
        ax.plot(agg.index, agg.values, marker="o", color="tab:blue")
    ax.set_xlabel("sequencing depth (X)")
    ax.set_ylabel("F1 (mean over pairs/replicates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _iteration_effect_plot(grid: pd.DataFrame, label: str, out_path: str) -> None:
    filled = _forward_fill_iterations(grid, ["f1"])
    agg = filled.groupby(["depth", "iteration"])["f1"].mean().reset_index()
    agg["depth"] = agg["depth"].astype(str) + "X"
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=agg, x="iteration", y="f1", hue="depth", marker="o", ax=ax, palette="viridis")
    ax.set_xlabel("iteration")
    ax.set_ylabel("F1 (mean over pairs/replicates)")
    ax.legend(title="depth", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _depth_iteration_grid(grid: pd.DataFrame, metric: str, order: list[str],
                           depths: tuple[int, ...], iterations: tuple[int, ...],
                           label: str, out_path: str) -> None:
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
    fig.colorbar(im, ax=axes, shrink=0.6, label=metric)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _divergence_boxplot(grid: pd.DataFrame, has_iteration: bool, depths: tuple[int, int],
                         label: str, out_path: str) -> None:
    """Fig. 8 style: boxplots of F1 and identity at two representative
    depths, split by phylogenetic (divergence) relationship between
    reference and target."""
    if has_iteration:
        sub = grid.loc[grid.groupby(["reference", "target", "depth", "replicate"])["iteration"].idxmax()]
    else:
        sub = grid
    sub = sub[sub.depth.isin(depths)]
    if sub.empty:
        print(f"  [skip boxplot for {label}] no rows at depths {depths}")
        return

    div_order = [d for d in ["same-genus", "same-family", "same-order"] if d in sub.divergence.unique()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, METRICS):
        data = [sub[(sub.depth == d) & (sub.divergence == div)][metric].dropna()
                for d in depths for div in div_order]
        labels = [f"{d}X\n{div}" for d in depths for div in div_order]
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    species_order = list(SPECIES.keys())

    for label, rel_dir, has_iteration, box_depths in DATASETS:
        results_dir = os.path.join(SCRIPT_DIR, rel_dir)
        csv_path = os.path.join(results_dir, "grid_results.csv")
        if not os.path.exists(csv_path):
            print(f"[skip {label}] no grid_results.csv at {csv_path}")
            continue
        grid = pd.read_csv(csv_path)
        if grid.empty:
            print(f"[skip {label}] empty grid_results.csv")
            continue

        fig_dir = os.path.join(results_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        print(f"=== {label} ({len(grid)} rows, has_iteration={has_iteration}) ===")

        for metric in METRICS:
            _results_heatmap(grid, metric, species_order, has_iteration, label,
                              os.path.join(fig_dir, f"results_heatmap_{metric}.png"))

        _depth_effect_plot(grid, has_iteration, label, os.path.join(fig_dir, "depth_effect.png"))

        if has_iteration:
            _iteration_effect_plot(grid, label, os.path.join(fig_dir, "iteration_effect.png"))
            for metric in METRICS:
                _depth_iteration_grid(
                    grid, metric, species_order, depths=(2, 4, 6, 8), iterations=(1, 2, 3),
                    label=label, out_path=os.path.join(fig_dir, f"depth_iteration_grid_{metric}.png"),
                )

        _divergence_boxplot(grid, has_iteration, box_depths, label,
                             os.path.join(fig_dir, "divergence_boxplot.png"))

        print(f"  wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
