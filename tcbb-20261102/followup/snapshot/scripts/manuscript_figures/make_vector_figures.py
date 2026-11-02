#!/usr/bin/env python3
"""Regenerate the manuscript's data figures as true vector PDFs.

Each figure is drawn directly from the benchmark CSVs in
asem-refine-benchmarks/scripts/results. Heatmap cells use pcolormesh (vector
quads), never imshow, so no figure embeds a raster image.

Figures that depend on the de novo assemblers (NOVOPlasty/GetOrganelle) are
produced only when every successful job has a normalized score.

Run with: MANUSCRIPT_FIGURE_DIR=<out dir> python scripts/manuscript_figures/make_vector_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

BENCH = Path(__file__).resolve().parents[1]
RESULTS = BENCH / "results"
# The manuscript build points this at its own assets/ directory.
ASSETS = Path(os.environ.get("MANUSCRIPT_FIGURE_DIR", str(RESULTS / "manuscript_figures")))
ASSETS.mkdir(parents=True, exist_ok=True)
KEYS = ["reference", "target", "depth", "replicate"]

sys.path.insert(0, str(BENCH))


def _vector_rc() -> None:
    # Embed TrueType fonts (Type 42) so text stays editable/searchable text.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # Colorbars rasterize their gradient once it has >= n_rasterize steps;
    # keep the colour ramp as vector quads too.
    matplotlib.colorbar.Colorbar.n_rasterize = 10**9


def final_iteration(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(KEYS)["iteration"].idxmax()]


def panel_label(ax, letter: str) -> None:
    """Springer Nature style: bold lowercase part letter at the top left."""
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="left", va="bottom")


def print_fit(fig, size: tuple[float, float], min_font: float = 8.0) -> None:
    """Resize a figure drawn by an external plotting function and raise any
    text below `min_font` so it stays legible at the printed width."""
    from matplotlib.text import Text
    fig.set_size_inches(*size)
    for t in fig.findobj(Text):
        if t.get_text() and t.get_fontsize() < min_font:
            t.set_fontsize(min_font)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Figures reproduced with the released benchmark plotting functions
# ---------------------------------------------------------------------------

def _seamless(mesh) -> None:
    # Adjacent vector quads are antialiased independently, which leaves
    # hairline seams in PDF viewers; stroking each quad in its own fill
    # colour closes them.
    mesh.set_edgecolor("face")
    mesh.set_linewidth(0.3)


def fig_initial_state_heatmap() -> None:
    """30-species initial-state F1 matrix, phylogeny-clustered.

    Same construction as make_report._clustermap (average linkage on the
    symmetrised F1 distance), applied to results/full_panel.
    """
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    from full_panel_species import list_species

    _vector_rc()
    df = pd.read_csv(RESULTS / "full_panel" / "initial_state_30species.csv")
    order = list(list_species().keys())
    if len(order) != 30 or len(df) != 900:
        raise ValueError(f"expected 30 species / 900 pairs, got {len(order)} / {len(df)}")
    mat = df.pivot_table(index="reference", columns="target", values="f1", aggfunc="mean")
    mat = mat.reindex(index=order, columns=order)
    sym = (mat.values + mat.values.T) / 2.0
    dist = sym.max() - sym
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    link = linkage(squareform(dist, checks=False), method="average")
    # mako (wide lightness range, no yellow) keeps this narrow-range
    # similarity scale visibly distinct from the 0--1 viridis scale of the
    # cross-method heatmaps, so the two are not read as one scale.
    g = sns.clustermap(mat, row_linkage=link, col_linkage=link, cmap="mako_r",
                       xticklabels=order, yticklabels=order, figsize=(6.4, 6.2),
                       cbar_kws={"label": "initial F1 (no refinement)"}, cbar_pos=(0.90, 0.03, 0.02, 0.14))
    _seamless(g.ax_heatmap.collections[0])
    _seamless(g.ax_cbar.collections[0])
    g.ax_heatmap.set_xlabel("target", fontsize=9)
    g.ax_heatmap.set_ylabel("reference", fontsize=9)
    g.ax_heatmap.tick_params(labelsize=7.5)
    g.ax_cbar.tick_params(labelsize=8)
    g.ax_cbar.yaxis.label.set_size(8.5)
    g.savefig(ASSETS / "fig_initial_state_heatmap.pdf")
    plt.close("all")


DEPTH_COLORS = {4: "#86b6ef", 8: "#1c5cab"}  # validated ordinal blue ramp


def fig_divergence_boxplot() -> None:
    """ASEM with recursion at 4X and 8X: box + every job as a point.

    Same data selection as make_algo_figures._divergence_boxplot (final
    iteration of baseline2_ojemb), redrawn for print: the closer divergence
    classes have only six jobs per depth, so points are shown, not just boxes.
    """
    _vector_rc()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    grid = final_iteration(pd.read_csv(RESULTS / "baseline2_ojemb" / "grid_results.csv"))
    sub = grid[grid["depth"].isin([4, 8])]
    metrics = [("f1", "F1", "F1"), ("identity_pct", "Identity (%)", "Identity")]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for g, div in enumerate(DIVERGENCES):
            for k, depth in enumerate((4, 8)):
                vals = sub[(sub["divergence"] == div) & (sub["depth"] == depth)][col].to_numpy()
                x = g + (k - 0.5) * 0.36
                color = DEPTH_COLORS[depth]
                ax.boxplot(vals, positions=[x], widths=0.28, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.4),
                           medianprops=dict(color=color, linewidth=2.2),
                           whiskerprops=dict(color=color, linewidth=1.2),
                           capprops=dict(color=color, linewidth=1.2))
                jitter = rng.uniform(-0.09, 0.09, size=len(vals))
                ax.scatter(x + jitter, vals, s=13, color=color, edgecolor="white",
                           linewidth=0.4, zorder=3)
        counts = sub[sub["depth"] == 4].groupby("divergence").size()
        ax.set_xticks(range(3), [f"{d.split('-')[1]}\nn = {counts[d]}"
                                 for d in DIVERGENCES])
        ax.set_xlim(-0.6, 2.6)
        ax.set_xlabel("Shared taxonomic rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        panel_label(ax, "ab"[metrics.index((col, ylabel, title))])
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10, color=DEPTH_COLORS[d],
                          label=f"{d}X") for d in (4, 8)]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03), title="Sequencing depth", title_fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    fig.savefig(ASSETS / "fig_divergence_boxplot.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()


def fig_hybrid_vs_plain() -> None:
    import make_hybrid_comparison_figure as mod

    _vector_rc()
    mod.OUT_PATH = str(ASSETS / "fig_hybrid_vs_plain_depth.pdf")
    orig_savefig = mod.plt.Figure.savefig

    def savefig(fig, *a, **k):
        for ax in fig.axes:
            ax.set_ylabel("Mean F1, same-order pairs")
            ax.set_xlabel("Sequencing depth (X)")
        print_fit(fig, (5.0, 3.5), 8.5)
        return orig_savefig(fig, *a, **k)
    mod.plt.Figure.savefig = savefig
    try:
        mod.main()
    finally:
        mod.plt.Figure.savefig = orig_savefig


def fig_read_length() -> None:
    import make_read_length_figure as mod

    _vector_rc()
    mod.OUT_DIR = str(ASSETS)
    mod.PANELS = [("ASEM", {"ieee_access", "ojemb"}), ("SRSC", {"ecticon"})]
    orig_subplots = mod.plt.subplots

    def labelled_subplots(*a, **k):
        fig, axes = orig_subplots(*a, **k)
        save = fig.savefig

        def savefig(*aa, **kk):
            for ax, letter in zip(np.atleast_1d(axes), "abc"):
                panel_label(ax, letter)
            print_fit(fig, (6.6, 4.5), 8.0)
            for ax in np.atleast_1d(axes):
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for ax in np.atleast_1d(axes):
                for t in ax.texts:
                    if t.get_text().startswith("Variable"):
                        # Vertical label fits inside the narrow shaded column.
                        t.set_text("Variable length")
                        t.set_rotation(90)
                        t.set_va("center")
                        t.set_position((t.get_position()[0], 0.725))
            for legend in fig.legends:
                legend.set_loc("lower center")
                legend.set_bbox_to_anchor((0.5, 0.0))
            fig.tight_layout(rect=(0, 0.10, 1, 1))
            save(*aa, **kk)
        fig.savefig = savefig
        return fig, axes
    mod.plt.subplots = labelled_subplots
    mod.OUT_PATH = str(ASSETS / "fig_read_length_effect.pdf")
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        mod.main()
    finally:
        sys.argv = argv
        mod.plt.subplots = orig_subplots


# ---------------------------------------------------------------------------
# Reference-guided cross-method figures (include Pilon and MIA)
# ---------------------------------------------------------------------------

# Fixed categorical order (validated palette: CVD-safe adjacent pairs; markers
# and line styles give a second, non-colour encoding).
METHODS = [
    ("ASEM", RESULTS / "baseline1_ieee_access" / "grid_results.csv", True, "#2a78d6", "o", "-"),
    ("ASEM-Hybrid", RESULTS / "baseline_hybrid_boundary" / "grid_results.csv", True, "#eb6834", "s", "-"),
    ("SRSC", RESULTS / "baseline3_ecticon" / "grid_results.csv", False, "#1baf7a", "^", "-"),
    ("Pilon", RESULTS / "ext_pilon" / "grid_results.csv", False, "#eda100", "D", "-"),
    ("MIA", RESULTS / "ext_mia" / "grid_results.csv", False, "#e87ba4", "v", "--"),
]
DIVERGENCES = ["same-genus", "same-family", "same-order"]


def load_method(path: Path, iterative: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if iterative:
        df = final_iteration(df)
    if len(df) != 720 or df.duplicated(KEYS).any():
        raise ValueError(f"{path}: expected 720 unique jobs, found {len(df)}")
    return df


# De novo assemblers: success/failure comes from the original grid; the F1
# of every successful job is replaced by the strand/circular-origin
# normalized rerun. Plotting refuses to proceed if any
# successful job lacks a normalized score, so normalized and unnormalized
# values are never mixed.

# Accepted normalized reruns live in this repository; override only to
# inspect another rerun copy.
DENOVO_RERUN_RESULTS = Path(os.environ.get("DENOVO_RERUN_RESULTS", str(RESULTS)))
DENOVO = [
    ("NOVOPlasty", "ext_novoplasty", "#008300", "P", ":"),
    ("GetOrganelle", "ext_getorganelle", "#4a3aa7", "X", ":"),
]


def load_denovo(tool_dir: str, allow_preliminary: bool = False) -> tuple[pd.DataFrame, bool]:
    """Return (jobs, preliminary). preliminary=True means the normalized
    rerun is incomplete and *every* success keeps its original score."""
    grid = pd.read_csv(RESULTS / tool_dir / "grid_results.csv")
    if len(grid) != 1080 or grid.duplicated(KEYS).any():
        raise ValueError(f"{tool_dir}: expected 1080 unique jobs, found {len(grid)}")
    ok = grid["error"].fillna("").eq("") & grid["f1"].notna()
    rerun = pd.read_csv(DENOVO_RERUN_RESULTS / tool_dir / "normalized_successful_rerun.csv")
    rerun = rerun[rerun["error"].fillna("").eq("")]
    if rerun.duplicated(KEYS).any():
        raise ValueError(f"{tool_dir}: duplicate keys in normalized rerun")
    normalized = rerun.set_index(KEYS)["f1"]
    out = grid.set_index(KEYS)
    out["f1"] = np.nan
    success_keys = out.index[ok.to_numpy()]
    missing = success_keys.difference(normalized.index)
    if len(missing):
        if not allow_preliminary:
            raise ValueError(f"{tool_dir}: {len(missing)} successful jobs lack a normalized score")
        print(f"[PRELIMINARY] {tool_dir}: {len(missing)}/{len(success_keys)} successes "
              "not yet normalized; using original scores for all successes")
        out.loc[success_keys, "f1"] = grid.set_index(KEYS).loc[success_keys, "f1"].to_numpy()
        return out.reset_index(), True
    out.loc[success_keys, "f1"] = normalized.loc[success_keys].to_numpy()
    return out.reset_index(), False


def stamp(ax_or_fig, text: str = "PRELIMINARY", **kw) -> None:
    kw.setdefault("transform", getattr(ax_or_fig, "transAxes", None) or ax_or_fig.transFigure)
    ax_or_fig.text(0.5, 0.5, text, ha="center", va="center", rotation=30, fontsize=kw.pop("fontsize", 15),
                   color="#c0392b", alpha=0.85, fontweight="bold", zorder=20,
                   bbox=dict(facecolor="white", alpha=0.75, edgecolor="#c0392b"), **kw)


HEATMAP_SPECIES = ["Homo_sapiens", "Gorilla_gorilla", "Saimiri_boliviensis",
                   "Saimiri_sciureus", "Aotus_azarai", "Varecia_variegata"]
HEATMAP_LABELS = ["H. sapiens", "G. gorilla", "S. boliviensis", "S. sciureus",
                  "A. azarai", "V. variegata"]


def _heatmap_matrix(df: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    sub = df[df["depth"] == 8]
    if len(sub) != 90:
        raise ValueError(f"expected 90 jobs at 8X, found {len(sub)}")
    means = sub.groupby(["reference", "target"])["f1"].mean()
    n = len(HEATMAP_SPECIES)
    a = np.full((n, n), np.nan)
    for i, ref in enumerate(HEATMAP_SPECIES):
        for j, tgt in enumerate(HEATMAP_SPECIES):
            if ref == tgt:
                a[i, j] = 1.0
            elif (ref, tgt) in means.index:
                a[i, j] = means.loc[(ref, tgt)]
    return a, int(sub["f1"].notna().sum()), len(sub)


HEATMAP_ABBR = ["Hsa", "Ggo", "Sbo", "Ssc", "Aaz", "Vva"]


def fig_cross_method_heatmaps(panels: list[tuple[str, pd.DataFrame]],
                              preliminary: frozenset = frozenset()) -> None:
    """Matched 8X heatmaps, 2 x 4 grid sized for the printed text width.

    Species are abbreviated (key in the caption) so the text can stay large;
    self cells are left white, no-output cells are gray with an x.
    """
    from matplotlib.patches import Rectangle
    _vector_rc()
    plt.rcParams["font.family"] = "DejaVu Sans"
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#c4c8cf")
    fig, grid = plt.subplots(2, 4, figsize=(7.2, 4.3), facecolor="white")
    axes = list(grid.ravel())
    n = len(HEATMAP_SPECIES)
    centers = np.arange(n) + 0.5
    mesh = None
    for idx, (ax, (name, df)) in enumerate(zip(axes, panels)):
        a, ok, total = _heatmap_matrix(df)
        np.fill_diagonal(a, np.nan)
        mesh = ax.pcolormesh(np.ma.masked_invalid(a), cmap=cmap, vmin=0, vmax=1,
                             edgecolors="white", linewidth=0.8)
        for i in range(n):
            ax.add_patch(Rectangle((i, i), 1, 1, facecolor="white", edgecolor="white", zorder=3))
            for j in range(n):
                if i != j and np.isnan(a[i, j]):
                    ax.text(j + 0.5, i + 0.5, "×", ha="center", va="center",
                            fontsize=10, color="#374151", zorder=4)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\n{ok}/{total} outputs", fontsize=9.5, weight="bold", pad=4)
        ax.set_xticks(centers, HEATMAP_ABBR, rotation=90, fontsize=9)
        ax.set_yticks(centers, HEATMAP_ABBR if idx % 4 == 0 else [], fontsize=9)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if name in preliminary:
            stamp(ax, fontsize=9)
    # Eighth slot: colour bar and axis key.
    key = axes[7]
    key.axis("off")
    cax = key.inset_axes([0.02, 0.10, 0.10, 0.82])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)
    key.text(0.50, 0.92, "mean F1, 8X", fontsize=9, transform=key.transAxes, va="top")
    key.text(0.50, 0.70, "rows: starting\nreference\ncolumns: target", fontsize=8.5,
             transform=key.transAxes, va="top", linespacing=1.3)
    key.text(0.50, 0.24, "× no output", fontsize=8.5, transform=key.transAxes, va="top")
    fig.tight_layout(w_pad=0.6)
    fig.subplots_adjust(hspace=0.62)
    fig.savefig(ASSETS / "fig_cross_method_heatmaps.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_reference_guided_depth(data: dict[str, pd.DataFrame]) -> None:
    _vector_rc()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    fig, grid_axes = plt.subplots(2, 2, figsize=(6.6, 5.6))
    axes = grid_axes.ravel()
    depths = list(range(1, 9))
    titles = {"same-genus": "same genus", "same-family": "same family",
              "same-order": "same order"}
    handles = []
    for ax, div in zip(axes[:3], DIVERGENCES):
        for name, _p, _it, color, marker, ls in METHODS:
            df = data[name]
            means = df[df["divergence"] == div].groupby("depth")["f1"].mean().reindex(depths)
            mfc = "white" if name == "MIA" else color
            # ASEM and ASEM-Hybrid are identical outside same-order pairs: draw
            # Hybrid as larger squares underneath so both series stay visible.
            size = {"ASEM": 4.5, "ASEM-Hybrid": 8.5}.get(name, 5.5)
            (line,) = ax.plot(depths, means.values, color=color, marker=marker, linestyle=ls,
                              linewidth=1.8, markersize=size, markerfacecolor=mfc,
                              zorder={"ASEM": 4, "ASEM-Hybrid": 3}.get(name, 2),
                              markeredgecolor=color, markeredgewidth=1.3,
                              label="MIA (scoreable outputs only)" if name == "MIA" else name)
            if div == DIVERGENCES[0]:
                handles.append(line)
        ax.set_title(titles[div], fontsize=12)
        panel_label(ax, "abc"[DIVERGENCES.index(div)])
        # Common 0--1 F1 scale across (a--c) so between-panel differences
        # are not exaggerated by per-panel autoscaling.
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(depths)
        ax.set_xlabel("Sequencing depth (X)")
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for i in (0, 1, 2):
        axes[i].set_ylabel("Mean F1")

    ax = axes[3]
    mia = data["MIA"]
    for div, color, marker in zip(DIVERGENCES, ["#4b5563", "#4b5563", "#111827"], ["o", "s", "^"]):
        sub = mia[mia["divergence"] == div]
        rate = sub.groupby("depth")["f1"].apply(lambda s: 100 * s.notna().mean()).reindex(depths)
        ls = {"same-genus": ":", "same-family": "--", "same-order": "-"}[div]
        ax.plot(depths, rate.values, color=color, marker=marker, linestyle=ls,
                linewidth=1.6, markersize=5, label=div.replace("-", " "))
    ax.set_ylim(0, 105)
    ax.set_title("MIA scoreable outputs", fontsize=12)
    panel_label(ax, "d")
    ax.set_xticks(depths)
    ax.set_xlabel("Sequencing depth (X)")
    ax.set_ylabel("Jobs with output (%)")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="lower left", title="divergence",
              title_fontsize=10)

    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0, h_pad=2.0)
    fig.savefig(ASSETS / "fig_reference_guided_depth.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


DEPTH_AXIS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30]


def fig_all_methods_depth(data: dict[str, pd.DataFrame],
                          denovo: dict[str, pd.DataFrame],
                          preliminary: frozenset = frozenset()) -> None:
    """Output rate and conditional F1 by depth for all seven methods.

    Reference-guided methods were run at 1--8X only; the de novo assemblers
    additionally at 10--30X. Depths are placed at evenly spaced categorical
    positions with a visible break after 8X.
    """
    _vector_rc()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    # Wider spacing after the break keeps the 10/15/20/30 labels apart.
    pos = {d: (i if d <= 8 else 8 + 0.9 + 1.5 * (i - 8)) for i, d in enumerate(DEPTH_AXIS)}
    series = [(n, data[n], c, m, ls) for n, _p, _it, c, m, ls in METHODS]
    series += [(n, denovo[n], c, m, ls) for n, _d, c, m, ls in DENOVO]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.3))
    for name, df, color, marker, ls in series:
        depths = [d for d in DEPTH_AXIS if d in set(df["depth"])]
        g = df.groupby("depth")["f1"]
        rate = (100 * g.apply(lambda v: v.notna().mean())).reindex(depths)
        cond = g.mean().reindex(depths)
        x = [pos[d] for d in depths]
        conditional = name in ("MIA", "NOVOPlasty", "GetOrganelle")
        style = dict(color=color, marker=marker, linestyle=ls, linewidth=1.8, markersize=6,
                     markeredgecolor=color, markeredgewidth=1.3,
                     markerfacecolor="white" if conditional else color)
        axes[0].plot(x, rate.values, label=name, **style)
        axes[1].plot(x, cond.values, label=name, **style)
    for ax in axes:
        ax.set_xticks([pos[d] for d in DEPTH_AXIS], [str(d) for d in DEPTH_AXIS], fontsize=10)
        ax.axvline((pos[8] + pos[10]) / 2, color="0.6", linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_xlabel("Sequencing depth (X)")
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-3 if ax is axes[0] else -0.03, 103 if ax is axes[0] else 1.03)
        ax.text((pos[10] + pos[30]) / 2, 1.0, "de novo only",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, color="0.4")
    axes[0].set_title("Jobs with scoreable output", fontsize=12, pad=16)
    panel_label(axes[0], "a")
    axes[0].set_ylabel("Jobs with output (%)")
    axes[1].set_title("Mean F1 among jobs with output", fontsize=12, pad=16)
    panel_label(axes[1], "b")
    axes[1].set_ylabel("Mean F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.5)
    if preliminary:
        for ax in axes:
            stamp(ax, "PRELIMINARY\n" + ", ".join(sorted(preliminary)) + " not normalized",
                  fontsize=12)
    fig.savefig(ASSETS / "fig_external_tools_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    fig_initial_state_heatmap()
    fig_divergence_boxplot()
    fig_hybrid_vs_plain()
    plt.rcdefaults()  # make_hybrid_comparison_figure applies a seaborn theme on import
    fig_read_length()
    plt.rcdefaults()
    data = {name: load_method(path, it) for name, path, it, *_ in METHODS}
    fig_reference_guided_depth(data)
    plt.rcdefaults()
    loaded = {name: load_denovo(d, allow_preliminary=True) for name, d, *_ in DENOVO}
    denovo = {n: df for n, (df, _) in loaded.items()}
    prelim = frozenset(n for n, (_, p) in loaded.items() if p)
    fig_cross_method_heatmaps([(n, data[n]) for n, *_ in METHODS]
                              + [(n, denovo[n]) for n, *_ in DENOVO], prelim)
    plt.rcdefaults()
    fig_all_methods_depth(data, denovo, prelim)
    for f in sorted(ASSETS.glob("fig_*.pdf")):
        print(f)


if __name__ == "__main__":
    main()
