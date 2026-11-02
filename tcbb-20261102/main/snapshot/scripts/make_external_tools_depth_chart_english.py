"""Two-panel external-tool figure for the English submission.

Panel A reports success rates over all attempted jobs; panel B reports F1
conditional on producing a scaffold. Keeping these quantities separate avoids
making a success-only mean look like unconditional performance. Both tools use
the same 30 ordered pairs and three replicates at every displayed depth.
"""

from __future__ import annotations

import os
import tempfile

_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "genome-figure-cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(
    SCRIPT_DIR, "..", "..", "bmc-genomics", "assets", "fig_external_tools_depth_english.png"
)
NOVO_CSV = os.path.join(SCRIPT_DIR, "results", "ext_novoplasty", "grid_results.csv")
GET_CSV = os.path.join(SCRIPT_DIR, "results", "ext_getorganelle", "grid_results.csv")
GET_NORMALIZED_CSV = os.path.join(
    SCRIPT_DIR, "results", "ext_getorganelle", "normalized_successful_rerun.csv"
)
LABELS = ["1--8X", "10X", "15X", "20X", "30X"]
KEYS = ["1-8X", 10, 15, 20, 30]


def summarize(
    path: str, normalized_path: str | None = None
) -> tuple[list[float], list[float], list[str]]:
    df = pd.read_csv(path)
    if normalized_path is not None:
        keys = ["reference", "target", "depth", "replicate"]
        normalized = pd.read_csv(normalized_path)
        normalized = normalized[normalized["error"].fillna("") == ""]
        expected = df["f1"].notna().sum()
        if len(normalized) != expected or normalized.duplicated(keys).any():
            raise ValueError(
                f"{normalized_path}: expected {expected} unique normalized successes, "
                f"found {len(normalized)}"
            )
        replacement = normalized.set_index(keys)["f1"]
        indexed = df.set_index(keys)
        indexed.loc[replacement.index, "f1"] = replacement
        df = indexed.reset_index()
    df = df.copy()
    df["bucket"] = df["depth"].map(lambda value: value if value > 8 else "1-8X")
    grouped = df.groupby("bucket")["f1"]
    mean_f1 = grouped.mean().to_dict()
    success = grouped.apply(lambda values: values.notna().sum()).to_dict()
    total = grouped.size().to_dict()
    rates = [success.get(key, 0) / total[key] for key in KEYS]
    means = [mean_f1.get(key, float("nan")) for key in KEYS]
    counts = [f"{success.get(key, 0)}/{total[key]}" for key in KEYS]
    return rates, means, counts


def main() -> None:
    novo_rate, novo_f1, novo_counts = summarize(NOVO_CSV)
    get_rate, get_f1, get_counts = summarize(GET_CSV, GET_NORMALIZED_CSV)
    x = range(len(LABELS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    colors = {"NOVOPlasty": "#2a78d6", "GetOrganelle": "#eb6834"}
    series = [
        ("NOVOPlasty", novo_rate, novo_f1, novo_counts),
        ("GetOrganelle", get_rate, get_f1, get_counts),
    ]
    for name, rates, means, counts in series:
        axes[0].plot(x, rates, marker="o", linewidth=2, color=colors[name], label=name)
        axes[1].plot(x, means, marker="o", linewidth=2, color=colors[name], label=name)
        for index, (rate, count) in enumerate(zip(rates, counts)):
            axes[0].annotate(count, (index, rate), xytext=(0, 6), textcoords="offset points",
                             ha="center", fontsize=7, color=colors[name])

    axes[0].set_title("(a) Scaffold success rate")
    axes[0].set_ylabel("successful jobs / attempted jobs")
    axes[0].set_ylim(-0.04, 1.08)
    axes[1].set_title("(b) F1 conditional on success")
    axes[1].set_ylabel("mean F1 among successful jobs")
    axes[1].set_ylim(-0.04, 1.08)
    for ax in axes:
        ax.set_xticks(list(x), LABELS)
        ax.set_xlabel("sequencing depth")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
