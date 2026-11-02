"""All-30-species initial-state (no refinement) pairwise matrix + Figure-5
style phylogeny-clustered heatmap, using the full original IEEE Access
panel (full_panel_species.py) rather than the 6-species Phase-1 subset
used for the actual iterative-refinement grid.

Outputs to scripts/refine_reference/results/full_panel/:
  - initial_state_30species.csv
  - figures/full_panel_clustermap_<metric>.png / _heatmap_<metric>.png
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

from common import evaluate_theta_vs_target, read_fasta  # noqa: E402
from full_panel_species import list_species  # noqa: E402
from make_report import METRICS, _clustermap, _heatmap  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "full_panel")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CSV_PATH = os.path.join(RESULTS_DIR, "initial_state_30species.csv")
N_WORKERS = 4  # modest, so this doesn't starve the concurrently-running experiment grid


def _score_pair(args: tuple[str, str, str, str]) -> dict:
    ref_name, ref_path, target_name, target_path = args
    _, s = read_fasta(ref_path)
    _, t = read_fasta(target_path)
    scores = evaluate_theta_vs_target(s, t)
    return {"reference": ref_name, "target": target_name, **scores}


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    species = list_species()
    names = list(species.keys())
    jobs = [
        (r, species[r], t, species[t])
        for r in names
        for t in names
    ]
    print(f"computing {len(jobs)} pairs ({len(names)} species) with {N_WORKERS} workers...", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_score_pair, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
                print(f"  {done}/{len(jobs)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}", flush=True)

    df = pd.DataFrame(rows)
    for metric in METRICS:
        _heatmap(
            df, metric, names,
            title=f"All 30 species, initial state (no refinement): {metric}",
            out_path=os.path.join(FIG_DIR, f"full_panel_heatmap_{metric}.png"),
        )
        _clustermap(
            df, metric, names,
            title=f"All 30 species, initial state (no refinement): {metric}, phylogeny-clustered",
            out_path=os.path.join(FIG_DIR, f"full_panel_clustermap_{metric}.png"),
        )
    print(f"wrote figures to {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
