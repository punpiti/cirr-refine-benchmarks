"""Characterize wall-clock runtime and peak memory of ASEM (both variants)
and SRSC, broken down by sequencing depth/read count and EM iteration
(runtime/peak memory by reference length, read count, coverage, and
iteration).

Reference length is not a variable in this study's scope (every experiment
uses mtDNA, ~16.5kb) -- that fixed scale is itself part of the paper's
declared scope (low-resource, organelle-scale refinement, not whole-genome),
so this script characterizes the two axes that do vary here: read count
(via sequencing depth) and iteration count.

Each configuration is run in a fresh subprocess so peak RSS is measured for
that run alone (not contaminated by whatever else happened earlier in a
long-lived process). Uses resource.getrusage(RUSAGE_SELF).ru_maxrss, sampled
from the child process just before it exits.

Usage: python3 profile_computational_cost.py

The output CSV has one row per (job, iteration): wall_s_total_run and
peak_rss_kb are per-job totals duplicated across every iteration row for
that job, not per-iteration values. Use make_computational_cost_table.py
to summarize this correctly (job-grain deduplication before averaging) --
averaging the raw CSV directly row-by-row silently over-weights jobs that
ran more EM iterations.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from phase1_species import fasta_path  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "computational_cost")
OUT_CSV = os.path.join(RESULTS_DIR, "cost_profile.csv")

DEPTHS = [1, 2, 4, 8]
N_REPLICATES = 3
METHODS = ["asem_no_recursion", "asem_recursion", "srsc"]

# One representative pair per divergence level (matches the paper's own
# "representative pair" convention used for the read-length experiment).
REPRESENTATIVE_PAIRS = {
    "same-genus": ("Saimiri_boliviensis", "Saimiri_sciureus"),
    "same-family": ("Homo_sapiens", "Gorilla_gorilla"),
    "same-order": ("Homo_sapiens", "Varecia_variegata"),
}

_WORKER_SNIPPET = r"""
import json, resource, sys, time
sys.path.insert(0, {script_dir!r})
from common import read_fasta, simulate_shotgun_reads, stable_seed, evaluate_theta_vs_target

ref_name, target_name, depth, replicate, method = {args!r}
_, s = read_fasta({ref_path!r})
_, t = read_fasta({target_path!r})
seed = stable_seed(target_name, depth, replicate)
reads = simulate_shotgun_reads(t, depth=float(depth), read_length=150, seed=seed)

t0 = time.time()
if method == "asem_no_recursion":
    from baseline_ieee_access import run_asem
    result = run_asem(theta_init=s, reads=reads, max_iterations=6, n_workers=1)
    thetas = list(zip((st.iteration for st in result.history), result.theta_by_iteration))
elif method == "asem_recursion":
    from baseline_ojemb import run_asem
    result = run_asem(theta_init=s, reads=reads, max_iterations=6, n_workers=1)
    thetas = list(zip((st.iteration for st in result.history), result.theta_by_iteration))
else:
    from baseline_ecticon import run_ecticon
    result_e = run_ecticon(reference=s, reads=reads)
    thetas = [(1, result_e.theta)]

# Stop the clock and sample peak RSS right after the algorithm itself
# returns, before any post-hoc scoring -- evaluate_theta_vs_target's
# semi-global scoring pass is reporting overhead, not part of the
# algorithm's own cost.
wall_s = time.time() - t0
peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

per_iter = [{{"iteration": it, "f1": evaluate_theta_vs_target(theta, t)["f1"]}} for it, theta in thetas]
print(json.dumps({{"wall_s": wall_s, "peak_rss_kb": peak_rss_kb, "n_reads": len(reads), "per_iter": per_iter}}))
"""


def _run_one(ref_name: str, target_name: str, depth: int, replicate: int, method: str) -> dict | None:
    script = _WORKER_SNIPPET.format(
        script_dir=os.path.dirname(__file__),
        args=(ref_name, target_name, depth, replicate, method),
        ref_path=fasta_path(ref_name),
        target_path=fasta_path(target_name),
    )
    python_bin = os.environ.get("GENOME_ENV_PYTHON", sys.executable)
    proc = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"    ERROR: {proc.stderr[-800:]}", flush=True)
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    jobs = [
        (div, ref, target, depth, rep, method)
        for div, (ref, target) in REPRESENTATIVE_PAIRS.items()
        for depth in DEPTHS
        for rep in range(N_REPLICATES)
        for method in METHODS
    ]
    total = len(jobs)
    t0 = time.time()
    for done, (div, ref, target, depth, rep, method) in enumerate(jobs, start=1):
        result = _run_one(ref, target, depth, rep, method)
        if result is None:
            continue
        for it_row in result["per_iter"]:
            rows.append({
                "divergence": div, "reference": ref, "target": target,
                "depth": depth, "replicate": rep, "method": method,
                "n_reads": result["n_reads"], "iteration": it_row["iteration"],
                "f1": it_row["f1"],
                "wall_s_total_run": result["wall_s"],
                "peak_rss_kb": result["peak_rss_kb"],
            })
        elapsed = time.time() - t0
        print(f"  [{done}/{total} {100*done/total:5.1f}%] {method} {div} depth={depth}X rep={rep} "
              f"n_reads={result['n_reads']} wall={result['wall_s']:.2f}s peak_rss={result['peak_rss_kb']/1024:.1f}MB "
              f"| {elapsed:.0f}s elapsed", flush=True)

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
