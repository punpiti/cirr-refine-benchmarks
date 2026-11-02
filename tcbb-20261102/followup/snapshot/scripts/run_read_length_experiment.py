"""Read-length effect experiment, isolated from the depth-sweep grids
(run_grid_baseline*.py). Two questions, requested separately per
algorithm_sketch_iterative_refinement.md's "read length" row:

1. Fixed read length: how does each baseline's effectiveness change across
   realistic short-read lengths (50, 75, 100, 150, 250, 300bp), holding
   depth constant so the read-length effect isn't confounded with the
   depth effect already characterized in the main grids?
2. Variable read length: does a realistic post-QC length *distribution*
   (common.simulate_shotgun_reads_variable_length -- mostly full-length,
   a minority trimmed shorter) behave differently from any single fixed
   length, e.g. because a mix of lengths changes how often reads clip
   cleanly vs. get discarded?

Reduced scope vs. the main 30-pair grids (deliberately -- this is a
diagnostic sweep, not a benchmark): 3 representative pairs (one per
divergence level), depth fixed at 4X, 3 replicates, all three baselines.
The reported experiment uses 3 pairs x 7 length-conditions x 3 replicates x
3 baselines (ieee_access, ojemb, ecticon) = 189 jobs. The later legacy Hybrid
variant remains selectable through ``READ_LENGTH_METHODS`` but is not run by
default because it is not reported in the manuscript.
each cheap (~1-10s based on the main grids' per-job rates).

Output: scripts/refine_reference/results/read_length_experiment/results.csv
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv  # noqa: E402
import functools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

from asem_hybrid import run_asem_hybrid_loop  # noqa: E402
from baseline_ecticon import run_ecticon  # noqa: E402
from baseline_ieee_access import _align_read_ieee_access, run_asem as run_asem_ieee_access  # noqa: E402
from baseline_ojemb import run_asem as run_asem_ojemb  # noqa: E402
from common import (  # noqa: E402
    evaluate_theta_vs_target,
    read_fasta,
    simulate_shotgun_reads,
    simulate_shotgun_reads_variable_length,
    stable_seed,
)
from phase1_species import divergence_level, fasta_path  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "read_length_experiment")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

# One representative pair per divergence level (same choices used elsewhere
# in this project's write-ups, e.g. the ECTI-CON threshold sweep and
# baseline1/2's "hardest pair" comparisons).
PAIRS = [
    ("Saimiri_boliviensis", "Saimiri_sciureus"),  # same-genus
    ("Homo_sapiens", "Gorilla_gorilla"),          # same-family
    ("Homo_sapiens", "Varecia_variegata"),        # same-order
]
DEPTH = 4.0  # held constant so the read-length effect isn't confounded with depth
FIXED_LENGTHS = [50, 75, 100, 150, 250, 300]
N_REPLICATES = 3
N_PARALLEL_RUNS = int(os.environ.get("ASEM_READ_LENGTH_WORKERS", "2"))

_HYBRID_ALIGN_FN = functools.partial(_align_read_ieee_access, tau=0.5)  # matches run_grid_baseline_hybrid.py's TAU

_ALL_BASELINES = {
    "ieee_access": lambda theta_init, reads: run_asem_ieee_access(theta_init, reads),
    "ojemb": lambda theta_init, reads: run_asem_ojemb(theta_init, reads),
    "ecticon": lambda theta_init, reads: run_ecticon(theta_init, reads),
    "hybrid": lambda theta_init, reads: run_asem_hybrid_loop(
        theta_init, reads, align_read_fn=_HYBRID_ALIGN_FN, boundary_flank=None, max_iterations=6, n_workers=1
    ),
}
_requested_methods = os.environ.get(
    "READ_LENGTH_METHODS", "ieee_access,ojemb,ecticon"
).split(",")
BASELINES = {name: _ALL_BASELINES[name.strip()] for name in _requested_methods}

_SEQ_CACHE: dict[str, str] = {}


def _get_seq(name: str) -> str:
    if name not in _SEQ_CACHE:
        _, seq = read_fasta(fasta_path(name))
        _SEQ_CACHE[name] = seq
    return _SEQ_CACHE[name]


def _simulate_reads(target_seq: str, length_condition, seed: int) -> list[str]:
    if length_condition == "variable":
        return simulate_shotgun_reads_variable_length(target_seq, depth=DEPTH, seed=seed)
    return simulate_shotgun_reads(target_seq, depth=DEPTH, read_length=length_condition, seed=seed)


def _run_one(args: tuple[str, str, str, object, int]) -> dict:
    baseline_name, ref_name, target_name, length_condition, replicate = args
    s = _get_seq(ref_name)
    t = _get_seq(target_name)
    seed = stable_seed("read_length_experiment", target_name, length_condition, replicate)
    reads = _simulate_reads(t, length_condition, seed)

    theta = BASELINES[baseline_name](s, reads).theta
    scores = evaluate_theta_vs_target(theta, t)
    return {
        "baseline": baseline_name,
        "reference": ref_name,
        "target": target_name,
        "divergence": divergence_level(ref_name, target_name),
        "length_condition": length_condition,
        "replicate": replicate,
        "depth": DEPTH,
        "n_reads": len(reads),
        "mean_read_len": sum(len(r) for r in reads) / len(reads),
        **scores,
    }


_FIELDNAMES = [
    "baseline", "reference", "target", "divergence", "length_condition", "replicate",
    "depth", "n_reads", "mean_read_len",
    "identity_pct", "recall", "precision", "f1", "c_theta", "w_theta", "u_t",
]


def _load_completed() -> set[tuple]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    completed = set()
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            completed.add((row["baseline"], row["reference"], row["target"], row["length_condition"], int(row["replicate"])))
    return completed


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    length_conditions = [*FIXED_LENGTHS, "variable"]
    all_jobs = [
        (baseline_name, ref_name, target_name, length_condition, replicate)
        for baseline_name in BASELINES
        for ref_name, target_name in PAIRS
        for length_condition in length_conditions
        for replicate in range(N_REPLICATES)
    ]
    completed = _load_completed()
    jobs = [j for j in all_jobs if (j[0], j[1], j[2], str(j[3]), j[4]) not in completed]
    print(f"=== read_length_experiment -- {len(all_jobs)} total combinations, "
          f"{len(completed)} already done (resumed), {len(jobs)} remaining, "
          f"{N_PARALLEL_RUNS} in parallel ===", flush=True)

    if not jobs:
        print("nothing to do.", flush=True)
        return

    is_new_file = not os.path.exists(RESULTS_CSV)
    out_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=_FIELDNAMES)
    if is_new_file:
        writer.writeheader()
        out_fh.flush()

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_PARALLEL_RUNS) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            row = fut.result()
            writer.writerow(row)
            out_fh.flush()
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
            print(
                f"  [{done}/{len(jobs)} {100*done/len(jobs):5.1f}%] "
                f"{job[0]} {job[1]} vs {job[2]} len={job[3]} rep={job[4]} f1={row['f1']:.4f} "
                f"| {elapsed:6.0f}s elapsed, ~{eta:5.0f}s remaining, {rate:.2f} runs/s",
                flush=True,
            )

    out_fh.close()
    print(f"\n=== DONE ({time.time()-t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
