"""Full Phase-1 experimental grid for Baseline 2 (OJEMB ASEM,
+recursive partial alignment), reproducing the original paper's
experimental design: all ordered (reference, target) pairs x depths 1-8X x
3 replicates, with per-iteration (1-6) metrics recorded from each run's
history.

Structurally identical to run_grid_baseline1.py (same dataset, same seeds
via common.stable_seed, same job grid, same resumability/parallelism
design) -- the only difference is importing baseline_ojemb.run_asem instead
of baseline_ieee_access.run_asem, and writing to a separate results
subdirectory so the two baselines' outputs never collide and can be
compared read-for-read (both grids simulate identical reads per
(target, depth, replicate) job since they share the same stable_seed
formula).

Outputs (in scripts/refine_reference/results/baseline2_ojemb/):
  - initial_state.csv : S vs T metrics per pair, no refinement (identical
                         to baseline 1's -- refinement hasn't run yet -- but
                         recomputed here for a self-contained results dir)
  - grid_results.csv  : one row per (pair, depth, replicate, iteration)

Resumable: grid_results.csv is appended to as each (pair, depth, replicate)
job finishes (not batched), and is read back on startup to skip
already-completed jobs. Safe to kill and re-run at any point.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

from baseline_ojemb import run_asem  # noqa: E402
from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed  # noqa: E402
from phase1_species import SPECIES, divergence_level, fasta_path, ordered_pairs  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "baseline2_ojemb")
INITIAL_STATE_CSV = os.path.join(RESULTS_DIR, "initial_state.csv")
GRID_RESULTS_CSV = os.environ.get(
    "ASEM_GRID_RESULTS_CSV", os.path.join(RESULTS_DIR, "grid_results.csv")
)
DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
N_REPLICATES = 3
MAX_ITERATIONS = 6
N_PARALLEL_RUNS = int(os.environ.get("ASEM_GRID_WORKERS", "14"))

_SEQ_CACHE: dict[str, str] = {}


def _get_seq(name: str) -> str:
    if name not in _SEQ_CACHE:
        _, seq = read_fasta(fasta_path(name))
        _SEQ_CACHE[name] = seq
    return _SEQ_CACHE[name]


def compute_initial_state() -> None:
    if os.path.exists(INITIAL_STATE_CSV):
        print(f"  {INITIAL_STATE_CSV} already exists, skipping (delete it to recompute)", flush=True)
        return

    all_pairs = [(r, t) for r in SPECIES for t in SPECIES]
    total = len(all_pairs)
    rows = []
    t0 = time.time()
    for done, (ref_name, target_name) in enumerate(all_pairs, start=1):
        s = _get_seq(ref_name)
        t = _get_seq(target_name)
        scores = evaluate_theta_vs_target(s, t)
        rows.append(
            {
                "reference": ref_name,
                "target": target_name,
                "divergence": "self" if ref_name == target_name else divergence_level(ref_name, target_name),
                **scores,
            }
        )
        elapsed = time.time() - t0
        print(f"  [initial-state] {done}/{total} ({100*done/total:.0f}%) "
              f"{ref_name} vs {target_name}  ({elapsed:.1f}s elapsed)", flush=True)
    _write_csv(INITIAL_STATE_CSV, rows)
    print(f"  wrote {len(rows)} rows to initial_state.csv ({time.time()-t0:.0f}s)\n", flush=True)


def _run_one(args: tuple[str, str, int, int]) -> list[dict]:
    ref_name, target_name, depth, replicate = args
    s = _get_seq(ref_name)
    t = _get_seq(target_name)
    # Same seed formula as baseline 1's grid runner: depends only on
    # (target, depth, replicate), so both baselines see byte-identical
    # simulated reads for the same job -- a real read-for-read comparison.
    seed = stable_seed(target_name, depth, replicate)
    reads = simulate_shotgun_reads(t, depth=float(depth), read_length=150, seed=seed)

    result = run_asem(
        theta_init=s,
        reads=reads,
        max_iterations=MAX_ITERATIONS,
        n_workers=1,
    )

    rows = []
    for stats, theta_snapshot in zip(result.history, result.theta_by_iteration):
        scores = evaluate_theta_vs_target(theta_snapshot, t)
        rows.append(
            {
                "reference": ref_name,
                "target": target_name,
                "divergence": divergence_level(ref_name, target_name),
                "depth": depth,
                "replicate": replicate,
                "iteration": stats.iteration,
                "n_reads": len(reads),
                "n_placed": stats.n_placed,
                "n_unplaced": stats.n_unplaced,
                "theta_len": len(theta_snapshot),
                **scores,
            }
        )
    return rows


_GRID_FIELDNAMES = [
    "reference", "target", "divergence", "depth", "replicate", "iteration",
    "n_reads", "n_placed", "n_unplaced", "theta_len",
    "identity_pct", "recall", "precision", "f1", "c_theta", "w_theta", "u_t",
]


def _load_completed_jobs() -> set[tuple[str, str, int, int]]:
    if not os.path.exists(GRID_RESULTS_CSV):
        return set()
    completed = set()
    with open(GRID_RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            completed.add((row["reference"], row["target"], int(row["depth"]), int(row["replicate"])))
    return completed


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_t0 = time.time()

    print("=== STAGE 1/2: initial-state (no refinement) metrics for all pairs ===", flush=True)
    compute_initial_state()

    all_jobs = [
        (ref_name, target_name, depth, replicate)
        for ref_name, target_name in ordered_pairs()
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
    ]
    completed = _load_completed_jobs()
    jobs = [j for j in all_jobs if j not in completed]
    print(f"=== STAGE 2/2: grid -- {len(all_jobs)} total combinations, "
          f"{len(completed)} already done (resumed), {len(jobs)} remaining, "
          f"{N_PARALLEL_RUNS} in parallel ===", flush=True)

    if not jobs:
        print("nothing to do.", flush=True)
        return

    is_new_file = not os.path.exists(GRID_RESULTS_CSV)
    out_fh = open(GRID_RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=_GRID_FIELDNAMES)
    if is_new_file:
        writer.writeheader()
        out_fh.flush()

    t0 = time.time()
    total_rows_written = 0
    with ProcessPoolExecutor(max_workers=N_PARALLEL_RUNS) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            rows = fut.result()
            writer.writerows(rows)
            out_fh.flush()
            total_rows_written += len(rows)
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
            print(
                f"  [{done}/{len(jobs)} {100*done/len(jobs):5.1f}%] "
                f"finished {job[0]} vs {job[1]} depth={job[2]}X rep={job[3]}  "
                f"| {elapsed:6.0f}s elapsed, ~{eta:6.0f}s remaining, "
                f"{rate:.3f} runs/s",
                flush=True,
            )

    out_fh.close()
    total_elapsed = time.time() - run_t0
    print(f"\n=== DONE: appended {total_rows_written} rows to grid_results.csv this run "
          f"(total wall time {total_elapsed:.0f}s) ===", flush=True)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
