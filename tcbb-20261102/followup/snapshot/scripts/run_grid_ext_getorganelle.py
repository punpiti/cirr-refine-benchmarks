"""Full Phase-1 experimental grid for the external tool GetOrganelle
(replaces MITObim -- see ext_getorganelle.py module docstring), matching
baseline 1/2/3 and ext_novoplasty's grid design (same ordered pairs, same
stable_seed formula) for direct, read-for-read comparability.

Same depth design as run_grid_ext_novoplasty.py: 1-8X (baselines' native
range) plus 10/15/20/30X (head-to-head once there's enough depth to
assemble/circularize) -- see that module's docstring for the coverage-floor
rationale, which applies to any de novo/seed-and-extend assembler, not just
NOVOPlasty specifically.

Outputs (in scripts/refine_reference/results/ext_getorganelle/):
  - grid_results.csv : one row per (pair, depth, replicate); failed jobs get
                        f1=NaN and an "error" column populated.

Resumable: same append-and-skip-completed pattern as the other grid runners.
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

from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed, write_fasta  # noqa: E402
from ext_getorganelle import GetOrganelleFailure, run_getorganelle  # noqa: E402
from phase1_species import divergence_level, fasta_path, ordered_pairs  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "ext_getorganelle")
GRID_RESULTS_CSV = os.path.join(RESULTS_DIR, "grid_results.csv")
DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30]
N_REPLICATES = 3
N_PARALLEL_RUNS = 14
GETORGANELLE_TIMEOUT_S = 600

_SEQ_CACHE: dict[str, str] = {}


def _get_seq(name: str) -> str:
    if name not in _SEQ_CACHE:
        _, seq = read_fasta(fasta_path(name))
        _SEQ_CACHE[name] = seq
    return _SEQ_CACHE[name]


def _run_one(args: tuple[str, str, int, int]) -> dict:
    ref_name, target_name, depth, replicate = args
    s = _get_seq(ref_name)
    t = _get_seq(target_name)
    seed = stable_seed(target_name, depth, replicate)
    reads = simulate_shotgun_reads(t, depth=float(depth), read_length=150, seed=seed)

    base = {
        "reference": ref_name,
        "target": target_name,
        "divergence": divergence_level(ref_name, target_name),
        "depth": depth,
        "replicate": replicate,
        "n_reads": len(reads),
    }
    t0 = time.time()
    try:
        theta = run_getorganelle(s, reads, quality_seed=seed, timeout=GETORGANELLE_TIMEOUT_S)
    except GetOrganelleFailure as exc:
        return {
            **base, "theta_len": 0, "runtime_s": time.time() - t0, "error": str(exc)[:300],
            "identity_pct": float("nan"), "recall": float("nan"), "precision": float("nan"),
            "f1": float("nan"), "c_theta": 0, "w_theta": 0, "u_t": 0,
        }
    assemblies_dir = os.path.join(RESULTS_DIR, "assemblies")
    os.makedirs(assemblies_dir, exist_ok=True)
    write_fasta(
        os.path.join(assemblies_dir, f"{ref_name}__{target_name}__d{depth}__r{replicate}.fasta"),
        f"GetOrganelle {ref_name}->{target_name} depth={depth} replicate={replicate}",
        theta,
    )
    scores = evaluate_theta_vs_target(
        theta, t, normalize_strand=True, normalize_circular_origin=True
    )
    return {
        **base, "theta_len": len(theta), "runtime_s": time.time() - t0, "error": "",
        **scores,
    }


_GRID_FIELDNAMES = [
    "reference", "target", "divergence", "depth", "replicate",
    "n_reads", "theta_len", "runtime_s", "error",
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

    all_jobs = [
        (ref_name, target_name, depth, replicate)
        for ref_name, target_name in ordered_pairs()
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
    ]
    completed = _load_completed_jobs()
    jobs = [j for j in all_jobs if j not in completed]
    print(f"=== ext_getorganelle grid -- {len(all_jobs)} total combinations, "
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
            row = fut.result()
            writer.writerow(row)
            out_fh.flush()
            total_rows_written += 1
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
            err_flag = " ERROR" if row["error"] else ""
            print(
                f"  [{done}/{len(jobs)} {100*done/len(jobs):5.1f}%] "
                f"finished {job[0]} vs {job[1]} depth={job[2]}X rep={job[3]} "
                f"f1={row['f1']:.4f}{err_flag} "
                f"| {elapsed:6.0f}s elapsed, ~{eta:6.0f}s remaining, "
                f"{rate:.4f} runs/s",
                flush=True,
            )

    out_fh.close()
    total_elapsed = time.time() - run_t0
    print(f"\n=== DONE: appended {total_rows_written} rows to grid_results.csv this run "
          f"(total wall time {total_elapsed:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
