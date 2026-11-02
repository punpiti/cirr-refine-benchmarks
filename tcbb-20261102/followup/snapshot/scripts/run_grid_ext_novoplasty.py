"""Full Phase-1 experimental grid for the external tool NOVOPlasty, matching
baseline 1/2/3's grid design exactly (all ordered pairs x depths 1-8X x 3
replicates, same stable_seed formula) so the three ASEM/ECTI-CON baselines
and NOVOPlasty are directly, read-for-read comparable.

NOVOPlasty is a real subprocess (not pure-Python), so each job is much more
expensive than the internal baselines (~minutes, not milliseconds) and can
fail outright on divergent references at low depth -- failures are recorded
as a row with error info rather than crashing the grid.

Outputs (in scripts/refine_reference/results/ext_novoplasty/):
  - grid_results.csv : one row per (pair, depth, replicate); failed jobs get
                        f1=NaN and an "error" column populated.

Resumable: grid_results.csv is appended to as each job finishes, and is read
back on startup to skip already-completed jobs (same pattern as
run_grid_baseline3.py).
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
from ext_novoplasty import NovoplastyFailure, run_novoplasty  # noqa: E402
from phase1_species import divergence_level, fasta_path, ordered_pairs  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "ext_novoplasty")
GRID_RESULTS_CSV = os.path.join(RESULTS_DIR, "grid_results.csv")
# 1-8X matches the baselines' native grid exactly (read-for-read comparable
# via the shared stable_seed formula) and is expected to mostly hit
# NOVOPlasty's coverage floor -- that failure itself is the Phase 3
# "coverage below usable threshold" finding called for in BMC_REBUILD_PLAN.md.
# 10-30X is added on top so there is also a genuine head-to-head once
# NOVOPlasty has enough depth to actually extend/circularize.
DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30]
N_REPLICATES = 3
N_PARALLEL_RUNS = 14
NOVOPLASTY_TIMEOUT_S = int(os.environ.get("NOVOPLASTY_TIMEOUT_S", "300"))  # override for normalized reruns of successes that originally needed the longer retry-pass timeout; every observed successful job finishes well under this (max ~700s only rarely; most under 300s); a run past this is treated as a hang and skipped as an error row rather than blocking the grid for up to 900s
SNAPSHOT_EVERY = 50  # take a timestamped backups/ copy of grid_results.csv every N completions, so there's something stable to check from another machine mid-run without relying on the live file

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
    # Same seed formula as baseline 1/2/3's grid runners: identical simulated
    # reads per (target, depth, replicate) job across every method compared.
    seed = stable_seed(target_name, depth, replicate)
    reads = simulate_shotgun_reads(t, depth=float(depth), read_length=150, seed=seed)

    project = f"g{seed:08x}"
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
        theta = run_novoplasty(s, reads, project=project, timeout=NOVOPLASTY_TIMEOUT_S)
    except NovoplastyFailure as exc:
        return {
            **base, "theta_len": 0, "runtime_s": time.time() - t0, "error": str(exc)[:300],
            "identity_pct": float("nan"), "recall": float("nan"), "precision": float("nan"),
            "f1": float("nan"), "c_theta": 0, "w_theta": 0, "u_t": 0,
        }
    assemblies_dir = os.path.join(RESULTS_DIR, "assemblies")
    os.makedirs(assemblies_dir, exist_ok=True)
    write_fasta(
        os.path.join(assemblies_dir, f"{ref_name}__{target_name}__d{depth}__r{replicate}.fasta"),
        f"NOVOPlasty {ref_name}->{target_name} depth={depth} replicate={replicate}",
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
    import grid_run_guard as guard

    os.makedirs(RESULTS_DIR, exist_ok=True)
    guard.acquire_lock(RESULTS_DIR, "run_grid_ext_novoplasty.py")
    guard.snapshot_csv(GRID_RESULTS_CSV)
    run_t0 = time.time()

    all_jobs = [
        (ref_name, target_name, depth, replicate)
        for ref_name, target_name in ordered_pairs()
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
    ]
    completed = _load_completed_jobs()
    jobs = [j for j in all_jobs if j not in completed]
    guard.log(f"=== ext_novoplasty grid -- {len(all_jobs)} total combinations, "
              f"{len(completed)} already done (resumed), {len(jobs)} remaining, "
              f"{N_PARALLEL_RUNS} in parallel ===")

    if not jobs:
        guard.log("nothing to do.")
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
            guard.log(
                f"  [{done}/{len(jobs)} {100*done/len(jobs):5.1f}%] "
                f"finished {job[0]} vs {job[1]} depth={job[2]}X rep={job[3]} "
                f"f1={row['f1']:.4f}{err_flag} "
                f"| {elapsed:6.0f}s elapsed, ~{eta:6.0f}s remaining, "
                f"{rate:.4f} runs/s"
            )
            if done % SNAPSHOT_EVERY == 0:
                out_fh.flush()
                guard.snapshot_csv(GRID_RESULTS_CSV)

    out_fh.close()
    guard.snapshot_csv(GRID_RESULTS_CSV)
    total_elapsed = time.time() - run_t0
    guard.log(f"=== DONE: appended {total_rows_written} rows to grid_results.csv this run "
              f"(total wall time {total_elapsed:.0f}s) ===")


if __name__ == "__main__":
    main()
