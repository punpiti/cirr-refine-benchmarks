"""One-factor sensitivity analysis for ASEM's fixed parameters.

The design uses one representative ordered pair per divergence level, four
sequencing depths, and three deterministic replicates. Each method receives
byte-identical reads within a target/depth/replicate cell.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from baseline_ieee_access import run_asem
from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed
from phase1_species import divergence_level, fasta_path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "parameter_sensitivity")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
DEPTHS = (1, 2, 4, 8)
N_REPLICATES = 3
N_PARALLEL_RUNS = int(os.environ.get("SENSITIVITY_PARALLEL_RUNS", "12"))
PAIRS = (
    ("Saimiri_boliviensis", "Saimiri_sciureus"),
    ("Gorilla_gorilla", "Homo_sapiens"),
    ("Varecia_variegata", "Homo_sapiens"),
)
SETTINGS = tuple(
    [("tau", value, value, 0.10, 6) for value in (0.30, 0.40, 0.50, 0.60, 0.70)]
    + [("w", value, 0.50, value, 6) for value in (0.05, 0.20, 0.30)]
    + [("t_max", value, 0.50, 0.10, value) for value in (1, 3, 10)]
)


def _run_one(job: tuple[str, float | int, float, float, int, str, str, int, int]) -> dict:
    parameter, tested_value, tau, w, max_iterations, ref_name, target_name, depth, replicate = job
    reference = read_fasta(fasta_path(ref_name))[1]
    target = read_fasta(fasta_path(target_name))[1]
    reads = simulate_shotgun_reads(
        target,
        float(depth),
        150,
        stable_seed(target_name, depth, replicate),
    )
    result = run_asem(
        reference,
        reads,
        tau=tau,
        w=w,
        max_iterations=max_iterations,
        n_workers=1,
    )
    return {
        "parameter": parameter,
        "tested_value": tested_value,
        "tau": tau,
        "w": w,
        "t_max": max_iterations,
        "reference": ref_name,
        "target": target_name,
        "divergence": divergence_level(ref_name, target_name),
        "depth": depth,
        "replicate": replicate,
        "n_reads": len(reads),
        "last_iteration": result.history[-1].iteration,
        "hit_iteration_cap": int(result.history[-1].iteration == max_iterations),
        "theta_len": len(result.theta),
        **evaluate_theta_vs_target(result.theta, target),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = [
        (*setting, reference, target, depth, replicate)
        for setting in SETTINGS
        for reference, target in PAIRS
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
    ]
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_PARALLEL_RUNS) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 25 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] elapsed={time.time() - started:.0f}s", flush=True)
    rows.sort(key=lambda row: (
        row["parameter"], float(row["tested_value"]), row["reference"],
        row["target"], row["depth"], row["replicate"],
    ))
    with open(RESULTS_CSV, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
