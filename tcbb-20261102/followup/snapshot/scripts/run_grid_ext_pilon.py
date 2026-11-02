"""Pilon v1.24 on the Phase-1 benchmark grid.

The default remains the historical representative subset. Set
``PILON_SCOPE=full`` to run all 30 ordered pairs, depths 1--8X, and three
replicates with the same deterministic read seeds as the internal methods.
Both scopes are resumable and write to separate result directories.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(__file__))

from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed
from ext_pilon import PilonFailure, run_pilon
from phase1_species import divergence_level, fasta_path, ordered_pairs

PILON_SCOPE = os.environ.get("PILON_SCOPE", "representative").strip().lower()
if PILON_SCOPE not in {"representative", "full"}:
    raise ValueError("PILON_SCOPE must be 'representative' or 'full'")
RESULTS_NAME = "ext_pilon_representative" if PILON_SCOPE == "representative" else "ext_pilon"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", RESULTS_NAME)
RESULTS_CSV = os.path.join(RESULTS_DIR, "grid_results.csv")
DEPTHS = [1, 2, 4, 8] if PILON_SCOPE == "representative" else list(range(1, 9))
N_REPLICATES = 1 if PILON_SCOPE == "representative" else 3
REPRESENTATIVE_PAIRS = [
    ("Saimiri_boliviensis", "Saimiri_sciureus"),
    ("Homo_sapiens", "Gorilla_gorilla"),
    ("Homo_sapiens", "Varecia_variegata"),
]
WORKERS = int(os.environ.get("PILON_PARALLEL_RUNS", "6"))
TIMEOUT = int(os.environ.get("PILON_TIMEOUT_S", "180"))
FIELDS = [
    "reference", "target", "divergence", "depth", "replicate", "n_reads",
    "theta_len", "runtime_s", "error", "identity_pct", "recall", "precision",
    "f1", "c_theta", "w_theta", "u_t",
]


def _run_one(job: tuple[str, str, int, int]) -> dict:
    reference_name, target_name, depth, replicate = job
    _, reference = read_fasta(fasta_path(reference_name))
    _, target = read_fasta(fasta_path(target_name))
    reads = simulate_shotgun_reads(
        target, float(depth), 150, stable_seed(target_name, depth, replicate)
    )
    row = {
        "reference": reference_name,
        "target": target_name,
        "divergence": divergence_level(reference_name, target_name),
        "depth": depth,
        "replicate": replicate,
        "n_reads": len(reads),
    }
    started = time.time()
    try:
        theta = run_pilon(reference, reads, timeout=TIMEOUT)
    except PilonFailure as exc:
        return {
            **row, "theta_len": 0, "runtime_s": time.time() - started,
            "error": str(exc)[:500], "identity_pct": float("nan"),
            "recall": float("nan"), "precision": float("nan"), "f1": float("nan"),
            "c_theta": 0, "w_theta": 0, "u_t": 0,
        }
    return {
        **row, "theta_len": len(theta), "runtime_s": time.time() - started,
        "error": "", **evaluate_theta_vs_target(
            theta, target, normalize_strand=True, normalize_circular_origin=True
        ),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs = REPRESENTATIVE_PAIRS if PILON_SCOPE == "representative" else ordered_pairs()
    jobs = [
        (reference, target, depth, replicate)
        for reference, target in pairs
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
    ]
    completed = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as handle:
            completed = {
                (row["reference"], row["target"], int(row["depth"]), int(row["replicate"]))
                for row in csv.DictReader(handle)
            }
    pending = [job for job in jobs if job not in completed]
    print(
        f"=== Pilon {PILON_SCOPE} grid: {len(jobs)} total, "
        f"{len(completed)} completed, {len(pending)} remaining, {WORKERS} parallel ===",
        flush=True,
    )
    if not pending:
        return
    new_file = not os.path.exists(RESULTS_CSV)
    started = time.time()
    with open(RESULTS_CSV, "a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_run_one, job): job for job in pending}
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                writer.writerow(row)
                output.flush()
                suffix = " ERROR" if row["error"] else ""
                print(
                    f"[{done}/{len(pending)}] {row['reference']} -> {row['target']} "
                    f"{row['depth']}X F1={row['f1']:.4f}{suffix}; "
                    f"elapsed={time.time()-started:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
