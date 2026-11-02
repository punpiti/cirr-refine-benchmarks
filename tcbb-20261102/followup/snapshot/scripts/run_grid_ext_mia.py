"""MIA v1.0 on a matched representative subset of the Phase-1 grid.

One ordered pair per divergence level is evaluated at 1/2/4/8X using the
same stable replicate-0 read seed as the internal methods. A per-job timeout
is an explicit benchmark outcome because MIA can run for many minutes even
on these mtDNA-scale inputs. Set MIA_SCOPE=full to request the original
720-job grid. Results are resumable.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv  # noqa: E402
import fcntl  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed  # noqa: E402
from ext_mia import MiaFailure, run_mia  # noqa: E402
from phase1_species import divergence_level, fasta_path, ordered_pairs  # noqa: E402

MIA_SCOPE = os.environ.get("MIA_SCOPE", "representative").strip().lower()
if MIA_SCOPE not in {"representative", "full"}:
    raise ValueError("MIA_SCOPE must be 'representative' or 'full'")
RESULTS_NAME = "ext_mia_representative" if MIA_SCOPE == "representative" else "ext_mia"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", RESULTS_NAME)
GRID_RESULTS_CSV = os.path.join(RESULTS_DIR, "grid_results.csv")
DEPTHS = [1, 2, 4, 8] if MIA_SCOPE == "representative" else list(range(1, 9))
N_REPLICATES = 1 if MIA_SCOPE == "representative" else 3
REPRESENTATIVE_PAIRS = [
    ("Saimiri_boliviensis", "Saimiri_sciureus"),
    ("Homo_sapiens", "Gorilla_gorilla"),
    ("Homo_sapiens", "Varecia_variegata"),
]
N_PARALLEL_RUNS = int(os.environ.get("MIA_PARALLEL_RUNS", "6"))
MIA_TIMEOUT_S = int(os.environ.get("MIA_TIMEOUT_S", "180"))

_SEQ_CACHE: dict[str, str] = {}
_LOCK_HANDLE = None


def _get_seq(name: str) -> str:
    if name not in _SEQ_CACHE:
        _, sequence = read_fasta(fasta_path(name))
        _SEQ_CACHE[name] = sequence
    return _SEQ_CACHE[name]


def _run_one(args: tuple[str, str, int, int]) -> dict:
    ref_name, target_name, depth, replicate = args
    reference = _get_seq(ref_name)
    target = _get_seq(target_name)
    seed = stable_seed(target_name, depth, replicate)
    reads = simulate_shotgun_reads(target, float(depth), 150, seed)
    base = {
        "reference": ref_name,
        "target": target_name,
        "divergence": divergence_level(ref_name, target_name),
        "depth": depth,
        "replicate": replicate,
        "n_reads": len(reads),
    }
    started = time.time()
    try:
        theta = run_mia(reference, reads, timeout=MIA_TIMEOUT_S)
    except MiaFailure as exc:
        return {
            **base,
            "theta_len": 0,
            "n_ambiguous": 0,
            "runtime_s": time.time() - started,
            "error": str(exc)[:500],
            "identity_pct": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
            "c_theta": 0,
            "w_theta": 0,
            "u_t": 0,
        }
    return {
        **base,
        "theta_len": len(theta),
        "n_ambiguous": sum(symbol not in "ACGT" for symbol in theta),
        "runtime_s": time.time() - started,
        "error": "",
        **evaluate_theta_vs_target(
            theta, target, normalize_strand=True, normalize_circular_origin=True
        ),
    }


FIELDNAMES = [
    "reference", "target", "divergence", "depth", "replicate", "n_reads",
    "theta_len", "n_ambiguous", "runtime_s", "error", "identity_pct",
    "recall", "precision", "f1", "c_theta", "w_theta", "u_t",
]


def _completed_jobs() -> set[tuple[str, str, int, int]]:
    if not os.path.exists(GRID_RESULTS_CSV):
        return set()
    with open(GRID_RESULTS_CSV, newline="") as fh:
        return {
            (row["reference"], row["target"], int(row["depth"]), int(row["replicate"]))
            for row in csv.DictReader(fh)
        }


def main() -> None:
    global _LOCK_HANDLE
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lock_path = os.path.join(RESULTS_DIR, ".grid.lock")
    _LOCK_HANDLE = open(lock_path, "w")
    try:
        fcntl.flock(_LOCK_HANDLE, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another MIA {MIA_SCOPE} grid runner holds {lock_path}; exiting", flush=True)
        return
    pairs = REPRESENTATIVE_PAIRS if MIA_SCOPE == "representative" else ordered_pairs()
    all_jobs = [
        (reference, target, depth, replicate)
        for reference, target in pairs
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
    ]
    completed = _completed_jobs()
    jobs = [job for job in all_jobs if job not in completed]
    print(
        f"=== MIA grid: {len(all_jobs)} total, {len(completed)} completed, "
        f"{len(jobs)} remaining, {N_PARALLEL_RUNS} parallel ===",
        flush=True,
    )
    if not jobs:
        return

    is_new = not os.path.exists(GRID_RESULTS_CSV)
    started = time.time()
    with open(GRID_RESULTS_CSV, "a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
            output.flush()
        with ProcessPoolExecutor(max_workers=N_PARALLEL_RUNS) as pool:
            futures = {pool.submit(_run_one, job): job for job in jobs}
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                writer.writerow(row)
                output.flush()
                elapsed = time.time() - started
                rate = done / elapsed
                eta = (len(jobs) - done) / rate if rate else float("nan")
                error = " ERROR" if row["error"] else ""
                print(
                    f"[{done}/{len(jobs)}] {row['reference']} -> {row['target']} "
                    f"{row['depth']}X rep={row['replicate']} F1={row['f1']:.4f}{error}; "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
