"""Read-length experiment extended to every applicable comparator.

The v1.3.0 read-length experiment (run_read_length_experiment.py) covered
only ASEM, ASEM with recursive alignment, and SRSC. This runner adds
ASEM-Hybrid (boundary variant, as in the main grid), Pilon, MIA, NOVOPlasty
and GetOrganelle on byte-identical reads: same three representative pairs,
4X depth, six fixed lengths plus the variable-length condition, three
replicates, and the same read seed
stable_seed("read_length_experiment", target, length_condition, replicate).
The internal methods are rerun here too so every method in the output shares
one code snapshot.

Prespecified tool adaptations (fixed before any result was seen):
  * NOVOPlasty: "Read Length" is set to the simulated length (150 for the
    variable condition, whose maximum is 150).
  * GetOrganelle: the main-grid k-mer list 21,45,65,85,105 is restricted to
    k < read length (variable condition: k < 150, i.e. the full list).
  * Evaluation follows the main grid: internal methods use the fixed-origin
    evaluator; external tools use strand- and circular-origin normalization.
  * Per-job timeout is 3600 s for every external tool; a timeout is recorded
    as censoring (status "timeout"), never as F1 = 0.

Imports come from an unmodified git-archive export of
asem-refine-benchmarks v1.3.0 (BENCH_SCRIPTS below).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "panel30_grid"))
from resume_io import checkpoint, recover_csv  # noqa: E402
BENCH_SCRIPTS = os.environ.get(
    "BENCH_SCRIPTS", os.path.expanduser("~/plos-reruns/bench-v1.3.0/scripts")
)
sys.path.insert(0, BENCH_SCRIPTS)

import ext_getorganelle  # noqa: E402
import ext_novoplasty  # noqa: E402
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
from ext_mia import run_mia  # noqa: E402
from ext_pilon import run_pilon  # noqa: E402
from phase1_species import divergence_level, fasta_path  # noqa: E402

RESULTS_DIR = os.environ.get("CIRR_RESULTS_DIR", os.path.join(HERE, "results"))
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

PAIRS = [
    ("Saimiri_boliviensis", "Saimiri_sciureus"),  # same-genus
    ("Homo_sapiens", "Gorilla_gorilla"),          # same-family
    ("Homo_sapiens", "Varecia_variegata"),        # same-order
]
DEPTH = 4.0
LENGTH_CONDITIONS: list[object] = [50, 75, 100, 150, 250, 300, "variable"]
N_REPLICATES = 3
EXTERNAL_TIMEOUT_S = 3600
GETORGANELLE_KMERS = [21, 45, 65, 85, 105]
HYBRID_ALIGN_FN = functools.partial(_align_read_ieee_access, tau=0.5)


def _novoplasty(theta_init: str, reads: list[str], length: int, tag: str) -> str:
    template = ext_novoplasty._CONFIG_TEMPLATE
    ext_novoplasty._CONFIG_TEMPLATE = template.replace(
        "Read Length             = 150", f"Read Length             = {length}"
    )
    try:
        return ext_novoplasty.run_novoplasty(theta_init, reads, project=tag, timeout=EXTERNAL_TIMEOUT_S)
    finally:
        ext_novoplasty._CONFIG_TEMPLATE = template


def _getorganelle(theta_init: str, reads: list[str], length: int, seed: int) -> str:
    kmers = ",".join(str(k) for k in GETORGANELLE_KMERS if k < length)
    return ext_getorganelle.run_getorganelle(
        theta_init, reads, quality_seed=seed, timeout=EXTERNAL_TIMEOUT_S, kmers=kmers
    )


# name -> (callable(theta_init, reads, length, seed, tag) -> theta, external?)
METHODS = {
    "ASEM": (lambda s, r, n, seed, tag: run_asem_ieee_access(s, r).theta, False),
    "ASEM_recursive": (lambda s, r, n, seed, tag: run_asem_ojemb(s, r).theta, False),
    "ASEM_Hybrid": (lambda s, r, n, seed, tag: run_asem_hybrid_loop(
        s, r, align_read_fn=HYBRID_ALIGN_FN, boundary_flank=150, max_iterations=6, n_workers=1
    ).theta, False),
    "SRSC": (lambda s, r, n, seed, tag: run_ecticon(s, r).theta, False),
    "Pilon": (lambda s, r, n, seed, tag: run_pilon(s, r, timeout=EXTERNAL_TIMEOUT_S), True),
    "MIA": (lambda s, r, n, seed, tag: run_mia(s, r, timeout=EXTERNAL_TIMEOUT_S), True),
    "NOVOPlasty": (lambda s, r, n, seed, tag: _novoplasty(s, r, n, tag), True),
    "GetOrganelle": (lambda s, r, n, seed, tag: _getorganelle(s, r, n, seed), True),
}

FIELDNAMES = [
    "method", "reference", "target", "divergence", "length_condition", "replicate",
    "depth", "n_reads", "mean_read_len", "status", "error", "runtime_s", "theta_len",
    "identity_pct", "recall", "precision", "f1", "c_theta", "w_theta", "u_t", "host",
]

_SEQ_CACHE: dict[str, str] = {}


def _seq(name: str) -> str:
    if name not in _SEQ_CACHE:
        _SEQ_CACHE[name] = read_fasta(fasta_path(name))[1]
    return _SEQ_CACHE[name]


def _reads(target: str, length_condition, seed: int) -> list[str]:
    if length_condition == "variable":
        return simulate_shotgun_reads_variable_length(target, depth=DEPTH, seed=seed)
    return simulate_shotgun_reads(target, depth=DEPTH, read_length=length_condition, seed=seed)


def run_one(job: tuple[str, str, str, object, int]) -> dict:
    method, ref_name, target_name, length_condition, replicate = job
    reference, target = _seq(ref_name), _seq(target_name)
    seed = stable_seed("read_length_experiment", target_name, length_condition, replicate)
    reads = _reads(target, length_condition, seed)
    length = 150 if length_condition == "variable" else int(length_condition)
    fn, external = METHODS[method]
    row = {
        "method": method, "reference": ref_name, "target": target_name,
        "divergence": divergence_level(ref_name, target_name),
        "length_condition": length_condition, "replicate": replicate, "depth": DEPTH,
        "n_reads": len(reads), "mean_read_len": sum(map(len, reads)) / len(reads),
        "host": platform.node(), "error": "", "theta_len": 0,
    }
    tag = f"rl_{target_name}_{length_condition}_{replicate}"
    started = time.time()
    try:
        theta = fn(reference, reads, length, seed, tag)
    except Exception as exc:  # each wrapper raises its own failure type
        message = str(exc)
        lowered = message.lower()
        if "timed out" in lowered:
            status = "timeout"
        elif "no scaffold" in lowered or "no contig" in lowered or "empty" in lowered or "no final" in lowered:
            status = "no_output"
        else:
            status = "error"
        return {**row, "status": status, "error": message[:500], "runtime_s": time.time() - started}
    runtime = time.time() - started
    scores = evaluate_theta_vs_target(
        theta, target, normalize_strand=external, normalize_circular_origin=external
    )
    return {**row, "status": "completed", "runtime_s": runtime, "theta_len": len(theta), **scores}


def _done() -> set[tuple]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, newline="") as fh:
        return {(r["method"], r["reference"], r["target"], r["length_condition"], int(r["replicate"]))
                for r in csv.DictReader(fh)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    lock = open(os.path.join(RESULTS_DIR, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another runner holds the lock; exiting", flush=True)
        return

    recover_csv(RESULTS_CSV, FIELDNAMES)
    done = _done()
    jobs = [(m, ref, tgt, lc, rep) for m in methods for ref, tgt in PAIRS
            for lc in LENGTH_CONDITIONS for rep in range(N_REPLICATES)
            if (m, ref, tgt, str(lc), rep) not in done]
    with open(os.path.join(RESULTS_DIR, f"manifest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"), "w") as fh:
        json.dump({"bench_scripts": BENCH_SCRIPTS, "methods": methods, "n_jobs": len(jobs),
                   "parallel": args.parallel, "host": platform.node(), "python": sys.version,
                   "external_timeout_s": EXTERNAL_TIMEOUT_S, "getorganelle_kmers": GETORGANELLE_KMERS},
                  fh, indent=2)
    print(f"=== read-length all methods: {len(jobs)} jobs, {args.parallel} parallel ===", flush=True)

    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for n, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
                checkpoint(fh)
                print(f"[{n}/{len(jobs)}] {row['method']} {row['target']} len={row['length_condition']} "
                      f"rep={row['replicate']} {row['status']} f1={row.get('f1', '')} "
                      f"t={row['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
