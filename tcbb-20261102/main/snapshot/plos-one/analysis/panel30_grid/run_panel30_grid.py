"""Single simulation grid for the PLOS ONE manuscript: every method on the
30-species panel (870 ordered pairs x depths 1-8X x 3 replicates).

This replaces the separate 6-species main grid and the Macaca same-genus
experiment, both of which are subsets of this panel (the 6-species grid minus
the doubtful Saimiri sciureus record NC_012775).

Design, fixed before any result was seen:
  * Reads: simulate_shotgun_reads(target, depth, 150, stable_seed(target_name,
    depth, replicate)), exactly as in the 6-species grid, so every job whose
    target was in that grid receives byte-identical reads.
  * Methods: CIRR (benchmark key ASEM), CIRR-Hybrid (ASEM_Hybrid, boundary
    variant), SRSC, Pilon, MIA, NOVOPlasty, GetOrganelle, all from one code
    snapshot (BENCH_SCRIPTS) with the prior-plus-hashed tie rule.
  * External tools: fixed main-grid settings, one 3600-s limit per job; a
    timeout or a failure is recorded as an outcome, never as F1 = 0.
  * Scoring: N-neutral evaluator (evaluator_n_fix/nsafe.py); external tools
    are strand- and circular-origin normalized, internal methods are not.
  * Every output sequence is stored (gzip) with its SHA-256.

Results and sequences go to PANEL30_RESULTS_DIR (default ~/plos-reruns/panel30),
outside the synced project tree; copy results.csv back when the run is final.

    BENCH_SCRIPTS=... python -I run_panel30_grid.py --parallel 16
    python -I run_panel30_grid.py --methods ASEM,ASEM_Hybrid --limit 20   # smoke test
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import functools  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from importlib.metadata import version  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_SCRIPTS = os.environ.setdefault(
    "BENCH_SCRIPTS", os.path.join(HERE, "..", "..", "..", "asem-refine-benchmarks", "scripts")
)
sys.path.insert(0, BENCH_SCRIPTS)
sys.path.insert(0, os.path.join(HERE, "..", "evaluator_n_fix"))
sys.path.insert(0, HERE)  # python -I does not add the script directory

import nsafe  # noqa: E402
import common  # noqa: E402
import ext_getorganelle  # noqa: E402
import ext_novoplasty  # noqa: E402
from asem_hybrid import run_asem_hybrid_loop  # noqa: E402
from baseline_ecticon import run_ecticon  # noqa: E402
from baseline_ieee_access import _align_read_ieee_access, run_asem  # noqa: E402
from common import read_fasta, simulate_shotgun_reads, stable_seed  # noqa: E402
from ext_mia import run_mia  # noqa: E402
from ext_pilon import run_pilon  # noqa: E402
from panel30_species import SPECIES, divergence_level, fasta_path, ordered_pairs  # noqa: E402
from resume_io import checkpoint, recover_csv, sync_directory  # noqa: E402

nsafe.install()

RESULTS_DIR = os.environ.get("PANEL30_RESULTS_DIR", os.path.expanduser("~/plos-reruns/panel30"))
DEPTHS = list(range(1, 9))
N_REPLICATES = 3
TIMEOUT_S = 3600
HYBRID_ALIGN_FN = functools.partial(_align_read_ieee_access, tau=0.5)

# name -> (callable(reference, reads, seed, tag) -> theta, external?)
METHODS = {
    "ASEM": (lambda s, r, seed, tag: run_asem(s, r).theta, False),
    "ASEM_Hybrid": (lambda s, r, seed, tag: run_asem_hybrid_loop(
        s, r, align_read_fn=HYBRID_ALIGN_FN, boundary_flank=150, max_iterations=6, n_workers=1
    ).theta, False),
    "SRSC": (lambda s, r, seed, tag: run_ecticon(s, r).theta, False),
    "Pilon": (lambda s, r, seed, tag: run_pilon(s, r, timeout=TIMEOUT_S), True),
    "MIA": (lambda s, r, seed, tag: run_mia(s, r, timeout=TIMEOUT_S), True),
    "NOVOPlasty": (lambda s, r, seed, tag: ext_novoplasty.run_novoplasty(
        s, r, project=tag, timeout=TIMEOUT_S), True),
    "GetOrganelle": (lambda s, r, seed, tag: ext_getorganelle.run_getorganelle(
        s, r, quality_seed=seed, timeout=TIMEOUT_S), True),
}

FIELDS = ["method", "reference", "target", "divergence", "depth", "replicate", "n_reads", "status",
          "error", "runtime_s", "theta_len", "n_ambiguous", "f1", "identity_pct", "recall", "precision",
          "c_theta", "w_theta", "u_t", "theta_fasta", "theta_sha256", "host"]

_SEQ: dict[str, str] = {}


def _seq(name: str) -> str:
    if name not in _SEQ:
        _SEQ[name] = read_fasta(fasta_path(name))[1]
    return _SEQ[name]


def _save_theta(results_dir: str, method: str, slug: str, theta: str) -> tuple[str, str]:
    path = os.path.join(results_dir, "thetas", method, f"{slug}.fa.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + f".tmp-{os.getpid()}"
    with gzip.GzipFile(temporary, "wb", mtime=0) as fh:
        fh.write(f">{method} {slug}\n{theta}\n".encode())
    with open(temporary, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(temporary, path)
    sync_directory(os.path.dirname(path))
    return os.path.relpath(path, results_dir), hashlib.sha256(theta.encode()).hexdigest()


def run_job(job: tuple) -> dict:
    results_dir, method, ref_name, target_name, depth, replicate = job
    target = _seq(target_name)
    seed = stable_seed(target_name, depth, replicate)
    reads = simulate_shotgun_reads(target, float(depth), 150, seed)
    slug = f"{ref_name}__{target_name}__{depth}X__r{replicate}"
    row = {"method": method, "reference": ref_name, "target": target_name,
           "divergence": divergence_level(ref_name, target_name), "depth": depth,
           "replicate": replicate, "n_reads": len(reads), "host": platform.node()}
    fn, external = METHODS[method]
    started = time.time()
    try:
        theta = fn(_seq(ref_name), reads, seed, f"p30_{slug}")
    except Exception as exc:  # each wrapper raises its own failure type
        message = str(exc)
        log = Path(results_dir) / "errors" / method / f"{slug}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(message)
        lowered = message.lower()
        if "timed out" in lowered:
            status = "timeout"
        elif any(k in lowered for k in ("no scaffold", "no contig", "empty", "no final")):
            status = "no_output"
        else:
            status = "error"
        return {**row, "status": status, "error": message[:300], "runtime_s": round(time.time() - started, 2)}
    runtime = time.time() - started
    if not theta:
        return {**row, "status": "no_output", "runtime_s": round(runtime, 2)}
    path, sha = _save_theta(results_dir, method, slug, theta)
    scores = common.evaluate_theta_vs_target(
        theta, target, normalize_strand=external, normalize_circular_origin=external
    )
    return {**row, "status": "completed", "runtime_s": round(runtime, 2), "theta_len": len(theta),
            "n_ambiguous": sum(c not in "ACGT" for c in theta), "theta_fasta": path, "theta_sha256": sha,
            **{k: scores[k] for k in ("f1", "identity_pct", "recall", "precision", "c_theta", "w_theta", "u_t")}}


def _done(results_csv: str) -> set[tuple]:
    if not os.path.exists(results_csv):
        return set()
    with open(results_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    keys = [(r["method"], r["reference"], r["target"], int(r["depth"]), int(r["replicate"]))
            for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result keys; inspect results before resuming")
    if any(r["status"] not in {"completed", "timeout", "no_output", "error"} for r in rows):
        raise ValueError("incomplete or invalid result row; inspect results before resuming")
    return set(keys)


def execution_contract() -> dict:
    """Pin actual imported source and inputs, including archive exports without .git."""
    paths = list(Path(BENCH_SCRIPTS).glob("*.py"))
    paths += [Path(__file__), Path(HERE) / "panel30_species.py", Path(HERE) / "resume_io.py", Path(nsafe.__file__)]
    files = {("bench/" + p.name if p.parent == Path(BENCH_SCRIPTS) else "analysis/" + p.name):
             hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data = {name: hashlib.sha256(Path(fasta_path(name)).read_bytes()).hexdigest() for name in SPECIES}
    return {"sources_sha256": files, "inputs_sha256": data, "taxonomy": "original-panel",
            "depths": DEPTHS, "replicates": N_REPLICATES, "timeout_s": TIMEOUT_S,
            "read_length": 150, "python": sys.version,
            "packages": {name: version(name) for name in ("numpy", "scikit-bio", "parasail")},
            "thread_limits": {v: os.environ[v] for v in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def _git_head(path: str) -> str:
    if os.environ.get("BENCH_COMMIT"):
        return os.environ["BENCH_COMMIT"]
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N jobs, results under smoke/")
    args = parser.parse_args()
    if args.parallel < 1 or args.limit < 0:
        parser.error("parallel must be positive and limit must be nonnegative")
    methods = [m.strip() for m in args.methods.split(",")]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")

    results_dir = os.path.join(RESULTS_DIR, "smoke") if args.limit else RESULTS_DIR
    results_csv = os.path.join(results_dir, "results.csv")
    os.makedirs(results_dir, exist_ok=True)
    lock = open(os.path.join(results_dir, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another runner holds the lock; exiting", flush=True)
        return

    recover_csv(results_csv, FIELDS)
    done = _done(results_csv)
    contract = execution_contract()
    contract_path = Path(results_dir) / "execution_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise SystemExit("source/input/environment contract changed; use a new results directory")
    elif done:
        raise SystemExit("existing results have no execution contract; use a new results directory")
    else:
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True))
    # Depth-major order spreads the slow high-depth jobs over the whole run.
    jobs = [(results_dir, m, ref, tgt, d, rep) for d in DEPTHS for rep in range(N_REPLICATES)
            for ref, tgt in ordered_pairs() for m in methods
            if (m, ref, tgt, d, rep) not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(results_dir, f"manifest_{stamp}.json"), "w") as fh:
        json.dump({"bench_scripts": os.path.abspath(BENCH_SCRIPTS), "bench_commit": _git_head(BENCH_SCRIPTS),
                   "methods": methods, "n_jobs": len(jobs), "n_done_before": len(done),
                   "host": platform.node(), "python": sys.version, "started_utc": stamp}, fh, indent=2)
    print(f"{len(jobs)} jobs to run ({len(done)} already done), {args.parallel} workers", flush=True)

    new_file = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                writer.writerow(future.result())
                checkpoint(out)
                if i % 200 == 0 or i == len(jobs):
                    elapsed = time.time() - started
                    print(f"  {i}/{len(jobs)} ({elapsed / 3600:.1f} h elapsed, "
                          f"~{elapsed / i * (len(jobs) - i) / 3600:.1f} h remaining)", flush=True)


if __name__ == "__main__":
    main()
