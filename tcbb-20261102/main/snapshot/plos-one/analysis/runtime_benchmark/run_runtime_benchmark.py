"""Controlled runtime/peak-memory benchmark for every method.

Historical grid wall times came from overlapping parallel batches and are not
comparable across methods (the MIA follow-up showed jobs that hit 300 s in the
grid finishing in 52-130 s on an idle machine). This benchmark runs ONE job at
a time, each in a fresh process under /usr/bin/time, on identical inputs:

  * methods: ASEM, ASEM_Hybrid, SRSC, Pilon, MIA, NOVOPlasty, GetOrganelle
    (principal set) plus ASEM_recursive (ablation);
  * the three representative pairs (one per divergence level);
  * depths 1, 2, 4, 8X; replicates 0-2 with the main-grid read seed
    stable_seed(target, depth, replicate) and 150-bp reads;
  * single thread per tool (OMP/BLAS=1, bwa -t 1, GetOrganelle -t 1);
  * included boundary: method call on in-memory reads -> returned sequence,
    including each wrapper's own file I/O and indexing; excluding read
    simulation and evaluation;
  * per-job limit 3600 s; a limit hit is reported as right-censored.

Memory: method_peak_rss_kb = max(RUSAGE_SELF, RUSAGE_CHILDREN) ru_maxrss,
taken in the worker after the method returns and BEFORE evaluation (Python
process incl. imports, and the largest external-tool subprocess).
peak_rss_kb (/usr/bin/time %M) also covers the quadratic-memory evaluation
alignment and must not be reported as method memory.

Run only on an otherwise idle machine: the runner refuses to start when the
1-minute load average exceeds --max-load.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "panel30_grid"))
from resume_io import checkpoint, recover_csv  # noqa: E402
RESULTS_DIR = os.environ.get("CIRR_RESULTS_DIR", os.path.join(HERE, "results"))
RESULTS_CSV = os.path.join(RESULTS_DIR, "runtime_results.csv")
WORKER = os.path.join(HERE, "runtime_worker.py")
PYTHON = sys.executable

METHODS = ["ASEM", "ASEM_recursive", "ASEM_Hybrid", "SRSC", "Pilon", "MIA", "NOVOPlasty", "GetOrganelle"]
PAIRS = [
    ("Saimiri_boliviensis", "Saimiri_sciureus"),
    ("Homo_sapiens", "Gorilla_gorilla"),
    ("Homo_sapiens", "Varecia_variegata"),
]
DEPTHS = [1, 2, 4, 8]
N_REPLICATES = 3
TIMEOUT_S = 3600

FIELDNAMES = [
    "method", "reference", "target", "depth", "replicate", "status", "returncode",
    "wall_s", "worker_method_s", "method_peak_rss_kb", "method_rss_self_kb", "method_rss_child_kb", "peak_rss_kb", "theta_len", "f1", "identity_pct",
    "error", "load1_before", "started_utc", "host",
]


def _done() -> set[tuple]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, newline="") as fh:
        return {(r["method"], r["reference"], r["target"], int(r["depth"]), int(r["replicate"]))
                for r in csv.DictReader(fh)}


def run_job(method: str, ref: str, target: str, depth: int, rep: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="rtbench_") as tmp:
        time_path = os.path.join(tmp, "time.txt")
        out_path = os.path.join(tmp, "out.json")
        cmd = ["/usr/bin/time", "-f", "%e %M", "-o", time_path,
               PYTHON, "-I", WORKER, method, ref, target, str(depth), str(rep), out_path]
        row = {"method": method, "reference": ref, "target": target, "depth": depth,
               "replicate": rep, "load1_before": os.getloadavg()[0],
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "host": platform.node(), "error": ""}
        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
                                  env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        except subprocess.TimeoutExpired:
            return {**row, "status": "timeout", "returncode": "", "wall_s": TIMEOUT_S,
                    "error": f"right-censored at {TIMEOUT_S}s"}
        row["wall_s"] = round(time.time() - started, 3)
        row["returncode"] = proc.returncode
        if os.path.exists(time_path):
            last = open(time_path).read().strip().splitlines()[-1].split()
            if len(last) == 2:
                row["peak_rss_kb"] = int(last[1])
        if os.path.exists(out_path):
            row.update(json.load(open(out_path)))
        else:
            row.update(status="error", error=proc.stderr[-400:])
        return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-load", type=float, default=1.5)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    load = os.getloadavg()[0]
    if load > args.max_load:
        sys.exit(f"1-min load {load:.2f} > {args.max_load}; machine not idle, refusing to start")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    recover_csv(RESULTS_CSV, FIELDNAMES)
    done = _done()
    methods = args.methods.split(",")
    # Interleave methods inside each input so slow drifts in machine state do
    # not line up with one method.
    jobs = [(m, ref, tgt, d, rep) for ref, tgt in PAIRS for d in DEPTHS for rep in range(N_REPLICATES)
            for m in methods if (m, ref, tgt, d, rep) not in done]
    with open(os.path.join(RESULTS_DIR, f"manifest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"), "w") as fh:
        json.dump({"host": platform.node(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                   "python": sys.version, "n_jobs": len(jobs), "timeout_s": TIMEOUT_S,
                   "methods": methods, "pairs": PAIRS, "depths": DEPTHS, "replicates": N_REPLICATES,
                   "bench_scripts": os.environ.get("BENCH_SCRIPTS",
                                                   os.path.expanduser("~/plos-reruns/bench-v1.3.0/scripts"))},
                  fh, indent=2)
    print(f"=== runtime benchmark: {len(jobs)} jobs, sequential ===", flush=True)
    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        for n, job in enumerate(jobs, start=1):
            row = run_job(*job)
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
            checkpoint(fh)
            print(f"[{n}/{len(jobs)}] {job} {row.get('status')} wall={row.get('wall_s')} "
                  f"rss={row.get('method_peak_rss_kb')} f1={row.get('f1', '')}", flush=True)


if __name__ == "__main__":
    main()
