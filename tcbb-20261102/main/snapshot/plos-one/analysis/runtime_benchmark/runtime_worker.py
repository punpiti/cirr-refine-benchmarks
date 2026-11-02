"""One timed benchmark job in a fresh process (called by run_runtime_benchmark.py).

argv: method reference target depth replicate out_json
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

BENCH_SCRIPTS = os.environ.get("BENCH_SCRIPTS", os.path.expanduser("~/plos-reruns/bench-v1.3.0/scripts"))
sys.path.insert(0, BENCH_SCRIPTS)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "read_length_all_methods"))

from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed  # noqa: E402
from phase1_species import fasta_path  # noqa: E402
from run_read_length_all_methods import METHODS  # noqa: E402


def main() -> None:
    method, ref_name, target_name, depth, rep, out_path = sys.argv[1:]
    depth, rep = int(depth), int(rep)
    reference = read_fasta(fasta_path(ref_name))[1]
    target = read_fasta(fasta_path(target_name))[1]
    seed = stable_seed(target_name, depth, rep)
    reads = simulate_shotgun_reads(target, float(depth), 150, seed)
    fn, external = METHODS[method]
    started = time.perf_counter()
    try:
        theta = fn(reference, reads, 150, seed, f"rt_{target_name}_{depth}_{rep}")
    except Exception as exc:
        elapsed = time.perf_counter() - started
        message = str(exc)
        status = "timeout" if "timed out" in message.lower() else "no_output_or_error"
        json.dump({"status": status, "worker_method_s": round(elapsed, 3), "error": message[:400]},
                  open(out_path, "w"))
        return
    elapsed = time.perf_counter() - started
    # Peak memory of the method itself, captured before the quadratic-memory
    # evaluation alignment: this process (imports + method) and the largest
    # external tool subprocess it waited for. ru_maxrss is in KB on Linux.
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scores = evaluate_theta_vs_target(theta, target, normalize_strand=external,
                                      normalize_circular_origin=external)
    json.dump({"status": "completed", "worker_method_s": round(elapsed, 3), "theta_len": len(theta),
               "method_rss_self_kb": self_kb, "method_rss_child_kb": child_kb,
               "method_peak_rss_kb": max(self_kb, child_kb),
               "f1": scores["f1"], "identity_pct": scores["identity_pct"]}, open(out_path, "w"))


if __name__ == "__main__":
    main()
