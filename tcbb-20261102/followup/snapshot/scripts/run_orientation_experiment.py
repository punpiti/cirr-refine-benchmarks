"""Mixed-orientation read test for ASEM and ASEM-Hybrid.

The simulated grid uses forward-strand reads only. Raw FASTQ mixes strands.
This experiment takes the exact grid reads for the hardest same-order pair
(Homo_sapiens reference -> Varecia_variegata target; lowest mean ordinary-ASEM
F1 in the Phase-1 grid) and compares:

  forward : the grid reads unchanged
  mixed   : the same reads, each reverse-complemented with probability 0.5
            (deterministic per job; target truth is never used to re-orient)

Both conditions are run with the released `asem-refine` package (v0.4.1),
whose local aligner tries both strands. Its Hybrid read--read overlap
assembly does not reverse-complement reads, which is the limitation this
test measures. Settings match the paper: tau=0.5, w=0.1, six-iteration cap,
no recursive re-alignment; Hybrid contig_tau=0.25, min_anchor_len=300,
min_overlap=20, boundary_flank=150.

Writes results/orientation_experiment/results.csv.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(__file__))

from asem_refine.align import reverse_complement  # noqa: E402
from asem_refine.estep import build_align_read_fn, run_asem  # noqa: E402
from asem_refine.hybrid import run_asem_hybrid_loop  # noqa: E402

from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed  # noqa: E402
from phase1_species import fasta_path  # noqa: E402

REFERENCE, TARGET = "Homo_sapiens", "Varecia_variegata"
DEPTHS = [1, 2, 4, 8]
N_REPLICATES = 3
CONDITIONS = ["forward", "mixed"]
METHODS = ["asem", "hybrid"]
WORKERS = int(os.environ.get("ORIENTATION_WORKERS", "4"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "orientation_experiment")
OUT_CSV = os.path.join(OUT_DIR, "results.csv")


def job_reads(depth: int, replicate: int, condition: str) -> tuple[list[str], int]:
    _, t = read_fasta(fasta_path(TARGET))
    reads = simulate_shotgun_reads(t, depth=float(depth), read_length=150,
                                   seed=stable_seed(TARGET, depth, replicate))
    if condition == "forward":
        return reads, 0
    rng = random.Random(stable_seed(TARGET, depth, replicate, "orientation"))
    flipped = [rng.random() < 0.5 for _ in reads]
    return [reverse_complement(r) if f else r for r, f in zip(reads, flipped)], sum(flipped)


def run_one(job: tuple[str, str, int, int]) -> dict:
    method, condition, depth, replicate = job
    _, s = read_fasta(fasta_path(REFERENCE))
    _, t = read_fasta(fasta_path(TARGET))
    reads, n_flipped = job_reads(depth, replicate, condition)
    if method == "asem":
        result = run_asem(theta_init=s, reads=reads, tau=0.5, w=0.1, max_iterations=6,
                          recursive=False, try_reverse_complement=True)
    else:
        align = build_align_read_fn(0.5, False, 30, True)
        result = run_asem_hybrid_loop(theta_init=s, reads=reads, align_read_fn=align,
                                      contig_tau=0.25, min_anchor_len=300, boundary_flank=150,
                                      w=0.1, max_iterations=6, min_overlap=20)
    last = asdict(result.history[-1]) if result.history else {}
    scores = evaluate_theta_vs_target(result.theta, t)
    return {
        "method": method, "condition": condition, "reference": REFERENCE, "target": TARGET,
        "depth": depth, "replicate": replicate, "n_reads": len(reads), "n_flipped": n_flipped,
        "iterations": len(result.history), "theta_len": len(result.theta),
        **{f"last_{k}": v for k, v in last.items() if isinstance(v, (int, float))},
        **scores,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(m, c, d, r) for m in METHODS for c in CONDITIONS for d in DEPTHS for r in range(N_REPLICATES)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"{row['method']:6s} {row['condition']:7s} d={row['depth']} r={row['replicate']} "
                  f"F1={row['f1']:.4f}", flush=True)
    rows.sort(key=lambda r: (r["method"], r["condition"], r["depth"], r["replicate"]))
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    fields = list(rows[0].keys()) + [k for k in fields if k not in rows[0]]
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
