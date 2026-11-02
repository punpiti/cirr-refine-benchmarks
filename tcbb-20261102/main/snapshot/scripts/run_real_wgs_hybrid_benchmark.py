"""Real-read validation of ordinary ASEM and ASEM-Hybrid.

The input is the mtDNA-mapped subset of ENA PRJEB31736 sample NA07000.
Two starting references test complementary questions on byte-identical read
subsets: chimpanzee mtDNA (same-family; empirical-noise safety check) and
Varecia variegata mtDNA (same-order; the divergence regime in which Hybrid
improved simulated-read results).  Human rCRS is the evaluation target.

The script writes one final row per method/reference/depth/replicate job.  It
does not alter the earlier real-WGS CSV, which contains only the two original
ASEM variants and remains the provenance for the corresponding manuscript
results.  Set ``ASEM_HYBRID_VARIANT=boundary`` to evaluate the experimental
boundary-recruited ASEM-Hybrid separately, in ``real_wgs_hybrid_boundary/``.
"""

from __future__ import annotations

import csv
import functools
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(__file__))

from asem_hybrid import run_asem_hybrid_loop  # noqa: E402
from baseline_ieee_access import _align_read_ieee_access, run_asem  # noqa: E402
from baseline_ojemb import run_asem as run_asem_recursive  # noqa: E402
from common import evaluate_theta_vs_target, read_fasta, stable_seed  # noqa: E402

WGS_DATA_DIR = os.environ.get("GENOME_WGS_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
BAM_PATH = os.path.join(WGS_DATA_DIR, "NA07000_chrM.bam")  # see extract_real_wgs_reads.sh
HUMAN_FASTA = os.path.join(WGS_DATA_DIR, "refs", "chrM_rCRS.fasta")  # see fetch_data.py
CHIMP_FASTA = os.path.join(WGS_DATA_DIR, "refs", "chimp_NC_001643.fasta")  # see fetch_data.py
VARECIA_FASTA = os.path.join(
    os.path.dirname(__file__), "..", "data", "phase1_reproduction",
    "Varecia_variegata_NC_012773.fasta",
)
SAMTOOLS = os.environ.get("SAMTOOLS_BIN", "samtools")

VARIANT = os.environ.get("ASEM_HYBRID_VARIANT", "legacy").strip().lower()
if VARIANT not in {"legacy", "boundary"}:
    raise ValueError("ASEM_HYBRID_VARIANT must be 'legacy' or 'boundary'")

RESULTS_NAME = "real_wgs_hybrid" if VARIANT == "legacy" else "real_wgs_hybrid_boundary"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", RESULTS_NAME)
RESULTS_CSV = os.path.join(RESULTS_DIR, "final_results.csv")
DEPTHS = range(1, 9)
N_REPLICATES = 3
MAX_ITERATIONS = 6
READ_LENGTH = 150
N_PARALLEL_RUNS = int(os.environ.get("ASEM_REAL_WGS_WORKERS", "2"))
TAU = 0.5

def _extract_reads() -> list[str]:
    proc = subprocess.run(
        [SAMTOOLS, "view", "-F", "3332", BAM_PATH],
        capture_output=True, text=True, check=True,
    )
    reads: list[str] = []
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        seq = fields[9]
        if len(seq) != READ_LENGTH:
            continue
        # SAM/BAM already stores mapped reverse-strand SEQ in reference
        # orientation when FLAG 0x10 is set; do not reverse-complement again.
        reads.append(seq)
    return reads


def _run_one(args: tuple[str, str, str, int, int, list[str], str]) -> dict:
    reference_name, reference_seq, method, depth, replicate, reads, target_seq = args
    started = time.perf_counter()
    if method == "ASEM_no_recursion":
        result = run_asem(
            theta_init=reference_seq,
            reads=reads,
            max_iterations=MAX_ITERATIONS,
            n_workers=1,
        )
        n_contigs = n_anchors = 0
    elif method == "ASEM_recursion":
        result = run_asem_recursive(
            theta_init=reference_seq,
            reads=reads,
            max_iterations=MAX_ITERATIONS,
            n_workers=1,
        )
        n_contigs = n_anchors = 0
    else:
        align_fn = functools.partial(_align_read_ieee_access, tau=TAU)
        result = run_asem_hybrid_loop(
            theta_init=reference_seq,
            reads=reads,
            align_read_fn=align_fn,
            boundary_flank=None if VARIANT == "legacy" else 150,
            max_iterations=MAX_ITERATIONS,
            n_workers=1,
        )
        n_contigs = sum(stats.n_contigs for stats in result.history)
        n_anchors = sum(stats.n_contigs_anchored for stats in result.history)

    scores = evaluate_theta_vs_target(result.theta, target_seq)
    return {
        "reference": reference_name,
        "target": "Homo_sapiens_NA07000_real",
        "divergence": "same-family" if reference_name.startswith("Pan_") else "same-order",
        "method": "ASEM_Hybrid_boundary" if method == "ASEM_Hybrid" and VARIANT == "boundary" else method,
        "depth": depth,
        "replicate": replicate,
        "n_reads": len(reads),
        "last_iteration": result.history[-1].iteration,
        "hit_iteration_cap": int(result.history[-1].iteration == MAX_ITERATIONS),
        "n_contigs_total": n_contigs,
        "n_anchors_total": n_anchors,
        "wall_s": time.perf_counter() - started,
        **scores,
    }


def _write_csv(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(f"=== ASEM-Hybrid variant: {VARIANT}; results: {RESULTS_DIR} ===", flush=True)
    _, human_seq = read_fasta(HUMAN_FASTA)
    references = {}
    for name, path in (
        ("Pan_troglodytes_NC_001643", CHIMP_FASTA),
        ("Varecia_variegata_NC_012773", VARECIA_FASTA),
    ):
        _, references[name] = read_fasta(path)

    print("Extracting 150-bp mtDNA-mapped reads from NA07000...", flush=True)
    all_reads = _extract_reads()
    print(f"  retained {len(all_reads):,} reads", flush=True)

    jobs = []
    for reference_name, reference_seq in references.items():
        for depth in DEPTHS:
            n_needed = min(round(depth * len(human_seq) / READ_LENGTH), len(all_reads))
            for replicate in range(N_REPLICATES):
                seed = stable_seed("NA07000_real", depth, replicate)
                reads = random.Random(seed).sample(all_reads, n_needed)
                methods = ["ASEM_no_recursion", "ASEM_Hybrid"]
                if reference_name.startswith("Pan_"):
                    methods.insert(1, "ASEM_recursion")
                for method in methods:
                    jobs.append(
                        (reference_name, reference_seq, method, depth, replicate, reads, human_seq)
                    )

    rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=N_PARALLEL_RUNS) as pool:
        futures = {pool.submit(_run_one, job): job[:5] for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            elapsed = time.time() - started
            rate = done / elapsed
            eta = (len(jobs) - done) / rate if rate else 0
            print(
                f"[{done:02d}/{len(jobs)}] {row['reference']} {row['method']} "
                f"{row['depth']}X rep={row['replicate']} F1={row['f1']:.4f} "
                f"anchors={row['n_anchors_total']} ETA={eta:.0f}s",
                flush=True,
            )

    rows.sort(key=lambda r: (r["reference"], r["method"], r["depth"], r["replicate"]))
    _write_csv(rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
