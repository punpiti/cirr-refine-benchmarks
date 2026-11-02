"""Real-WGS validation benchmark: human real short reads (not simulated)
refined against a divergent chimpanzee mtDNA reference, evaluated against
the true human rCRS -- the one thing every other grid in this project
cannot show, since every other experiment uses simulated reads.

Data provenance (see extract_real_wgs_reads.sh): human reads are the
primary, mapped, non-duplicate 150-bp records from the mtDNA-mapped slice
of ENA PRJEB31736 sample NA07000 (`data/NA07000_chrM.bam`). The chimp
reference (NC_001643.1) was fetched fresh from NCBI for this run.

Design notes:
  - Reads are treated as single-end 150bp sequences, matching every other
    grid's read model. SAM/BAM stores SEQ for a mapped reverse-strand record
    already reverse-complemented into reference orientation (FLAG 0x10), so
    the sequence must not be reverse-complemented a second time.
  - Unmapped, secondary, supplementary, and duplicate-marked records are
    excluded with samtools `-F 3332` before sampling.
  - Depths 1-8X match the main Phase-1 grid exactly for direct
    comparability against the simulated-read baseline1/baseline2 numbers.
    Reads are subsampled without replacement from the 1.86M-record filtered pool,
    seeded the same way as the rest of the project (stable_seed) for
    reproducibility.
  - Both ASEM variants (IEEE Access: no recursion: OJEMB: with recursion)
    are run, matching baseline1/baseline2's design so this is a real vs.
    simulated comparison at matched (depth, replicate) cells, not a new
    method.
"""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from baseline_ieee_access import run_asem as run_asem_no_recursion  # noqa: E402
from baseline_ojemb import run_asem as run_asem_recursion  # noqa: E402
from common import evaluate_theta_vs_target, read_fasta, stable_seed  # noqa: E402

WGS_DATA_DIR = os.environ.get("GENOME_WGS_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
BAM_PATH = os.path.join(WGS_DATA_DIR, "NA07000_chrM.bam")  # see extract_real_wgs_reads.sh
CHIMP_FASTA = os.path.join(WGS_DATA_DIR, "refs", "chimp_NC_001643.fasta")  # see fetch_data.py
HUMAN_FASTA = os.path.join(WGS_DATA_DIR, "refs", "chrM_rCRS.fasta")  # see fetch_data.py
SAMTOOLS = os.environ.get("SAMTOOLS_BIN", "samtools")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "real_wgs_benchmark")
GRID_RESULTS_CSV = os.path.join(RESULTS_DIR, "grid_results.csv")

DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
N_REPLICATES = 3
MAX_ITERATIONS = 6
READ_LENGTH = 150

def _extract_reads() -> list[str]:
    """Extract primary, mapped, non-duplicate 150-bp BAM records.

    SEQ is already represented in reference orientation for mapped records,
    including those with FLAG 0x10; applying another reverse complement here
    would be a strand-orientation error.
    """
    proc = subprocess.run(
        [SAMTOOLS, "view", "-F", "3332", BAM_PATH],
        capture_output=True, text=True, check=True,
    )
    reads = []
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        seq = fields[9]
        if len(seq) != READ_LENGTH:
            continue
        reads.append(seq)
    return reads


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading references...", flush=True)
    _, chimp_seq = read_fasta(CHIMP_FASTA)
    _, human_seq = read_fasta(HUMAN_FASTA)
    print(f"  chimp (reference): {len(chimp_seq)}bp, human (truth target): {len(human_seq)}bp", flush=True)

    print("Extracting real reads from BAM...", flush=True)
    t0 = time.time()
    all_reads = _extract_reads()
    print(f"  {len(all_reads)} usable 150bp reads ({time.time()-t0:.0f}s)", flush=True)

    genome_len = len(human_seq)

    initial_scores = evaluate_theta_vs_target(chimp_seq, human_seq)
    print(f"  initial state (chimp vs human, no refinement): f1={initial_scores['f1']:.4f} "
          f"identity={initial_scores['identity_pct']:.2f}%", flush=True)
    initial_row = {
        "reference": "Pan_troglodytes_NC_001643", "target": "Homo_sapiens_NA07000_real",
        "divergence": "same-family", "depth": 0, "replicate": 0, "iteration": 0,
        "method": "initial_state", "n_reads": 0, **initial_scores,
    }

    rows = [initial_row]
    jobs = [
        (method_name, run_fn, depth, rep)
        for method_name, run_fn in [
            ("ASEM_no_recursion", run_asem_no_recursion),
            ("ASEM_recursion", run_asem_recursion),
        ]
        for depth in DEPTHS
        for rep in range(N_REPLICATES)
    ]
    total = len(jobs)
    t0 = time.time()
    for done, (method_name, run_fn, depth, rep) in enumerate(jobs, start=1):
        seed = stable_seed("NA07000_real", depth, rep)
        n_needed = min(round(depth * genome_len / READ_LENGTH), len(all_reads))
        reads = random.Random(seed).sample(all_reads, n_needed)

        result = run_fn(theta_init=chimp_seq, reads=reads, max_iterations=MAX_ITERATIONS, n_workers=1)

        for stats, theta_snapshot in zip(result.history, result.theta_by_iteration):
            scores = evaluate_theta_vs_target(theta_snapshot, human_seq)
            rows.append({
                "reference": "Pan_troglodytes_NC_001643", "target": "Homo_sapiens_NA07000_real",
                "divergence": "same-family", "depth": depth, "replicate": rep,
                "iteration": stats.iteration, "method": method_name, "n_reads": len(reads),
                **scores,
            })
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else float("nan")
        last_f1 = rows[-1]["f1"]
        print(f"  [{done}/{total} {100*done/total:5.1f}%] {method_name} depth={depth}X rep={rep} "
              f"final_f1={last_f1:.4f} | {elapsed:6.0f}s elapsed, ~{eta:6.0f}s remaining", flush=True)

    _write_csv(GRID_RESULTS_CSV, rows)
    print(f"\nwrote {len(rows)} rows to {GRID_RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
