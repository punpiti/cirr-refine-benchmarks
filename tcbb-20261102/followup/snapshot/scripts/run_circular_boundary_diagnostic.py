"""Measure whether linear origin handling reduces terminal consensus accuracy.

Synthetic references contain uniformly distributed substitutions relative to a
human mtDNA target, avoiding confounding by the biologically hypervariable
control region. Read sets are either the study's linear sampler or a circular
sampler that permits origin-spanning reads. ASEM itself remains linear.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from skbio.alignment import pair_align_nucl
from skbio.sequence import DNA

sys.path.insert(0, os.path.dirname(__file__))

from baseline_ieee_access import run_asem
from common import read_fasta, simulate_shotgun_reads, stable_seed
from phase1_species import fasta_path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "circular_boundary")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
DEPTHS = (1, 2, 4, 8)
DIVERGENCES = (0.05, 0.10, 0.20)
N_REPLICATES = 10
READ_LENGTH = 150
N_PARALLEL_RUNS = int(os.environ.get("CIRCULAR_PARALLEL_RUNS", "6"))


def mutate_substitutions(sequence: str, fraction: float, seed: int) -> str:
    rng = random.Random(seed)
    bases = "ACGT"
    output = list(sequence)
    for position in rng.sample(range(len(sequence)), round(fraction * len(sequence))):
        output[position] = rng.choice(bases.replace(output[position], ""))
    return "".join(output)


def simulate_circular_reads(sequence: str, depth: float, seed: int) -> tuple[list[str], int]:
    rng = random.Random(seed)
    n_reads = round(depth * len(sequence) / READ_LENGTH)
    reads = []
    spanning = 0
    for _ in range(n_reads):
        start = rng.randrange(len(sequence))
        end = start + READ_LENGTH
        if end <= len(sequence):
            reads.append(sequence[start:end])
        else:
            spanning += 1
            reads.append(sequence[start:] + sequence[: end - len(sequence)])
    return reads, spanning


def regional_accuracy(theta: str, target: str, window: int = READ_LENGTH) -> dict[str, float]:
    result = pair_align_nucl(DNA(theta), DNA(target), mode="global")
    aligned_theta, aligned_target = result.paths[0].to_aligned([DNA(theta), DNA(target)])
    target_position = -1
    counts = {"terminal_correct": 0, "terminal_total": 0, "interior_correct": 0, "interior_total": 0}
    for estimate_base, target_base in zip(str(aligned_theta), str(aligned_target)):
        if target_base == "-":
            continue
        target_position += 1
        region = "terminal" if target_position < window or target_position >= len(target) - window else "interior"
        counts[f"{region}_total"] += 1
        counts[f"{region}_correct"] += int(estimate_base == target_base)
    return {
        **counts,
        "terminal_accuracy": counts["terminal_correct"] / counts["terminal_total"],
        "interior_accuracy": counts["interior_correct"] / counts["interior_total"],
    }


def _run_one(job: tuple[float, int, int, str]) -> dict:
    divergence, depth, replicate, sampling = job
    target = read_fasta(fasta_path("Homo_sapiens"))[1]
    reference = mutate_substitutions(target, divergence, stable_seed("reference", divergence))
    seed = stable_seed("circular-boundary", divergence, depth, replicate)
    if sampling == "circular":
        reads, n_spanning = simulate_circular_reads(target, depth, seed)
    else:
        reads = simulate_shotgun_reads(target, depth, READ_LENGTH, seed)
        n_spanning = 0
    result = run_asem(reference, reads, tau=0.5, w=0.1, max_iterations=6, n_workers=1)
    return {
        "divergence": divergence,
        "depth": depth,
        "replicate": replicate,
        "sampling": sampling,
        "n_reads": len(reads),
        "n_origin_spanning": n_spanning,
        "last_iteration": result.history[-1].iteration,
        **regional_accuracy(result.theta, target),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = [
        (divergence, depth, replicate, sampling)
        for divergence in DIVERGENCES
        for depth in DEPTHS
        for replicate in range(N_REPLICATES)
        for sampling in ("linear", "circular")
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=N_PARALLEL_RUNS) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 25 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}]", flush=True)
    rows.sort(key=lambda row: (row["divergence"], row["depth"], row["replicate"], row["sampling"]))
    with open(RESULTS_CSV, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
