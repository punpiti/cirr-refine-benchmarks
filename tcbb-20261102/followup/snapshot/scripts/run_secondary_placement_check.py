"""Check for competing read placements (multi-mapping) in the hardest pair.

ASEM assigns each read to its single best local alignment. To test whether
reads have a second, independently acceptable placement, this script takes the
grid reads for the hardest same-order pair (Homo_sapiens reference ->
Varecia_variegata target, 8X, three replicates; 905 reads each), finds each
read's best local alignment, masks that reference interval (extended by one
read length on each side so shifted copies of the same placement are not
counted), and recomputes the best alternative alignment. A placement passes
when its score is at least tau * s_match * |read| (tau = 0.5), the acceptance
rule used throughout the benchmark.

The check is run against two references: the initial reference (the first
E-step) and the final non-recursive ASEM estimate for the same reads.

Writes results/secondary_placement/results.csv and summary.json.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import parasail

sys.path.insert(0, os.path.dirname(__file__))

from baseline_ieee_access import run_asem  # noqa: E402
from common import (  # noqa: E402
    _SW_GAP_EXTEND, _SW_GAP_OPEN, align_local, read_fasta, simulate_shotgun_reads, stable_seed,
)
from phase1_species import fasta_path  # noqa: E402

REFERENCE, TARGET, DEPTH = "Homo_sapiens", "Varecia_variegata", 8
TAU, MATCH = 0.5, 2
_MASK_MATRIX = parasail.matrix_create("ACGTN", 2, -3)
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "secondary_placement")


def best_after_mask(read: str, ref: str, start: int, end: int) -> float:
    pad = len(read)
    lo, hi = max(0, start - pad), min(len(ref), end + pad)
    masked = ref[:lo] + "N" * (hi - lo) + ref[hi:]
    res = parasail.sw_striped_16(read, masked, _SW_GAP_OPEN, _SW_GAP_EXTEND, _MASK_MATRIX)
    return float(max(res.score, 0))


def check(args: tuple[int, str, str, list[str]]) -> list[dict]:
    replicate, which, ref, reads = args
    rows = []
    for i, read in enumerate(reads):
        full = MATCH * len(read)
        aln = align_local(read, ref)
        if aln is None:
            primary, secondary = 0.0, 0.0
        else:
            span = len(aln.ref_aligned) - aln.ref_aligned.count("-")
            primary = aln.score
            secondary = best_after_mask(read, ref, aln.ref_start, aln.ref_start + span)
        rows.append({
            "replicate": replicate, "reference_state": which, "read_index": i,
            "read_length": len(read), "primary_score": primary, "secondary_score": secondary,
            "primary_fraction_of_max": primary / full, "secondary_fraction_of_max": secondary / full,
            "primary_passes": primary >= TAU * full, "secondary_passes": secondary >= TAU * full,
            "secondary_ge_90pct_primary": primary > 0 and secondary >= 0.9 * primary,
        })
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    _, s = read_fasta(fasta_path(REFERENCE))
    _, t = read_fasta(fasta_path(TARGET))
    jobs = []
    for replicate in range(3):
        reads = simulate_shotgun_reads(t, depth=float(DEPTH), read_length=150,
                                       seed=stable_seed(TARGET, DEPTH, replicate))
        final = run_asem(s, reads).theta
        jobs.append((replicate, "initial", s, reads))
        jobs.append((replicate, "final", final, reads))
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = [r for part in pool.map(check, jobs) for r in part]
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for which in ("initial", "final"):
        for replicate in range(3):
            sub = [r for r in rows if r["reference_state"] == which and r["replicate"] == replicate]
            near = [r for r in sub if r["secondary_ge_90pct_primary"]]
            summary[f"{which}_rep{replicate}"] = {
                "reads": len(sub),
                "primary_passes": sum(r["primary_passes"] for r in sub),
                "secondary_passes": sum(r["secondary_passes"] for r in sub),
                "secondary_ge_90pct_primary": len(near),
                "near_pairs_both_fail": sum(not r["primary_passes"] and not r["secondary_passes"] for r in near),
                "near_pairs_primary_fraction_range": [min((r["primary_fraction_of_max"] for r in near), default=None),
                                                      max((r["primary_fraction_of_max"] for r in near), default=None)],
                "max_secondary_fraction_of_max": max(r["secondary_fraction_of_max"] for r in sub),
            }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
