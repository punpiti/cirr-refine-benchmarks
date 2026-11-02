#!/usr/bin/env python3
"""Compare the reported free-end evaluator with strict global alignment."""
from __future__ import annotations
import csv
from pathlib import Path
from skbio import DNA
from skbio.alignment import pair_align_nucl
from baseline_ieee_access import run_asem
from common import evaluate_theta_vs_target, read_fasta, simulate_shotgun_reads, stable_seed
from phase1_species import fasta_path

PAIRS = [
    ("Saimiri_boliviensis", "Saimiri_sciureus", "same-genus"),
    ("Gorilla_gorilla", "Homo_sapiens", "same-family"),
    ("Varecia_variegata", "Homo_sapiens", "same-order"),
]
OUT = Path(__file__).parent / "results" / "evaluator_endgap_sanity.csv"

def score(theta: str, target: str, free_ends: bool) -> dict[str, float]:
    result = pair_align_nucl(DNA(theta), DNA(target), mode="global", free_ends=free_ends)
    a, b = result.paths[0].to_aligned([DNA(theta), DNA(target)])
    c = w = u = ident = n = 0
    for x, y in zip(str(a), str(b)):
        n += 1
        if x == "-": u += 1
        elif y == "-": pass
        elif x == y: c += 1; ident += 1
        else: w += 1
    recall = c / (c + u) if c + u else 0.0
    precision = c / (c + w) if c + w else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"identity_pct": 100 * ident / n, "recall": recall, "precision": precision, "f1": f1}

rows = []
for reference_name, target_name, divergence in PAIRS:
    _, reference = read_fasta(fasta_path(reference_name))
    _, target = read_fasta(fasta_path(target_name))
    reads = simulate_shotgun_reads(target, 4.0, 150, stable_seed(target_name, 4, 0))
    result = run_asem(reference, reads, max_iterations=6, n_workers=1)
    theta = result.theta
    free = evaluate_theta_vs_target(theta, target)
    strict = score(theta, target, False)
    rows.append({
        "reference": reference_name, "target": target_name, "divergence": divergence,
        "depth": 4, "replicate": 0,
        "free_f1": free["f1"], "strict_f1": strict["f1"],
        "delta_f1_strict_minus_free": strict["f1"] - free["f1"],
        "free_identity_pct": free["identity_pct"], "strict_identity_pct": strict["identity_pct"],
        "delta_identity_pp_strict_minus_free": strict["identity_pct"] - free["identity_pct"],
    })
    print(rows[-1], flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows)
print(OUT)
