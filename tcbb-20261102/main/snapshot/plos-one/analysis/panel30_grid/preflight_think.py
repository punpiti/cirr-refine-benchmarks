"""Verify design, local tools, data agreement and package/benchmark core parity."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(project / "asem-refine-benchmarks/scripts"), str(project / "asem-refine/src"),
                str(Path(__file__).resolve().parent)]
import numpy as np
import asem_core as bench
from asem_refine import core as package
from panel30_species import SPECIES, divergence_level, fasta_path, ordered_pairs
import ext_mia
import ext_pilon
import ext_novoplasty
import ext_getorganelle

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()
counts = dict(collections.Counter(divergence_level(*p) for p in ordered_pairs()))
assert counts == {"same-genus": 8, "same-family": 124, "same-order": 468, "different-order": 270}
hashes = {}
for name, (filename, *_) in SPECIES.items():
    sha = hashlib.sha256(Path(fasta_path(name)).read_bytes()).hexdigest()
    assert sha == hashlib.sha256((project / "bmc-genomics/data/ieee2021_mtdna_panel" / filename).read_bytes()).hexdigest()
    hashes[name] = sha
tools = [ext_mia.MIA_BIN, ext_mia.MA_BIN, ext_pilon.BWA, ext_pilon.SAMTOOLS,
         ext_pilon.JAVA, ext_getorganelle.GETORGANELLE_PYTHON, ext_getorganelle.GETORGANELLE_BIN]
for tool in tools:
    assert Path(tool).is_file() and os.access(tool, os.X_OK), f"missing executable: {tool}"
assert Path(ext_pilon.PILON_JAR).is_file()
assert Path(ext_novoplasty.NOVOPLASTY_PL).is_file()
rng = np.random.default_rng(20261030)
theta = "".join(rng.choice(list("ACGT"), size=3000))
reference = rng.integers(0, 5, size=(5, len(theta)))
reference[:, ::7] = 0
insertions = {slot: rng.integers(0, 5, size=(4, 2)) for slot in range(0, len(theta) + 1, 3)}
spans = np.zeros(len(theta) + 1, dtype=np.int64)
for slot, values in insertions.items():
    spans[slot] = int(values.sum(axis=0).max()) + 2
a = bench.CombinedAlignmentCounts(reference.copy(), insertions, spans.copy())
b = package.CombinedAlignmentCounts(reference.copy(), insertions, spans.copy())
assert bench.build_theta(theta, a, 0.1) == package.build_theta(theta, b, 0.1)
report = {"status": "pass", "tiers": counts, "jobs": len(ordered_pairs()) * 8 * 3 * 7,
          "input_sha256": hashes, "parity_reference_columns": len(theta),
          "parity_insertion_columns": len(insertions) * 2, "external_executables_checked": len(tools)}
if args.output:
    args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "input_sha256"}))
