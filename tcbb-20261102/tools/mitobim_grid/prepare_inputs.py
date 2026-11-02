"""Export MITObim inputs from the frozen panel30 snapshot (run on the coordinator).

Reads depend only on (target, depth, replicate), so the grid needs 720 read
sets. They are produced by the snapshot's own simulator and seed, so MITObim
sees byte-identical reads to every other method. The export directory is
self-contained: run_mitobim_jobs.py needs only the Python standard library
and the MITObim environment.

Job order puts a prespecified stratified subset first: every ordered pair at
depths 1, 2, 4 and 8, replicate 0. The remaining grid follows. The order is
fixed before any MITObim result exists.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

PRIORITY_DEPTHS = (1, 2, 4, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original run root holding snapshot/ and snapshot.json")
    parser.add_argument("--pool-tools", type=Path, required=True, help="directory with pool_queue.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.pool_tools))
    from pool_queue import signature, verify_snapshot

    metadata = verify_snapshot(args.root)
    snap = args.root / "snapshot"
    os.environ["PANEL30_DATA_DIR"] = str(snap / "data/ieee2021_mtdna_panel")
    sys.path.insert(0, str(snap / "scripts"))
    sys.path.insert(0, str(snap / "plos-one/analysis/panel30_grid"))
    from common import read_fasta, simulate_shotgun_reads, stable_seed
    from panel30_species import SPECIES, divergence_level, fasta_path, ordered_pairs

    out = args.output
    (out / "reads").mkdir(parents=True, exist_ok=True)
    (out / "references").mkdir(exist_ok=True)
    files = {}
    for name in SPECIES:
        sequence = read_fasta(fasta_path(name))[1]
        path = out / "references" / f"{name}.fasta"
        path.write_text(f">{name}\n{sequence}\n")
        files[str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for depth in range(1, 9):
            for replicate in range(3):
                reads = simulate_shotgun_reads(sequence, float(depth), 150, stable_seed(name, depth, replicate))
                path = out / "reads" / f"{name}__{depth}X__r{replicate}.fastq.gz"
                # Constant Phred 40: the simulator has no errors, and MIRA requires qualities.
                body = "".join(f"@r{i}\n{r}\n+\n{'I' * len(r)}\n" for i, r in enumerate(reads))
                with gzip.GzipFile(path, "wb", mtime=0) as handle:
                    handle.write(body.encode())
                files[str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()

    pairs = ordered_pairs()
    first = [(r, t, d, 0) for d in PRIORITY_DEPTHS for r, t in pairs]
    chosen = set(first)
    rest = [(r, t, d, rep) for d in range(1, 9) for rep in range(3) for r, t in pairs
            if (r, t, d, rep) not in chosen]
    with (out / "jobs.tsv").open("w") as handle:
        handle.write("reference\ttarget\tdivergence\tdepth\treplicate\tpriority\n")
        for tier, jobs in (("subset", first), ("rest", rest)):
            for r, t, d, rep in jobs:
                handle.write(f"{r}\t{t}\t{divergence_level(r, t)}\t{d}\t{rep}\t{tier}\n")
    files["jobs.tsv"] = hashlib.sha256((out / "jobs.tsv").read_bytes()).hexdigest()

    manifest = {"snapshot_sha256": signature(args.root), "bench_commit": metadata["bench_commit"],
                "read_length": 150, "quality": "constant Phred 40 ('I')",
                "priority_subset": {"depths": PRIORITY_DEPTHS, "replicate": 0, "jobs": len(first)},
                "total_jobs": len(first) + len(rest), "files_sha256": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"jobs": manifest["total_jobs"], "subset": len(first), "files": len(files)}))


if __name__ == "__main__":
    main()
