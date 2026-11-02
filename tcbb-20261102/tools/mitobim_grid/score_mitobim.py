"""Score collected MITObim results with the frozen panel30 evaluator.

Copy each worker's work directory back first, for example
  rsync -a black5-wsl:research-work/mitobim-run/ ~/research-work/mitobim-collected/black5/
then pass every collected directory. Duplicate keys across directories are an
error. External-tool scoring matches the grid: strand and circular-origin
normalization on. The primary sequence is the longest record (the grid
convention for external tools); n_contigs and total length are reported so a
multi-record output is never silently reduced.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys

FIELDS = ["method", "reference", "target", "divergence", "depth", "replicate", "priority", "status",
          "error", "runtime_s", "stationary", "last_iteration", "n_contigs", "total_len", "theta_len",
          "n_ambiguous", "f1", "identity_pct", "recall", "precision", "c_theta", "w_theta", "u_t",
          "assembly", "assembly_sha256", "host"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original run root holding snapshot/")
    parser.add_argument("--pool-tools", type=Path, required=True)
    parser.add_argument("--collected", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.pool_tools))
    from pool_queue import verify_snapshot

    verify_snapshot(args.root)
    snap = args.root / "snapshot"
    os.environ["PANEL30_DATA_DIR"] = str(snap / "data/ieee2021_mtdna_panel")
    sys.path.insert(0, str(snap / "scripts"))
    sys.path.insert(0, str(snap / "plos-one/analysis/panel30_grid"))
    sys.path.insert(0, str(snap / "plos-one/analysis/evaluator_n_fix"))
    import common
    import nsafe
    from panel30_species import fasta_path

    nsafe.install()  # same N handling as the grid evaluator
    truth: dict[str, str] = {}
    rows, seen = [], set()
    for directory in args.collected:
        for line in (directory / "results.jsonl").read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["key"] in seen:
                raise SystemExit(f"duplicate key across collected runs: {record['key']}")
            seen.add(record["key"])
            row = {k: record.get(k, "") for k in FIELDS if k in record}
            row.update(method="MITObim", error=(record.get("error") or "")[:300])
            if record["status"] == "completed":
                with gzip.open(directory / record["assembly"], "rt") as handle:
                    seqs = [s.strip() for s in handle.read().split(">")[1:]]
                seqs = ["".join(s.split("\n")[1:]) for s in seqs]
                theta = max(seqs, key=len)
                target = truth.setdefault(record["target"], common.read_fasta(fasta_path(record["target"]))[1])
                scores = common.evaluate_theta_vs_target(theta, target, normalize_strand=True,
                                                         normalize_circular_origin=True)
                row.update(n_contigs=len(seqs), total_len=sum(map(len, seqs)), theta_len=len(theta),
                           n_ambiguous=sum(c not in "ACGT" for c in theta),
                           assembly=str(directory / record["assembly"]),
                           **{k: scores[k] for k in ("f1", "identity_pct", "recall", "precision",
                                                     "c_theta", "w_theta", "u_t")})
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
