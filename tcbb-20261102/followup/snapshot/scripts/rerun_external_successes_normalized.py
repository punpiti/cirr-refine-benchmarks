"""Rerun report-relevant successful de novo jobs with normalized evaluation.

The original wrappers used temporary directories and did not retain assemblies.
Failure rows need no rerun because strand/origin normalization cannot turn a
missing scaffold into a sequence. NOVOPlasty is restricted to the same-genus
subset used in the manuscript table; all successful GetOrganelle rows are
rerun. Each recovered assembly is retained under the tool's results directory.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def _module(tool: str):
    if tool == "novoplasty":
        import run_grid_ext_novoplasty as module
    elif tool == "getorganelle":
        import run_grid_ext_getorganelle as module
    else:
        raise ValueError(tool)
    return module


def _key(row: dict) -> tuple[str, str, int, int]:
    return row["reference"], row["target"], int(row["depth"]), int(row["replicate"])


def _run(job: tuple[str, tuple[str, str, int, int]]) -> dict:
    tool, args = job
    return _module(tool)._run_one(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=("novoplasty", "getorganelle"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--all-divergences", action="store_true",
        help="NOVOPlasty: rerun successful rows at every divergence level, "
             "not only the same-genus subset.",
    )
    args = parser.parse_args()

    module = _module(args.tool)
    source_csv = module.GRID_RESULTS_CSV
    output_csv = os.path.join(module.RESULTS_DIR, "normalized_successful_rerun.csv")
    with open(source_csv, newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    selected = [row for row in source_rows if not row.get("error")]
    if args.tool == "novoplasty" and not args.all_divergences:
        selected = [row for row in selected if row["divergence"] == "same-genus"]
    original = {_key(row): row for row in selected}

    completed: set[tuple[str, str, int, int]] = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as handle:
            completed = {_key(row) for row in csv.DictReader(handle)}

    jobs = [key for key in original if key not in completed]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    fields = module._GRID_FIELDNAMES + [
        "original_identity_pct",
        "original_f1",
        "delta_identity_pct",
        "delta_f1",
        "assembly_fasta",
    ]
    write_header = not os.path.exists(output_csv)
    os.makedirs(module.RESULTS_DIR, exist_ok=True)
    with open(output_csv, "a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run, (args.tool, job)): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                row = future.result()
                old = original[job]
                row["original_identity_pct"] = old["identity_pct"]
                row["original_f1"] = old["f1"]
                row["delta_identity_pct"] = float(row["identity_pct"]) - float(old["identity_pct"])
                row["delta_f1"] = float(row["f1"]) - float(old["f1"])
                ref_name, target_name, depth, replicate = job
                row["assembly_fasta"] = os.path.join(
                    "assemblies", f"{ref_name}__{target_name}__d{depth}__r{replicate}.fasta"
                ) if not row.get("error") else ""
                writer.writerow(row)
                print(
                    f"{args.tool} {ref_name}->{target_name} d={depth} r={replicate} "
                    f"delta_f1={row['delta_f1']:+.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
