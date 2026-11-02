"""Summarize normalized external-assembler reruns for manuscript updates."""

from __future__ import annotations

import csv
import json
import os
from statistics import mean


SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def rows(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(tool_dir: str, same_genus_only: bool) -> dict:
    base = os.path.join(RESULTS_DIR, tool_dir)
    original = rows(os.path.join(base, "grid_results.csv"))
    rerun = rows(os.path.join(base, "normalized_successful_rerun.csv"))
    if same_genus_only:
        original = [row for row in original if row["divergence"] == "same-genus"]

    successful = [row for row in rerun if not row.get("error")]
    expected = [row for row in original if not row.get("error")]
    deltas = [float(row["delta_f1"]) for row in successful]
    groups = {}
    for label, predicate in [
        ("1--8X", lambda depth: depth <= 8),
        ("10X", lambda depth: depth == 10),
        ("15X", lambda depth: depth == 15),
        ("20X", lambda depth: depth == 20),
        ("30X", lambda depth: depth == 30),
    ]:
        attempted = [row for row in original if predicate(int(row["depth"]))]
        completed = [row for row in successful if predicate(int(row["depth"]))]
        groups[label] = {
            "mean_f1": mean(float(row["f1"]) for row in completed) if completed else None,
            "successful": len(completed),
            "attempted": len(attempted),
        }
    return {
        "expected_successful_reruns": len(expected),
        "completed_successful_reruns": len(successful),
        "rerun_failures": len(rerun) - len(successful),
        "delta_f1_min": min(deltas) if deltas else None,
        "delta_f1_max": max(deltas) if deltas else None,
        "delta_f1_mean": mean(deltas) if deltas else None,
        "groups": groups,
    }


def main() -> None:
    summary = {
        # The accepted normalized rerun (2026-10-01) rescored every successful
        # NOVOPlasty job, not only the same-genus subset.
        "novoplasty": summarize("ext_novoplasty", same_genus_only=False),
        "getorganelle": summarize("ext_getorganelle", same_genus_only=False),
    }
    output = os.path.join(RESULTS_DIR, "external_normalized_summary.json")
    with open(output, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
