"""Validate and compare complete post-tie-rule grids against released results."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from statistics import mean


SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
STAMP = "20260929"
EXPECTED_JOBS = 720

GRIDS = {
    "baseline1": "baseline1_ieee_access",
    "baseline2": "baseline2_ojemb",
    "hybrid_boundary": "baseline_hybrid_boundary",
}


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def job_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    return (
        row["reference"],
        row["target"],
        int(row["depth"]),
        int(row["replicate"]),
    )


def final_by_job(rows: list[dict[str, str]]) -> dict[tuple, dict[str, str]]:
    final: dict[tuple, dict[str, str]] = {}
    for row in rows:
        key = job_key(row)
        if key not in final or int(row["iteration"]) > int(final[key]["iteration"]):
            final[key] = row
    return final


def main() -> None:
    report: dict[str, dict] = {}
    final_grids: dict[str, dict[tuple, dict[str, str]]] = {}
    for name, directory in GRIDS.items():
        old_path = os.path.join(
            RESULTS_DIR,
            directory,
            f"grid_results.pre_tie_rule_{STAMP}.csv",
        )
        new_path = os.path.join(
            RESULTS_DIR,
            directory,
            f"grid_results.post_tie_rule_partial_{STAMP}.csv",
        )
        old = final_by_job(read_rows(old_path))
        new_rows = read_rows(new_path)
        new = final_by_job(new_rows)
        if len(new) != EXPECTED_JOBS:
            raise SystemExit(
                f"{name}: incomplete post-tie grid: {len(new)}/{EXPECTED_JOBS} jobs"
            )
        if set(old) != set(new):
            raise SystemExit(f"{name}: pre/post job keys differ")
        final_grids[name] = new

        changed = []
        for key in sorted(new):
            before, after = old[key], new[key]
            deltas = {
                metric: float(after[metric]) - float(before[metric])
                for metric in ("f1", "identity_pct", "recall", "precision")
            }
            if any(abs(value) > 1e-12 for value in deltas.values()):
                changed.append({
                    "reference": key[0],
                    "target": key[1],
                    "depth": key[2],
                    "replicate": key[3],
                    **deltas,
                })

        report[name] = {
            "jobs": len(new),
            "rows": len(new_rows),
            "iterations": dict(sorted(Counter(int(r["iteration"]) for r in new_rows).items())),
            "changed_final_jobs": len(changed),
            "max_abs_delta_f1": max((abs(x["f1"]) for x in changed), default=0.0),
            "mean_delta_f1_changed_jobs": (
                sum(x["f1"] for x in changed) / len(changed) if changed else 0.0
            ),
            "changed_jobs": changed,
        }

    manuscript_metrics: dict[str, dict] = {"divergence_means": {}}
    for name, final in final_grids.items():
        by_divergence: dict[str, list[dict[str, str]]] = {}
        for row in final.values():
            by_divergence.setdefault(row["divergence"], []).append(row)
        manuscript_metrics["divergence_means"][name] = {
            divergence: {
                metric: mean(float(row[metric]) for row in rows)
                for metric in ("f1", "identity_pct", "recall", "precision")
            }
            for divergence, rows in sorted(by_divergence.items())
        }

    baseline1 = final_grids["baseline1"]
    baseline2 = final_grids["baseline2"]
    hybrid = final_grids["hybrid_boundary"]
    manuscript_metrics["baseline1_vs_baseline2"] = {
        "jobs_with_any_final_metric_or_length_difference": sum(
            any(baseline1[key][field] != baseline2[key][field] for field in (
                "f1", "identity_pct", "recall", "precision", "theta_len"
            ))
            for key in baseline1
        ),
        "jobs_with_f1_difference": sum(
            baseline1[key]["f1"] != baseline2[key]["f1"] for key in baseline1
        ),
        "mean_signed_f1_difference": mean(
            float(baseline2[key]["f1"]) - float(baseline1[key]["f1"])
            for key in baseline1
        ),
    }
    hybrid_f1_deltas = [
        float(hybrid[key]["f1"]) - float(baseline1[key]["f1"])
        for key in baseline1
    ]
    manuscript_metrics["hybrid_vs_baseline1"] = {
        "jobs_f1_improved": sum(delta > 1e-12 for delta in hybrid_f1_deltas),
        "jobs_f1_unchanged": sum(abs(delta) <= 1e-12 for delta in hybrid_f1_deltas),
        "jobs_f1_regressed": sum(delta < -1e-12 for delta in hybrid_f1_deltas),
        "min_f1_delta": min(hybrid_f1_deltas),
        "max_f1_delta": max(hybrid_f1_deltas),
        "mean_f1_delta": mean(hybrid_f1_deltas),
    }
    report["manuscript_metrics"] = manuscript_metrics

    output = os.path.join(RESULTS_DIR, "post_tie_rule_summary.json")
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
