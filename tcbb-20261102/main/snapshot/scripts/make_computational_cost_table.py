"""Job-grain-safe summary of results/computational_cost/cost_profile.csv,
computing tbl:computational-cost's numbers (bmc/thai-sections/03-results.tex).

cost_profile.csv (written by profile_computational_cost.py) has one row per
(job, iteration): wall_s_total_run and peak_rss_kb are per-job totals, so
they're duplicated identically across every iteration row for that job (a
job that ran 5 EM iterations contributes the same wall_s_total_run value 5
times). Averaging the raw CSV directly is therefore row-weighted, not
job-weighted -- jobs that happened to run more iterations get counted more
times, silently inflating the mean (verified 2026-08-20: naive row mean
gives 12.43s for ASEM-no-recursion at 8X vs the correct 10.54s job mean --
this is the exact drift a second-opinion review caught in Table 4's
predecessor CSVs and is why this script exists as the one place that does
the deduplication, instead of every consumer re-deriving it by hand).

Usage: python3 make_computational_cost_table.py
"""

from __future__ import annotations

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(SCRIPT_DIR, "results", "computational_cost", "cost_profile.csv")

JOB_COLS = ["divergence", "reference", "target", "depth", "replicate", "method"]
METHOD_LABELS = {
    "asem_no_recursion": "ASEM (no rec.)",
    "asem_recursion": "ASEM (with rec.)",
    "srsc": "SRSC",
}


def job_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per job: wall_s_total_run/peak_rss_kb are constant within a
    job (duplicated per-iteration in the raw CSV), so drop_duplicates on the
    job key is the correct de-duplication -- not a groupby-mean, which would
    silently average identical values with itself and hide a mixed-iteration
    join bug rather than catch it."""
    per_job = df[[*JOB_COLS, "wall_s_total_run", "peak_rss_kb", "n_reads"]].drop_duplicates(subset=JOB_COLS)
    # Sanity check: exactly one wall_s_total_run/peak_rss_kb value per job.
    n_bad = df.groupby(JOB_COLS)[["wall_s_total_run", "peak_rss_kb"]].nunique().gt(1).any(axis=1).sum()
    if n_bad:
        raise ValueError(f"{n_bad} jobs have inconsistent wall_s_total_run/peak_rss_kb across iterations")
    return per_job


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    jobs = job_level(df)

    print(f"job-level rows: {len(jobs)} (raw CSV rows: {len(df)}, "
          f"{len(df) / len(jobs):.2f}x duplication from per-iteration rows)")
    print()

    runtime = jobs.groupby(["method", "depth"])["wall_s_total_run"].mean().unstack("depth")
    rss = jobs.groupby("method")["peak_rss_kb"].mean() / 1024

    print("=== Table 4 (tbl:computational-cost): mean wall_s_total_run by method x depth ===")
    for method in ["asem_no_recursion", "asem_recursion", "srsc"]:
        row = runtime.loc[method]
        vals = " & ".join(f"{row[d]:.2f}" for d in [1, 2, 4, 8])
        print(f"  {METHOD_LABELS[method]:<20} {vals} & {rss[method]:.1f} \\\\")

    print()
    print("=== sanity: row-weighted (WRONG) mean for comparison ===")
    wrong = df.groupby(["method", "depth"])["wall_s_total_run"].mean().unstack("depth")
    for method in ["asem_no_recursion", "asem_recursion", "srsc"]:
        print(f"  {method}: {wrong.loc[method].to_dict()}")


if __name__ == "__main__":
    main()
