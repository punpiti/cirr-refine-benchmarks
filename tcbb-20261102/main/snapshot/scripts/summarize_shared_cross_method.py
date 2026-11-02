"""Summarize every method on the shared 720-job Phase-1 grid.

The output keeps completion rate separate from F1 conditional on a method
producing an evaluable sequence. This prevents timeouts or assembly failures
from disappearing inside a success-only accuracy mean.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
KEYS = ["reference", "target", "depth", "replicate"]
DEPTHS = set(range(1, 9))
EXPECTED_JOBS = 30 * 8 * 3

SOURCES = [
    ("ASEM", "baseline1_ieee_access", True),
    ("ASEM+rec.", "baseline2_ojemb", True),
    ("ASEM-Hybrid", "baseline_hybrid_boundary", True),
    ("SRSC", "baseline3_ecticon", False),
    ("Pilon", "ext_pilon", False),
    ("MIA", "ext_mia", False),
    ("NOVOPlasty", "ext_novoplasty", False),
    ("GetOrganelle", "ext_getorganelle", False),
]


def load_final_rows(directory: str, iterative: bool) -> pd.DataFrame:
    path = RESULTS_DIR / directory / "grid_results.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["depth"].isin(DEPTHS)].copy()
    if iterative:
        frame = frame.sort_values("iteration").groupby(KEYS, as_index=False).tail(1)
    if directory == "ext_getorganelle":
        normalized_path = RESULTS_DIR / directory / "normalized_successful_rerun.csv"
        normalized = pd.read_csv(normalized_path)
        normalized = normalized[normalized["error"].fillna("") == ""]
        normalized = normalized[normalized["depth"].isin(DEPTHS)]
        expected = int(frame["f1"].notna().sum())
        if len(normalized) != expected or normalized.duplicated(KEYS).any():
            raise ValueError(
                f"{normalized_path}: expected {expected} unique normalized successes, "
                f"found {len(normalized)}"
            )
        replacement = normalized.set_index(KEYS)["f1"]
        indexed = frame.set_index(KEYS)
        indexed.loc[replacement.index, "f1"] = replacement
        frame = indexed.reset_index()
    duplicates = frame.duplicated(KEYS, keep=False)
    if duplicates.any():
        raise ValueError(f"{path}: duplicate job keys after final-row selection")
    if len(frame) != EXPECTED_JOBS:
        raise ValueError(f"{path}: expected {EXPECTED_JOBS} jobs, found {len(frame)}")
    return frame


def summarize(group: pd.DataFrame, method: str, stratum: str, value: str) -> dict:
    successful = group["f1"].notna()
    count = int(successful.sum())
    attempted = len(group)
    return {
        "method": method,
        stratum: value,
        "attempted": attempted,
        "successful": count,
        "success_rate": count / attempted,
        "mean_f1_success": group.loc[successful, "f1"].mean(),
    }


def main() -> None:
    by_divergence: list[dict] = []
    by_depth: list[dict] = []
    canonical_jobs: pd.DataFrame | None = None
    for method, directory, iterative in SOURCES:
        frame = load_final_rows(directory, iterative)
        job_signature = frame.set_index(KEYS)[["n_reads"]].sort_index()
        if canonical_jobs is None:
            canonical_jobs = job_signature
        elif not job_signature.equals(canonical_jobs):
            raise ValueError(f"{method}: job keys or read counts differ from the shared grid")
        for divergence, group in frame.groupby("divergence", sort=False):
            by_divergence.append(summarize(group, method, "divergence", divergence))
        by_divergence.append(summarize(frame, method, "divergence", "all"))
        for depth, group in frame.groupby("depth", sort=True):
            by_depth.append(summarize(group, method, "depth", int(depth)))

    divergence_path = RESULTS_DIR / "shared_cross_method_by_divergence.csv"
    depth_path = RESULTS_DIR / "shared_cross_method_by_depth.csv"
    pd.DataFrame(by_divergence).to_csv(divergence_path, index=False)
    pd.DataFrame(by_depth).to_csv(depth_path, index=False)
    print(divergence_path)
    print(depth_path)


if __name__ == "__main__":
    main()
