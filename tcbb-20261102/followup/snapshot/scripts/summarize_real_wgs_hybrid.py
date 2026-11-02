"""Validate and summarize the final real-read ASEM/Hybrid benchmark CSV."""

from __future__ import annotations

import os

import pandas as pd

from common import evaluate_theta_vs_target, read_fasta

SCRIPT_DIR = os.path.dirname(__file__)
VARIANT = os.environ.get("ASEM_HYBRID_VARIANT", "boundary").strip().lower()
if VARIANT not in {"legacy", "boundary"}:
    raise ValueError("ASEM_HYBRID_VARIANT must be 'legacy' or 'boundary'")
RESULTS_NAME = "real_wgs_hybrid" if VARIANT == "legacy" else "real_wgs_hybrid_boundary"
HYBRID_METHOD = "ASEM_Hybrid" if VARIANT == "legacy" else "ASEM_Hybrid_boundary"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", RESULTS_NAME)
INPUT_CSV = os.path.join(RESULTS_DIR, "final_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_by_depth.csv")
INITIAL_CSV = os.path.join(RESULTS_DIR, "initial_states.csv")
WGS_DATA_DIR = os.environ.get("GENOME_WGS_DATA_DIR", os.path.join(SCRIPT_DIR, "..", "data"))
HUMAN_FASTA = os.path.join(WGS_DATA_DIR, "refs", "chrM_rCRS.fasta")  # see fetch_data.py
REFERENCES = {
    "Pan_troglodytes_NC_001643": os.path.join(WGS_DATA_DIR, "refs", "chimp_NC_001643.fasta"),
    "Varecia_variegata_NC_012773": os.path.join(
        SCRIPT_DIR, "..", "data", "phase1_reproduction",
        "Varecia_variegata_NC_012773.fasta",
    ),
}
METRICS = ["f1", "identity_pct", "precision", "recall"]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    key = ["reference", "method", "depth", "replicate"]
    assert len(df) == 120
    assert not df.duplicated(key).any()
    assert not df[key + METRICS].isna().any().any()
    assert df["depth"].between(1, 8).all()
    assert df["replicate"].isin([0, 1, 2]).all()
    assert df[METRICS].apply(lambda col: col.between(0, 100 if col.name == "identity_pct" else 1).all()).all()

    expected = {
        ("Pan_troglodytes_NC_001643", "ASEM_no_recursion"): 24,
        ("Pan_troglodytes_NC_001643", "ASEM_recursion"): 24,
        ("Pan_troglodytes_NC_001643", HYBRID_METHOD): 24,
        ("Varecia_variegata_NC_012773", "ASEM_no_recursion"): 24,
        ("Varecia_variegata_NC_012773", HYBRID_METHOD): 24,
    }
    assert df.groupby(["reference", "method"]).size().to_dict() == expected

    wide = df.pivot(index=["reference", "depth", "replicate"], columns="method")
    chimp = wide.loc["Pan_troglodytes_NC_001643"]
    for metric in METRICS:
        assert (chimp[(metric, HYBRID_METHOD)] == chimp[(metric, "ASEM_no_recursion")]).all()

    varecia = wide.loc["Varecia_variegata_NC_012773"]
    delta = varecia[("f1", HYBRID_METHOD)] - varecia[("f1", "ASEM_no_recursion")]
    assert (delta > 0).all()

    summary = (
        df.groupby(["reference", "divergence", "method", "depth"], as_index=False)
        .agg(
            n_jobs=("replicate", "size"),
            f1_mean=("f1", "mean"),
            f1_min=("f1", "min"),
            f1_max=("f1", "max"),
            identity_mean=("identity_pct", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            anchors_mean=("n_anchors_total", "mean"),
        )
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    _, human = read_fasta(HUMAN_FASTA)
    initial_rows = []
    for name, path in REFERENCES.items():
        _, reference = read_fasta(path)
        initial_rows.append({"reference": name, **evaluate_theta_vs_target(reference, human)})
    pd.DataFrame(initial_rows).to_csv(INITIAL_CSV, index=False)

    print("validated 120 unique final-job rows; all expected cells present")
    chimp_hybrid = df[
        (df.reference == "Pan_troglodytes_NC_001643")
        & (df.method == HYBRID_METHOD)
    ]
    print(
        "chimp: Hybrid identical to plain ASEM in 24/24 jobs; "
        f"{int((chimp_hybrid.n_anchors_total > 0).sum())} jobs accepted anchors"
    )
    print(
        "Varecia: Hybrid F1 improved in 24/24 jobs; "
        f"mean delta={delta.mean():.6f}, range={delta.min():.6f}--{delta.max():.6f}"
    )
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {INITIAL_CSV}")


if __name__ == "__main__":
    main()
