"""NA07000 empirical-read table: F1 and identity against the phase-3 truth (primary mask).

Source: results/results.csv; MIA, Pilon and NOVOPlasty rows are replaced by the
N-safe rerun (../evaluator_n_fix/results/real_reads.csv) once it is complete.
Also reports each starting reference against the truth (depth 0) and the
strict-mask and rCRS sensitivity summaries.

Outputs: plos-one/generated/tbl_real_reads.tex, results/real_reads_summary.csv
"""

from __future__ import annotations

import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "results.csv")
FIXED = os.path.join(HERE, "..", "evaluator_n_fix", "results", "real_reads.csv")
INITIAL = os.path.join(HERE, "results", "initial_states.csv")
OUT_TEX = os.path.join(HERE, "..", "..", "generated", "tbl_real_reads.tex")
OUT_CSV = os.path.join(HERE, "results", "real_reads_summary.csv")
REFS = [("Pan_troglodytes_NC_001643", "Chimpanzee (same family)"),
        ("Varecia_variegata_NC_012773", r"\textit{Varecia} (same order)")]
METHODS = [("ASEM", "CIRR"), ("ASEM_Hybrid", "CIRR-Hybrid"), ("SRSC", "SRSC"), ("Pilon", "Pilon"), ("MIA", "MIA")]
DEPTHS = [1, 2, 4, 8]
LABELS = ("truth_primary", "truth_strict", "truth_unmasked", "rcrs")


def load() -> list[dict]:
    rows = list(csv.DictReader(open(SRC, newline="")))
    if os.path.exists(FIXED):
        fixed = list(csv.DictReader(open(FIXED, newline="")))
        replaced = {r["method"] for r in fixed}
        if len(fixed) == 48 * len(replaced):
            rows = [r for r in rows if r["method"] not in replaced] + fixed
        else:
            print(f"N-safe rerun incomplete ({len(fixed)} rows); using original rows")
    return rows


def initial_states() -> dict:
    """Starting reference scored against each target exactly like a fixed-origin method output."""
    if not os.path.exists(INITIAL):
        sys.path.insert(0, HERE)
        import run_real_reads_all_methods as mod
        from common import read_fasta
        truth = read_fasta(mod.TRUTH_FASTA)[1].replace("-", "")
        rcrs = read_fasta(mod.RCRS_FASTA)[1]
        with open(INITIAL, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["reference", "label", "f1", "identity_pct"])
            for ref in mod.REFERENCES:
                seq = read_fasta(mod.REFERENCES[ref])[1]
                for label, target, mask in (("truth_primary", truth, mod.PRIMARY_MASK),
                                            ("truth_strict", truth, mod.STRICT_MASK),
                                            ("rcrs", rcrs, set())):
                    m = mod.masked_metrics(seq, target, mask, False)
                    w.writerow([ref, label, m["f1"], m["identity_pct"]])
    return {(r["reference"], r["label"]): r for r in csv.DictReader(open(INITIAL, newline=""))}


def main() -> None:
    rows = load()
    init = initial_states()
    summary = []
    lines = [r"\begin{tabular}{ll" + "rr" * len(DEPTHS) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d}X}}" for d in DEPTHS) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(DEPTHS))),
             "Reference & Method & " + " & ".join(["F1 & Id. (\\%)"] * len(DEPTHS)) + r" \\", r"\midrule"]
    for r_i, (ref, ref_label) in enumerate(REFS):
        i0 = init[(ref, "truth_primary")]
        lines.append(rf"\multicolumn{{2}}{{l}}{{{ref_label}}} & \multicolumn{{{2 * len(DEPTHS)}}}{{l}}"
                     rf"{{starting reference: F1 {float(i0['f1']):.4f}, identity {float(i0['identity_pct']):.2f}\%}} \\")
        for m, name in METHODS:
            cells = []
            for d in range(1, 9):
                rs = [r for r in rows if r["method"] == m and r["reference"] == ref and int(r["depth"]) == d]
                ok = [r for r in rs if r["status"] == "completed"]
                rec = {"reference": ref, "method": m, "depth": d, "n_completed": len(ok), "n_jobs": len(rs)}
                for label in LABELS:
                    for metric in ("f1", "identity_pct"):
                        vals = [float(r[f"{label}_{metric}"]) for r in ok]
                        rec[f"{label}_{metric}"] = round(statistics.mean(vals), 6) if vals else ""
                summary.append(rec)
                if d in DEPTHS:
                    if ok:
                        f1, idv = rec["truth_primary_f1"], rec["truth_primary_identity_pct"]
                        sup = rf"$^{{{len(ok)}/{len(rs)}}}$" if len(ok) < len(rs) else ""
                        cells += [f"{f1:.4f}{sup}", f"{idv:.2f}"]
                    else:
                        cells += ["--", "--"]
            lines.append(f" & {name} & " + " & ".join(cells) + r" \\")
        if r_i < len(REFS) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by plos-one/analysis/real_reads_all_methods/make_table_real_reads.py; do not edit.\n")
        fh.write("\n".join(lines) + "\n")
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    # Sensitivity: largest change of mean F1 when the strict mask or rCRS replaces the primary truth.
    for alt in ("truth_strict", "rcrs"):
        diffs = [(abs(s[f"{alt}_f1"] - s["truth_primary_f1"]), s["method"], s["reference"][:6], s["depth"])
                 for s in summary if s["truth_primary_f1"] != ""]
        print(alt, "max |mean F1 change|:", max(diffs))
    for k, v in sorted(init.items()):
        print("initial", k, round(float(v["f1"]), 4), round(float(v["identity_pct"]), 2))
    print("wrote", os.path.abspath(OUT_TEX))


if __name__ == "__main__":
    main()
