"""Real-read benchmark (NA07000) for every method against a sample-specific truth.

Author decision 2026-10-28 (Phase 5, option A): the truth is the 1000 Genomes
phase-3 manually curated NA07000 mtDNA sequence
(release/20130502/supporting/MT/chrMT_sequences_2534.20160505.fasta.gz),
derived from earlier 1000G Illumina data independent of the PRJEB31736 reads
used as input; its 27 SNVs called by Complete Genomics (SRR801821) all agree.
The rCRS-based score of the v1.3.0 analysis is kept for continuity.

Inputs reproduce run_real_wgs_hybrid_benchmark.py (v1.3.0) exactly: 150-bp
primary, mapped, non-duplicate records (samtools -F 3332) from
data/NA07000_chrM.bam, subsampled without replacement with
stable_seed("NA07000_real", depth, replicate); starting references chimpanzee
NC_001643 (same-family) and Varecia variegata NC_012773 (same-order);
1-8X; three replicates.

Masks are fixed before any result is examined (rCRS coordinates, 1-based):
  primary: 7230 (~22% minor allele in the input reads), poly-C tracts 303-317
           and 16180-16195 (310 insertion not representable in the truth);
  strict:  primary + Complete Genomics no-call sites 930, 1888, 4769, 5147,
           7028, 7227-7233, 8860 + minor-allele clusters 439-545, 3572-3631,
           10935-10953, 13761-13789 and sites 857, 1573, 8368.
The phase-3 sequence carries '-' at the rCRS 3107 placeholder; it is removed,
and mask coordinates after 3107 shift by one.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "panel30_grid"))
from resume_io import checkpoint, recover_csv  # noqa: E402
BENCH_SCRIPTS = os.environ.get("BENCH_SCRIPTS", os.path.expanduser("~/plos-reruns/bench-v1.3.0/scripts"))
sys.path.insert(0, BENCH_SCRIPTS)
sys.path.insert(0, os.path.join(HERE, "..", "read_length_all_methods"))

import common  # noqa: E402
from common import read_fasta, stable_seed  # noqa: E402
from run_read_length_all_methods import METHODS  # noqa: E402
from skbio.alignment import pair_align_nucl  # noqa: E402
from skbio.sequence import DNA  # noqa: E402

DATA = os.path.join(BENCH_SCRIPTS, "..", "data")
BAM = os.path.join(DATA, "NA07000_chrM.bam")
RCRS_FASTA = os.path.join(DATA, "refs", "chrM_rCRS.fasta")
REFERENCES = {
    "Pan_troglodytes_NC_001643": os.path.join(DATA, "refs", "chimp_NC_001643.fasta"),
    "Varecia_variegata_NC_012773": os.path.join(DATA, "phase1_reproduction", "Varecia_variegata_NC_012773.fasta"),
}
TRUTH_FASTA = os.path.join(HERE, "truth", "NA07000_1000G_phase3_chrMT.fasta")
SAMTOOLS = os.environ.get("SAMTOOLS_BIN", os.path.expanduser("~/.local/share/mamba/envs/genome/bin/samtools"))
RESULTS_DIR = os.environ.get("CIRR_RESULTS_DIR", os.path.join(HERE, "results"))
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
DEPTHS = range(1, 9)
N_REPLICATES = 3
READ_LENGTH = 150

PRIMARY_MASK_RCRS = {7230} | set(range(303, 318)) | set(range(16180, 16196))
STRICT_EXTRA_RCRS = ({930, 1888, 4769, 5147, 7028, 8860, 857, 1573, 8368} | set(range(7227, 7234))
                     | set(range(439, 546)) | set(range(3572, 3632)) | set(range(10935, 10954))
                     | set(range(13761, 13790)))


def rcrs_to_truth_index(pos: int) -> int | None:
    """1-based rCRS position -> 0-based index in the truth (3107 removed)."""
    if pos == 3107:
        return None
    return pos - 1 if pos < 3107 else pos - 2


PRIMARY_MASK = {i for p in PRIMARY_MASK_RCRS if (i := rcrs_to_truth_index(p)) is not None}
STRICT_MASK = PRIMARY_MASK | {i for p in STRICT_EXTRA_RCRS if (i := rcrs_to_truth_index(p)) is not None}


def masked_metrics(theta: str, target: str, mask: set[int], normalize: bool) -> dict:
    """common.evaluate_theta_vs_target with columns at masked target positions dropped.

    Orientation and circular origin are chosen exactly as in the v1.3.0
    evaluator; an inserted output base is attributed to the preceding target
    position for masking.
    """
    orientations = [theta]
    if normalize:
        rc = str(DNA(theta).reverse_complement())
        if rc != theta:
            orientations.append(rc)
    best = None
    for oriented in orientations:
        offset, score = common._rotation_offset_and_score(oriented, target) if normalize else (0, 0)
        if best is None or score > best[0]:
            best = (score, oriented, offset)
    _, oriented, offset = best
    rotated = target[offset:] + target[:offset]
    path = pair_align_nucl(DNA(oriented), DNA(rotated), mode="global").paths[0]
    a_theta, a_target = (str(s) for s in path.to_aligned([DNA(oriented), DNA(rotated)]))
    c = w = u = ident = n = 0
    tpos = -1
    for a, b in zip(a_theta, a_target):
        if b != "-":
            tpos += 1
        orig = (tpos + offset) % len(target) if tpos >= 0 else (offset - 1) % len(target)
        if orig in mask:
            continue
        n += 1
        if a == "-":
            u += 1
        elif b == "-":
            pass
        elif a == b:
            c += 1
            ident += 1
        else:
            w += 1
    recall = c / (c + u) if c + u else 0.0
    precision = c / (c + w) if c + w else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"identity_pct": 100 * ident / n if n else 0.0, "recall": recall, "precision": precision,
            "f1": f1, "c_theta": c, "w_theta": w, "u_t": u}


def extract_reads() -> list[str]:
    out = subprocess.run([SAMTOOLS, "view", "-F", "3332", BAM], capture_output=True, text=True, check=True).stdout
    return [f[9] for f in (line.split("\t") for line in out.splitlines()) if len(f[9]) == READ_LENGTH]


_CTX: dict = {}


def _init(all_reads: list[str], truth: str, rcrs: str) -> None:
    _CTX.update(all_reads=all_reads, truth=truth, rcrs=rcrs)


def run_one(job: tuple[str, str, int, int]) -> dict:
    method, ref_name, depth, rep = job
    truth, rcrs, all_reads = _CTX["truth"], _CTX["rcrs"], _CTX["all_reads"]
    reference = read_fasta(REFERENCES[ref_name])[1]
    n_needed = min(round(depth * len(rcrs) / READ_LENGTH), len(all_reads))
    seed = stable_seed("NA07000_real", depth, rep)
    reads = random.Random(seed).sample(all_reads, n_needed)
    fn, external = METHODS[method]
    row = {"method": method, "reference": ref_name,
           "divergence": "same-family" if ref_name.startswith("Pan_") else "same-order",
           "depth": depth, "replicate": rep, "n_reads": len(reads), "host": platform.node(), "error": ""}
    started = time.time()
    try:
        theta = fn(reference, reads, READ_LENGTH, seed, f"na07000_{ref_name[:3]}_{depth}_{rep}")
    except Exception as exc:
        msg = str(exc).lower()
        status = ("timeout" if "timed out" in msg else
                  "no_output" if any(s in msg for s in ("no scaffold", "no contig", "empty", "no final")) else
                  "error")
        return {**row, "status": status, "error": str(exc)[:400], "runtime_s": time.time() - started}
    row.update(status="completed", runtime_s=time.time() - started, theta_len=len(theta))
    for label, target, mask in (("truth_primary", truth, PRIMARY_MASK), ("truth_strict", truth, STRICT_MASK),
                                ("truth_unmasked", truth, set()), ("rcrs", rcrs, set())):
        for k, v in masked_metrics(theta, target, mask, external).items():
            row[f"{label}_{k}"] = v
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lock = open(os.path.join(RESULTS_DIR, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another runner holds the lock")

    recover_csv(RESULTS_CSV, run_fields())
    truth = read_fasta(TRUTH_FASTA)[1].replace("-", "")
    rcrs = read_fasta(RCRS_FASTA)[1]
    all_reads = extract_reads()
    done = set()
    if os.path.exists(RESULTS_CSV):
        done = {(r["method"], r["reference"], int(r["depth"]), int(r["replicate"]))
                for r in csv.DictReader(open(RESULTS_CSV, newline=""))}
    jobs = [(m, ref, d, rep) for ref in REFERENCES for d in DEPTHS for rep in range(N_REPLICATES)
            for m in args.methods.split(",") if (m, ref, d, rep) not in done]
    with open(os.path.join(RESULTS_DIR, f"manifest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"), "w") as fh:
        json.dump({"host": platform.node(), "python": sys.version, "bench_scripts": BENCH_SCRIPTS,
                   "n_pool_reads": len(all_reads), "truth_len": len(truth), "n_jobs": len(jobs),
                   "primary_mask_rcrs": sorted(PRIMARY_MASK_RCRS),
                   "strict_mask_rcrs": sorted(PRIMARY_MASK_RCRS | STRICT_EXTRA_RCRS)}, fh, indent=2)
    print(f"=== NA07000 all methods: {len(all_reads):,} pool reads, {len(jobs)} jobs ===", flush=True)

    fields = run_fields()
    with ProcessPoolExecutor(max_workers=args.parallel, initializer=_init,
                             initargs=(all_reads, truth, rcrs)) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            is_new = not os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if is_new:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in fields})
                checkpoint(fh)
            print(f"[{n}/{len(jobs)}] {row['method']} {row['reference'][:7]} {row['depth']}X r{row['replicate']} "
                  f"{row['status']} F1(primary)={row.get('truth_primary_f1', '')} t={row['runtime_s']:.0f}s",
                  flush=True)


def run_fields() -> list[str]:
    base = ["method", "reference", "divergence", "depth", "replicate", "n_reads", "status", "error",
            "runtime_s", "theta_len"]
    metrics = ["identity_pct", "recall", "precision", "f1", "c_theta", "w_theta", "u_t"]
    return base + [f"{lab}_{m}" for lab in ("truth_primary", "truth_strict", "truth_unmasked", "rcrs")
                   for m in metrics] + ["host"]


if __name__ == "__main__":
    main()
