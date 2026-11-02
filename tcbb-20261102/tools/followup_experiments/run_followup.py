"""Follow-up experiments on the strand-aware code snapshot (resumable, unattended).

Experiments (pairs prespecified before any result, same for both):
  mixed_orientation  1, 2, 4, 8X; each read reverse-complemented with p=0.5
                     (seeded); all methods plus the unchanged reference.
                     Forward-read CIRR/Hybrid controls on the same code.
  high_depth         15X and 30X forward reads; all methods plus reference.
Pairs: 3 per divergence tier (12 ordered pairs), drawn with a fixed seed.

Reads use the grid's simulator and seed, so forward reads at 1-8X are
byte-identical to the main grid. Scoring matches the grid: N-neutral
evaluator, strand and origin normalization for external tools only.
Each result is one fsynced JSON line in WORK/results.jsonl; rerunning skips
finished jobs and reruns a torn final line.

  python -I run_followup.py --snapshot SNAP plan --experiment mixed_orientation --work WORK
  python -I run_followup.py --snapshot SNAP run --work WORK --parallel 24
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
import fcntl  # noqa: E402
import functools  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

PAIR_SEED = 20261101
PAIRS_PER_TIER = 3
METHOD_ORDER = ["Reference", "ASEM", "ASEM_Hybrid", "SRSC", "Pilon", "MIA", "NOVOPlasty", "GetOrganelle"]
TIMEOUT_S = 3600
M = {}  # filled by load()


def load(snapshot: Path) -> dict:
    """Verify the follow-up snapshot and import its modules."""
    meta = json.loads((snapshot / "followup_snapshot.json").read_text())
    for relative, digest in meta["files_sha256"].items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"snapshot mismatch: {relative}")
    snap = snapshot / "snapshot"
    os.environ["PANEL30_DATA_DIR"] = str(snap / "data/ieee2021_mtdna_panel")
    for path in (snap / "scripts", snap / "plos-one/analysis/panel30_grid",
                 snap / "plos-one/analysis/evaluator_n_fix"):
        sys.path.insert(0, str(path))
    import common
    import ext_getorganelle
    import ext_novoplasty
    import nsafe
    from asem_hybrid import run_asem_hybrid_loop
    from baseline_ecticon import run_ecticon
    from baseline_ieee_access import _align_read_ieee_access, run_asem
    from ext_mia import run_mia
    from ext_pilon import run_pilon
    import panel30_species

    nsafe.install()
    hybrid_align = functools.partial(_align_read_ieee_access, tau=0.5)
    # Same parameters as the main grid (run_panel30_grid.METHODS).
    methods = {
        "Reference": (lambda s, r, seed, tag: s, False),
        "ASEM": (lambda s, r, seed, tag: run_asem(s, r).theta, False),
        "ASEM_Hybrid": (lambda s, r, seed, tag: run_asem_hybrid_loop(
            s, r, align_read_fn=hybrid_align, boundary_flank=150, max_iterations=6, n_workers=1).theta, False),
        "SRSC": (lambda s, r, seed, tag: run_ecticon(s, r).theta, False),
        "Pilon": (lambda s, r, seed, tag: run_pilon(s, r, timeout=TIMEOUT_S), True),
        "MIA": (lambda s, r, seed, tag: run_mia(s, r, timeout=TIMEOUT_S), True),
        "NOVOPlasty": (lambda s, r, seed, tag: ext_novoplasty.run_novoplasty(
            s, r, project=tag, timeout=TIMEOUT_S), True),
        "GetOrganelle": (lambda s, r, seed, tag: ext_getorganelle.run_getorganelle(
            s, r, quality_seed=seed, timeout=TIMEOUT_S), True),
    }
    M.update(common=common, species=panel30_species, methods=methods, meta=meta, seqs={})
    return meta


def seq(name: str) -> str:
    if name not in M["seqs"]:
        M["seqs"][name] = M["common"].read_fasta(M["species"].fasta_path(name))[1]
    return M["seqs"][name]


def select_pairs() -> list[tuple[str, str, str]]:
    by_tier: dict[str, list[tuple[str, str]]] = {}
    for ref, target in M["species"].ordered_pairs():
        by_tier.setdefault(M["species"].divergence_level(ref, target), []).append((ref, target))
    rng = random.Random(PAIR_SEED)
    chosen = []
    for tier in M["species"].TIERS:
        for ref, target in rng.sample(sorted(by_tier[tier]), PAIRS_PER_TIER):
            chosen.append((tier, ref, target))
    return chosen


def plan(experiment: str, work: Path) -> None:
    pairs = select_pairs()
    jobs = []
    if experiment == "mixed_orientation":
        for tier, ref, target in pairs:
            for depth in (1, 2, 4, 8):
                for rep in range(3):
                    for method in METHOD_ORDER:
                        jobs.append((method, ref, target, depth, rep, "mixed"))
                    for method in ("ASEM", "ASEM_Hybrid"):
                        jobs.append((method, ref, target, depth, rep, "forward"))
    elif experiment == "high_depth":
        for tier, ref, target in pairs:
            for depth in (15, 30):
                for rep in range(3):
                    for method in METHOD_ORDER:
                        jobs.append((method, ref, target, depth, rep, "forward"))
    else:
        raise SystemExit(f"unknown experiment {experiment}")
    work.mkdir(parents=True, exist_ok=True)
    if (work / "jobs.jsonl").exists():
        raise SystemExit("plan exists; refusing to overwrite")
    with (work / "jobs.jsonl").open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(dict(zip(("method", "reference", "target", "depth", "replicate", "reads"), job))) + "\n")
    (work / "plan.json").write_text(json.dumps({
        "experiment": experiment, "pair_seed": PAIR_SEED, "pairs": pairs, "jobs": len(jobs),
        "snapshot_files_sha256": hashlib.sha256(json.dumps(M["meta"]["files_sha256"], sort_keys=True).encode()).hexdigest(),
    }, indent=1))
    print(json.dumps({"experiment": experiment, "jobs": len(jobs), "pairs": pairs}))


def key(job: dict) -> str:
    return f"{job['method']}|{job['reference']}|{job['target']}|{job['depth']}|{job['replicate']}|{job['reads']}"


def run_job(args: tuple) -> dict:
    work, job = args
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    common = M["common"]
    target_name, depth, rep = job["target"], int(job["depth"]), int(job["replicate"])
    target = seq(target_name)
    seed = common.stable_seed(target_name, depth, rep)
    reads = common.simulate_shotgun_reads(target, float(depth), 150, seed)
    if job["reads"] == "mixed":
        flip = random.Random(common.stable_seed(target_name, depth, rep, "mixed"))
        reads = [common.reverse_complement(r) if flip.random() < 0.5 else r for r in reads]
    slug = f"{job['reference']}__{target_name}__{depth}X__r{rep}__{job['reads']}"
    row = {"key": key(job), **job, "divergence": M["species"].divergence_level(job["reference"], target_name),
           "n_reads": len(reads), "host": platform.node()}
    fn, external = M["methods"][job["method"]]
    started = time.time()
    try:
        theta = fn(seq(job["reference"]), reads, seed, f"fu_{slug}")
    except Exception as exc:  # each wrapper raises its own failure type
        message = str(exc)
        log = Path(work) / "errors" / job["method"] / f"{slug}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(message)
        lowered = message.lower()
        status = ("timeout" if "timed out" in lowered else
                  "no_output" if any(k in lowered for k in ("no scaffold", "no contig", "empty", "no final")) else "error")
        return {**row, "status": status, "error": message[:300], "runtime_s": round(time.time() - started, 2)}
    runtime = round(time.time() - started, 2)
    if not theta:
        return {**row, "status": "no_output", "runtime_s": runtime}
    path = Path(work) / "thetas" / job["method"] / f"{slug}.fa.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    with gzip.GzipFile(temporary, "wb", mtime=0) as handle:
        handle.write(f">{job['method']} {slug}\n{theta}\n".encode())
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    scores = common.evaluate_theta_vs_target(theta, target, normalize_strand=external,
                                             normalize_circular_origin=external)
    return {**row, "status": "completed", "runtime_s": runtime, "theta_len": len(theta),
            "n_ambiguous": sum(c not in "ACGT" for c in theta), "theta_fasta": str(path.relative_to(work)),
            "theta_sha256": hashlib.sha256(theta.encode()).hexdigest(),
            **{k: scores[k] for k in ("f1", "identity_pct", "recall", "precision", "c_theta", "w_theta", "u_t")}}


def run(work: Path, parallel: int) -> None:
    lock = (work / ".runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [json.loads(line) for line in (work / "jobs.jsonl").read_text().splitlines()]
    results = work / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # torn final line
    pending = iter([j for j in jobs if key(j) not in done])
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    finished, active = len(done), set()
    import multiprocessing
    with ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("fork")) as pool:
        while True:
            while not stopping and len(active) < parallel:
                job = next(pending, None)
                if job is None:
                    break
                active.add(pool.submit(run_job, (str(work), job)))
            if not active:
                break
            completed, active = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            with results.open("a") as handle:
                for future in completed:
                    handle.write(json.dumps(future.result(), sort_keys=True) + "\n")
                    finished += 1
                handle.flush()
                os.fsync(handle.fileno())
            (work / "status.json").write_text(json.dumps({
                "pid": os.getpid(), "host": platform.node(), "finished": finished, "total": len(jobs),
                "active": len(active), "stopping": stopping, "updated_unix": time.time()}))
    print(json.dumps({"finished": finished, "total": len(jobs), "stopped_early": stopping}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True, help="follow-up snapshot dir (followup_snapshot.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--experiment", required=True, choices=("mixed_orientation", "high_depth"))
    p.add_argument("--work", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()
    load(args.snapshot.resolve())
    if args.command == "plan":
        plan(args.experiment, args.work.resolve())
    else:
        run(args.work.resolve(), args.parallel)


if __name__ == "__main__":
    main()
