"""Run MITObim on exported panel30 inputs; resumable, unattended, stdlib only.

Each finished job is appended as one fsynced JSON line to results.jsonl in
the work directory, so a crash, reboot or lost SSH session loses at most the
jobs in flight. Restarting with the same arguments skips finished jobs. A
truncated final line is ignored and its job rerun. Scoring happens later on
the coordinator with the frozen evaluator (score_mitobim.py).

MITObim 1.9.1 bundles a statically linked MIRA 4.0.2. It needs a kernel with
vsyscall emulation (WSL2: kernelCommandLine=vsyscall=emulate) and the C
locale; without them mirabait segfaults or aborts in loadlocale.c.

MITObim settings: -quick with the reference as seed, defaults otherwise
(kbait 31; mapping mismatches 15% of read length), iterations 1..30, stopping
earlier when MITObim reports a stationary read count. Every record in the
final *_noIUPAC.fasta is kept.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time


def verify_inputs(inputs: Path) -> dict:
    manifest = json.loads((inputs / "manifest.json").read_text())
    for relative, digest in manifest["files_sha256"].items():
        if hashlib.sha256((inputs / relative).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"input checksum mismatch: {relative}")
    return manifest


def load_jobs(inputs: Path) -> list[dict]:
    lines = (inputs / "jobs.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def job_key(job: dict) -> str:
    return f"{job['reference']}__{job['target']}__{job['depth']}X__r{job['replicate']}"


def finished_keys(results: Path) -> set[str]:
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # torn final line from an interrupted write
    return done


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records, name, chunks = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks).upper()))
            name, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    return [(n, s) for n, s in records if s]


def run_one(args: tuple) -> dict:
    inputs, work, mitobim_bin, timeout, end_iteration, job = args
    key = job_key(job)
    row = {"key": key, **{k: job[k] for k in ("reference", "target", "divergence", "depth", "replicate", "priority")},
           "host": platform.node()}
    scratch = Path(tempfile.mkdtemp(prefix="mitobim_", dir=work / "tmp"))
    started = time.time()
    try:
        reads_gz = inputs / "reads" / f"{job['target']}__{job['depth']}X__r{job['replicate']}.fastq.gz"
        with gzip.open(reads_gz, "rb") as src, (scratch / "reads.fastq").open("wb") as dst:
            shutil.copyfileobj(src, dst)
        reference = (inputs / "references" / f"{job['reference']}.fasta").read_text().split("\n", 1)[1]
        (scratch / "seed.fasta").write_text(">refseed\n" + reference)
        env = {"PATH": f"{mitobim_bin}:/usr/bin:/bin", "HOME": os.environ["HOME"], "LC_ALL": "C", "LANG": "C"}
        command = ["perl", str(Path(mitobim_bin) / "MITObim.pl"), "-start", "1", "-end", str(end_iteration),
                   "-sample", "tgt", "-ref", "refseed", "-readpool", "reads.fastq", "--quick", "seed.fasta",
                   "--mirapath", f"{mitobim_bin}/", "--clean"]
        with (scratch / "mitobim.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = None
        runtime = round(time.time() - started, 2)
        log_text = (scratch / "mitobim.log").read_text(errors="replace")
        finals = sorted(scratch.glob("iteration*/*_noIUPAC.fasta"),
                        key=lambda p: int(re.search(r"iteration(\d+)", str(p)).group(1)))
        iterations = len(list(scratch.glob("iteration*")))
        row.update(runtime_s=runtime, exit_code=code, stationary="stationary read number" in log_text,
                   last_iteration=int(re.search(r"iteration(\d+)", str(finals[-1])).group(1)) if finals else iterations)
        records = read_fasta_records(finals[-1]) if finals else []
        if code is None:
            status = "timeout"
        elif records:
            status = "completed"
        elif code == 0:
            status = "no_output"
        else:
            status = "error"
        if status != "completed":
            (work / "errors").mkdir(exist_ok=True)
            (work / "errors" / f"{key}.txt").write_text(log_text[-4000:])
            row.update(status=status, error=log_text[-300:])
            return row
        body = "".join(f">{name}\n{seq}\n" for name, seq in records)
        target_path = work / "assemblies" / f"{key}.fa.gz"
        target_path.parent.mkdir(exist_ok=True)
        temporary = target_path.with_suffix(f".tmp-{os.getpid()}")
        with gzip.GzipFile(temporary, "wb", mtime=0) as handle:
            handle.write(body.encode())
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target_path)
        row.update(status="completed", n_contigs=len(records), lengths=[len(s) for _, s in records],
                   assembly=str(target_path.relative_to(work)),
                   assembly_sha256=hashlib.sha256(body.encode()).hexdigest())
        return row
    except Exception as exc:  # infrastructure failure, recorded rather than fatal
        row.update(status="error", error=f"runner: {exc!r}"[:300], runtime_s=round(time.time() - started, 2))
        return row
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def append_row(results: Path, row: dict) -> None:
    with results.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="local results directory outside OneDrive")
    parser.add_argument("--mitobim-bin", default=os.path.expanduser("~/.local/share/mamba/envs/mitobim/bin"))
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--shard", default="0/1", help="i/n: run jobs whose index modulo n equals i")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--end-iteration", type=int, default=30)
    parser.add_argument("--subset-only", action="store_true", help="stop after the prespecified subset")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "tmp").mkdir(exist_ok=True)
    lock = (args.work / ".runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify_inputs(args.inputs)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    jobs = [j for n, j in enumerate(load_jobs(args.inputs)) if n % shard_n == shard_i]
    if args.subset_only:
        jobs = [j for j in jobs if j["priority"] == "subset"]
    results = args.work / "results.jsonl"
    done = finished_keys(results)
    pending = [j for j in jobs if job_key(j) not in done]
    (args.work / "run_info.json").write_text(json.dumps({
        "snapshot_sha256": manifest["snapshot_sha256"], "shard": args.shard, "parallel": args.parallel,
        "timeout": args.timeout, "end_iteration": args.end_iteration, "host": platform.node()}))

    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    queue = iter(pending)
    active = set()
    finished = len(jobs) - len(pending)
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        while True:
            while not stopping and len(active) < args.parallel:
                job = next(queue, None)
                if job is None:
                    break
                active.add(pool.submit(run_one, (args.inputs, args.work, args.mitobim_bin,
                                                 args.timeout, args.end_iteration, job)))
            if not active:
                break
            completed, active = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in completed:
                append_row(results, future.result())
                finished += 1
            status = {"pid": os.getpid(), "host": platform.node(), "finished": finished, "total": len(jobs),
                      "active": len(active), "stopping": stopping, "updated_unix": time.time()}
            (args.work / "status.json").write_text(json.dumps(status))
    print(json.dumps({"finished": finished, "total": len(jobs), "stopped_early": stopping}), flush=True)


if __name__ == "__main__":
    main()
