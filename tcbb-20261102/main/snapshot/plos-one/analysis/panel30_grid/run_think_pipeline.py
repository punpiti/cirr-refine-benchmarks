"""Run controlled timing first, then affected auxiliary jobs and the full panel.

Execute this from the exported snapshot. Every phase is resumable and writes
separate outputs; historical results and manuscript displays are preserved.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resume_io import checkpoint, sync_directory

INTERNAL = "ASEM,ASEM_Hybrid,ASEM_recursive"


def validate(path: Path, expected: int, keys: tuple[str, ...], allowed: set[str]) -> dict:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == expected, (path, len(rows), expected)
    assert len({tuple(r[k] for k in keys) for r in rows}) == expected, f"duplicate keys: {path}"
    assert set(r["status"] for r in rows) <= allowed, f"unexpected status: {path}"
    assert all(r["status"] != "completed" or r.get("f1", r.get("truth_primary_f1", "")) != "" for r in rows)
    return {"rows": len(rows), "statuses": dict(collections.Counter(r["status"] for r in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / ".pipeline.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another pipeline is running; inspect status instead of starting a duplicate")
    snap = root / "snapshot"
    metadata = json.loads((root / "snapshot.json").read_text())
    for name, sha in {**metadata["input_sha256"], **metadata["source_sha256"]}.items():
        assert hashlib.sha256((snap / name).read_bytes()).hexdigest() == sha, f"snapshot changed: {name}"
    analysis = snap / "plos-one/analysis"
    env = {**os.environ, "BENCH_SCRIPTS": str(snap / "scripts"),
           "BENCH_COMMIT": metadata["bench_commit"], "PANEL30_DATA_DIR": str(snap / "data/ieee2021_mtdna_panel"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    phases = [
        ("runtime", "runtime_benchmark/run_runtime_benchmark.py", ["--methods", INTERNAL], 108, "runtime_results.csv"),
        ("read_length", "read_length_all_methods/run_read_length_all_methods.py", ["--methods", INTERNAL, "--parallel", "8"], 189, "results.csv"),
        ("real_reads", "real_reads_all_methods/run_real_reads_all_methods.py", ["--methods", INTERNAL, "--parallel", "8"], 144, "results.csv"),
        ("panel30", "panel30_grid/run_panel30_grid.py", ["--parallel", "14"], 146160, "results.csv"),
    ]
    state = {"snapshot": str(snap), "completed_phases": {}}
    state_path = root / "pipeline_status.json"
    def status(phase: str, outcome: str) -> None:
        state.update(phase=phase, status=outcome, updated_utc=datetime.now(timezone.utc).isoformat())
        temp = state_path.with_suffix(".tmp")
        with temp.open("w") as handle:
            handle.write(json.dumps(state, indent=2))
            checkpoint(handle)
        temp.replace(state_path)
        sync_directory(root)
        print(f"{state['updated_utc']} {phase}: {outcome}", flush=True)
    for name, script, flags, expected, filename in phases:
        output = root / name
        output.mkdir(exist_ok=True)
        status(name, "running")
        phase_env = {**env, "CIRR_RESULTS_DIR": str(output), "PANEL30_RESULTS_DIR": str(output)}
        with (root / f"{name}.log").open("a") as log:
            proc = subprocess.run([sys.executable, "-I", str(analysis / script), *flags],
                                  env=phase_env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            status(name, f"stopped: exit {proc.returncode}; inspect {name}.log")
            raise SystemExit(proc.returncode)
        keys = ("method", "reference", "depth", "replicate") if name == "real_reads" else (
            "method", "reference", "target", "length_condition" if name == "read_length" else "depth", "replicate")
        try:
            accepted = validate(output / filename, expected, keys,
                                {"completed", "no_output", "error", "timeout"} if name == "panel30" else {"completed"})
            if name == "panel30":
                from panel30_species import ordered_pairs
                with (output / filename).open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                methods = {"ASEM", "ASEM_Hybrid", "SRSC", "Pilon", "MIA", "NOVOPlasty", "GetOrganelle"}
                wanted = {(m, r, t, str(d), str(rep)) for m in methods for r, t in ordered_pairs()
                          for d in range(1, 9) for rep in range(3)}
                assert {tuple(r[k] for k in keys) for r in rows} == wanted, "unexpected grid keys"
                assert all(r["status"] != "error" or "required executable not found" not in r["error"] for r in rows)
            state["completed_phases"][name] = accepted
            (output / "acceptance.json").write_text(json.dumps(accepted, indent=2))
        except Exception as exc:
            status(name, f"validation stopped: {exc}")
            raise
        status(name, "complete")
    status("all", "compute complete; evidence review and manuscript rebuild pending")


if __name__ == "__main__":
    main()
