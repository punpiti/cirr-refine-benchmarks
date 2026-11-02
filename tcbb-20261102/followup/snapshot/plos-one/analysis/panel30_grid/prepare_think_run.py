"""Export committed code and local input copies for the authorized think rerun."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def export(repo: Path, paths: list[str], destination: Path) -> None:
    blob = subprocess.check_output(["git", "-C", str(repo), "archive", "HEAD", *paths])
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        members = [m for m in archive.getmembers() if not {"results", "__pycache__"} & set(Path(m.name).parts)]
        archive.extractall(destination, members=members, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    bench = project / "asem-refine-benchmarks"
    package = project / "asem-refine"
    if args.root.exists():
        raise SystemExit("run root already exists; inspect it before preparing another snapshot")
    for repo in (bench, package):
        if subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "scripts", "src", "tests"]):
            raise SystemExit(f"uncommitted implementation changes: {repo.name}")
    snap = args.root / "snapshot"
    snap.mkdir(parents=True)
    export(bench, ["scripts"], snap)
    export(package, ["src", "tests", "pyproject.toml"], snap / "package")
    export(project, [f"plos-one/analysis/{p}" for p in (
        "panel30_grid", "read_length_all_methods", "real_reads_all_methods", "runtime_benchmark")]
        + ["plos-one/analysis/evaluator_n_fix/nsafe.py"], snap)
    # Data are deliberately not tracked in the public benchmark repository.
    # Copy the input files, rather than importing a mutable OneDrive dataset.
    for directory in ("ieee2021_mtdna_panel", "phase1_reproduction", "refs"):
        shutil.copytree(bench / "data" / directory, snap / "data" / directory)
    shutil.copy2(bench / "data/NA07000_chrM.bam", snap / "data/NA07000_chrM.bam")
    inputs = {str(p.relative_to(snap)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (snap / "data").rglob("*") if p.is_file()}
    sources = {str(p.relative_to(snap)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in snap.rglob("*.py")}
    env_root = Path.home() / ".local/share/mamba/envs"
    environments = {}
    for name in ("genome", "mia-assembler", "pilon-benchmark", "getorganelle"):
        environments[name] = []
        for p in sorted((env_root / name / "conda-meta").glob("*.json")):
            record = json.loads(p.read_text())
            environments[name].append({k: record.get(k) for k in ("name", "version", "build")})
    metadata = {"project_commit": head(project), "bench_commit": head(bench),
                "package_commit": head(package), "taxonomy": "original-panel",
                "input_sha256": inputs, "source_sha256": sources, "environments": environments}
    (args.root / "snapshot.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({k: metadata[k] for k in ("project_commit", "bench_commit", "package_commit")}))
    print(f"Prepared {len(inputs)} inputs and {len(sources)} Python sources at {args.root}")


if __name__ == "__main__":
    main()
