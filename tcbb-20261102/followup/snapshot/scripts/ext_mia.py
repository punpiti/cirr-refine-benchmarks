"""Wrapper for MIA (Mapping Iterative Assembler) v1.0.

MIA performs iterative reference-assisted consensus calling and was used for
ancient mitochondrial reconstruction. Each call runs in an isolated scratch
directory, enables MIA's circular-reference and distant-reference modes, and
returns the final consensus produced at convergence.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile


MIA_ENV_BIN = os.environ.get(
    "MIA_ENV_BIN", os.path.expanduser("~/.local/share/mamba/envs/mia-assembler/bin")
)
MIA_BIN = os.environ.get("MIA_BIN", os.path.join(MIA_ENV_BIN, "mia"))
MA_BIN = os.environ.get("MA_BIN", os.path.join(MIA_ENV_BIN, "ma"))


class MiaFailure(Exception):
    pass


def _write_fasta(path: str, records: list[tuple[str, str]]) -> None:
    with open(path, "w") as fh:
        for name, sequence in records:
            fh.write(f">{name}\n{sequence}\n")


def _parse_single_fasta(text: str) -> str:
    sequence = "".join(
        line.strip() for line in text.splitlines() if line and not line.startswith(">")
    ).upper()
    if not sequence:
        raise MiaFailure("MIA conversion produced an empty consensus")
    return sequence


def run_mia(reference: str, reads: list[str], timeout: int = 300) -> str:
    """Run MIA to convergence and return its final FASTA consensus.

    ``-c`` models the mitochondrial reference as circular. ``-D`` keeps
    low-scoring reads during iterative assembly when the starting reference is
    distantly related. ``-F`` writes only the final MALN assembly.
    """
    for executable in (MIA_BIN, MA_BIN):
        if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
            raise MiaFailure(f"required executable not found: {executable}")

    workdir = tempfile.mkdtemp(prefix="mia_")
    try:
        reference_path = os.path.join(workdir, "reference.fasta")
        reads_path = os.path.join(workdir, "reads.fasta")
        output_root = os.path.join(workdir, "assembly")
        _write_fasta(reference_path, [("reference", reference)])
        _write_fasta(reads_path, [(f"read{i}", read) for i, read in enumerate(reads)])

        try:
            proc = subprocess.run(
                [
                    MIA_BIN,
                    "-r", reference_path,
                    "-f", reads_path,
                    "-m", output_root,
                    "-c",
                    "-D",
                    "-F",
                ],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise MiaFailure(f"timed out after {timeout}s") from exc
        if proc.returncode != 0:
            raise MiaFailure(
                f"MIA exited {proc.returncode}; stdout tail:\n{proc.stdout[-1000:]}\n"
                f"stderr tail:\n{proc.stderr[-1000:]}"
            )

        outputs = glob.glob(f"{output_root}.*")
        if not outputs:
            raise MiaFailure("MIA produced no final MALN assembly")
        try:
            final_maln = max(outputs, key=lambda path: int(path.rsplit(".", 1)[1]))
        except ValueError as exc:
            raise MiaFailure(f"unexpected MIA output names: {outputs}") from exc

        converted = subprocess.run(
            [MA_BIN, "-M", final_maln, "-f", "5", "-I", "MIA_consensus"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if converted.returncode != 0:
            raise MiaFailure(
                f"ma exited {converted.returncode}; stderr tail:\n{converted.stderr[-1000:]}"
            )
        return _parse_single_fasta(converted.stdout)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
