"""Run one isolated Pilon v1.24 polishing job on single-end reads."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ENV_BIN = Path(os.path.expanduser("~/.local/share/mamba/envs/pilon-benchmark/bin"))
BWA = ENV_BIN / "bwa"
SAMTOOLS = ENV_BIN / "samtools"
JAVA = ENV_BIN / "java"
PILON_JAR = ENV_BIN.parent / "share" / "pilon-1.24-0" / "pilon.jar"


class PilonFailure(RuntimeError):
    pass


def _run(command: list[str], *, timeout: int, stdout=None) -> None:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=stdout is None,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PilonFailure(f"timed out after {timeout}s") from exc
    if completed.returncode:
        error = completed.stderr if isinstance(completed.stderr, str) else completed.stderr.decode(errors="replace")
        raise PilonFailure(error[-500:].strip() or f"exit status {completed.returncode}")


def run_pilon(reference: str, reads: list[str], *, timeout: int = 180) -> str:
    with tempfile.TemporaryDirectory(prefix="pilon_") as directory:
        work = Path(directory)
        reference_path = work / "reference.fasta"
        reads_path = work / "reads.fasta"
        sam_path = work / "aligned.sam"
        bam_path = work / "aligned.bam"
        reference_path.write_text(">reference\n" + reference + "\n")
        reads_path.write_text("".join(f">read_{index}\n{read}\n" for index, read in enumerate(reads)))

        _run([BWA, "index", reference_path], timeout=timeout)
        with sam_path.open("wb") as sam_output:
            _run([BWA, "mem", "-t", "1", reference_path, reads_path], timeout=timeout, stdout=sam_output)
        _run([SAMTOOLS, "sort", "-o", bam_path, sam_path], timeout=timeout)
        _run([SAMTOOLS, "index", bam_path], timeout=timeout)
        _run(
            [
                JAVA, "-Xms512m", "-Xmx1g", "-jar", PILON_JAR,
                "--genome", reference_path,
                "--unpaired", bam_path,
                "--output", "pilon",
                "--outdir", work,
                "--fix", "all",
            ],
            timeout=timeout,
        )
        output = work / "pilon.fasta"
        if not output.exists():
            raise PilonFailure("Pilon produced no FASTA output")
        return "".join(
            line.strip() for line in output.read_text().splitlines() if not line.startswith(">")
        ).upper()
