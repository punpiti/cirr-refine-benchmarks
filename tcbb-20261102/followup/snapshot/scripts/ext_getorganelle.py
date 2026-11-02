"""Thin wrapper around the GetOrganelle external tool (v1.7.7.0, bioconda),
run as a subprocess in an isolated per-job scratch directory.

Replaces MITObim as the second external de novo comparator: MITObim's
bundled mirabait (MIRA 4.0.2, and separately-tested MIRA 4.9.6) is a
statically-linked binary that hardcodes the legacy vsyscall mechanism and
segfaults on any real work under this machine's WSL2 6.6 kernel
(vsyscall=none by default) -- confirmed via dmesg, reproduces on trivial
1-read input, not fixable within the project sandbox. GetOrganelle (Jin et
al. 2020) is actively maintained and Python/SPAdes-based instead of a
legacy static Perl/C toolchain.

Quality strings: our internal baselines never touch FASTQ quality scores
(NOVOPlasty also explicitly ignores them, "Use Quality Scores = no" in its
config), so a dummy uniform string was fine there. SPAdes (which
GetOrganelle shells out to) auto-detects the Phred offset from the *range*
of quality characters seen, and a perfectly uniform string is genuinely
ambiguous between Phred+33/+64 -- this generates varied, realistic-range
Phred+33 quality scores instead so auto-detection resolves correctly.
"""

from __future__ import annotations

import glob
import os
import random
import shutil
import subprocess
import tempfile

GETORGANELLE_ENV_BIN = os.environ.get(
    "GETORGANELLE_ENV_BIN", os.path.expanduser("~/.local/share/mamba/envs/getorganelle/bin")
)
GETORGANELLE_PYTHON = f"{GETORGANELLE_ENV_BIN}/python"
GETORGANELLE_BIN = f"{GETORGANELLE_ENV_BIN}/get_organelle_from_reads.py"


class GetOrganelleFailure(Exception):
    pass


def _fastq_block(read_id: int, seq: str, rng: random.Random) -> str:
    quals = "".join(chr(33 + rng.randint(28, 40)) for _ in range(len(seq)))
    return f"@read{read_id}\n{seq}\n+\n{quals}\n"


def run_getorganelle(
    seed_seq: str,
    reads: list[str],
    quality_seed: int,
    timeout: int = 600,
    rounds: int = 10,
    kmers: str = "21,45,65,85,105",
) -> str:
    """Run GetOrganelle in an isolated temp dir; return the longest
    path_sequence.fasta scaffold produced (GetOrganelle does not guarantee
    full circularization, especially at low depth -- picking the longest
    available scaffold is the same "best effort" a user would make
    manually, matching ext_novoplasty.py's longest-contig convention)."""
    rng = random.Random(quality_seed)
    workdir = tempfile.mkdtemp(prefix="getorganelle_")
    try:
        reads_path = os.path.join(workdir, "reads.fastq")
        with open(reads_path, "w") as fh:
            for i, r in enumerate(reads):
                fh.write(_fastq_block(i, r, rng))
        seed_path = os.path.join(workdir, "seed_ref.fasta")
        with open(seed_path, "w") as fh:
            fh.write(f">seed_ref\n{seed_seq}\n")

        out_dir = os.path.join(workdir, "out")
        env = dict(os.environ)
        env["PATH"] = f"{GETORGANELLE_ENV_BIN}:{env.get('PATH', '')}"
        try:
            proc = subprocess.run(
                [
                    GETORGANELLE_PYTHON, GETORGANELLE_BIN,
                    "-u", reads_path, "-s", seed_path, "-F", "animal_mt",
                    "-o", out_dir, "-R", str(rounds), "-k", kmers, "-t", "1",
                ],
                cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise GetOrganelleFailure(f"timed out after {timeout}s") from exc

        scaffold = _best_scaffold(out_dir)
        if scaffold is None:
            raise GetOrganelleFailure(
                f"no scaffold produced; stdout tail:\n{proc.stdout[-1500:]}\n"
                f"stderr tail:\n{proc.stderr[-500:]}"
            )
        return scaffold
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _best_scaffold(out_dir: str) -> str | None:
    best = None
    for path in glob.glob(os.path.join(out_dir, "*path_sequence.fasta")):
        for seq in _iter_fasta_seqs(path):
            if best is None or len(seq) > len(best):
                best = seq
    return best


def _iter_fasta_seqs(path: str):
    seq: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if seq:
                    yield "".join(seq).upper()
                    seq = []
            else:
                seq.append(line)
    if seq:
        yield "".join(seq).upper()
