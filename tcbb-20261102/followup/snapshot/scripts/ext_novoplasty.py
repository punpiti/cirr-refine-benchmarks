"""Thin wrapper around the NOVOPlasty external tool (v4.3.5, bioconda),
run as a subprocess in an isolated per-job scratch directory so parallel
grid jobs never collide on NOVOPlasty's CWD-relative output file names.

Config field values mirror a hand-built config.txt that was smoke-tested
manually first -- NOVOPlasty ships no config template of its own, so this
was reconstructed from the Perl script's own field-parsing regexes.

Coverage floor (verified against the tool's own paper, not just observed):
NOVOPlasty refuses to run below ~10X ("COVERAGE IS TOO LOW, SHOULD BE MORE
THAN 10X") and even just above that floor (8-10X here) only produces a tiny
partial fragment (precision=1.0 on what it assembles, but recall~0.02 --
most of the genome unassembled) instead of a complete contig. This is not
an artifact of our harness: Dierckxsens et al. 2017 (the NOVOPlasty paper,
`bmc/references/pdfs/Dierckxsens2017NOVOPlasty.pdf`, p.8) states verbatim
"It is recommended to have sufficient coverage (30X for the organelle
genome) ... Incomplete assemblies caused by low coverage regions ... could
be resolved by using higher coverage." Our grid's 1-8X range (the baselines'
native range, for read-for-read comparability) is therefore expected to
mostly fail outright or produce incomplete assemblies -- that is the
intended "coverage below usable threshold" comparison point, not a bug.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

NOVOPLASTY_PL = os.environ.get(
    "NOVOPLASTY_PL", os.path.expanduser("~/.local/share/mamba/envs/genome/bin/NOVOPlasty4.3.5.pl")
)

# IUPAC nucleotide codes plus gap ('-') and skbio's accepted missing-data
# placeholder ('.'); matches skbio.sequence.DNA's own validation alphabet.
_VALID_SEQ_CHARS = set("ACGTNRYSWKMBDHV-.")

_CONFIG_TEMPLATE = """Project:
-----------------------
Project name          = {project}
Type                  = mito
Genome Range           = 12000-22000
K-mer                  = 33
Max memory             = 4
Extended log           = 0
Save assembled reads   = no
Seed Input             = seed_ref.fasta
Extend seed directly   = no
Reference sequence     = seed_ref.fasta
Variance detection     =
Chloroplast sequence   =

Dataset 1:
-----------------------
Read Length             = 150
Insert size             = 300
Platform                = illumina
Single/Paired           = SE
Combined reads          = reads.fastq
Forward reads           =
Reverse reads           =
Store Hash              =

Heteroplasmy:
-----------------------
MAF                     =
HP exclude list         =
PCR-free                =

Optional:
-----------------------
Insert size auto        = yes
Use Quality Scores      = no
Reduce ambigious N's    =
Output path             =
"""


class NovoplastyFailure(Exception):
    pass


def run_novoplasty(seed_seq: str, reads: list[str], project: str, timeout: int = 1800) -> str:
    """Run NOVOPlasty in an isolated temp dir; return the longest contig it
    produced (NOVOPlasty does not guarantee a single circularized contig on
    divergent references -- picking the longest is the same "best effort"
    choice a user would make manually)."""
    workdir = tempfile.mkdtemp(prefix="novoplasty_")
    try:
        with open(os.path.join(workdir, "seed_ref.fasta"), "w") as fh:
            fh.write(f">seed_ref\n{seed_seq}\n")
        with open(os.path.join(workdir, "reads.fastq"), "w") as fh:
            for i, r in enumerate(reads):
                fh.write(f"@read{i}\n{r}\n+\n{'I' * len(r)}\n")
        with open(os.path.join(workdir, "config.txt"), "w") as fh:
            fh.write(_CONFIG_TEMPLATE.format(project=project))

        try:
            proc = subprocess.run(
                ["perl", NOVOPLASTY_PL, "-c", "config.txt"],
                cwd=workdir, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise NovoplastyFailure(f"timed out after {timeout}s for {project}") from exc

        contig = _read_best_contig(workdir, project)
        if contig is None:
            raise NovoplastyFailure(
                f"no contig produced for {project}; stdout tail:\n{proc.stdout[-1500:]}"
            )
        invalid = set(contig) - _VALID_SEQ_CHARS
        if invalid:
            # NOVOPlasty uses '*' (and related markers) internally while
            # resolving contig overlaps/merges (see NOVOPlasty4.3.5.pl
            # around its tr/\*// and (\w\*)+ handling); normally cleaned up
            # before the final FASTA is written, but on assemblies it can't
            # fully resolve -- expected on the divergent pairs this
            # benchmark targets -- the marker leaks into Merged_contigs_/
            # Option_ output instead of pure sequence. That is NOVOPlasty
            # reporting its own unresolved-merge state, not a parsing bug
            # here, so fail this job explicitly rather than handing
            # downstream scoring a string that looks like DNA but isn't.
            raise NovoplastyFailure(
                f"malformed contig for {project}: unresolved-merge marker(s) "
                f"{sorted(invalid)} in NOVOPlasty output (len={len(contig)})"
            )
        return contig
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _read_best_contig(workdir: str, project: str) -> str | None:
    candidates = [
        f"Contigs_1_{project}.fasta",
        f"Circularized_assembly_1_{project}.fasta",
        f"Merged_contigs_{project}.fasta",
        f"Option_1_{project}.fasta",
    ]
    best = None
    for name in candidates:
        path = os.path.join(workdir, name)
        if not os.path.exists(path):
            continue
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
