"""Shared I/O, read simulation, alignment, and evaluation helpers for the
ASEM/ECTI-CON baseline reimplementations.

Tooling notes (deviations from the original papers, documented for
reproducibility):
- Alignment (E-step): the original papers used scikit-bio's
  StripedSmithWaterman with default parameters (based on the Striped
  Smith-Waterman algorithm, Farrar 2007). That class no longer exists in the
  installed scikit-bio (0.7.x). We use `parasail`'s `sw_trace_striped_16`
  instead -- the same Farrar 2007 striped SW algorithm the original paper
  cited, just a different (actively maintained) binding of it. It is also
  ~15x faster than skbio's modern general-purpose `pair_align_nucl(mode=
  "local")`, which we benchmarked and rejected for that reason (measured on
  this project's actual read/reference sizes: 15.0s vs 1.0s for 800 reads
  against a ~16.5kb reference).
- Read simulation: the original papers used Grinder 0.5.4 for shotgun reads.
  Grinder is no longer packaged in bioconda. This module implements a
  minimal uniform-random shotgun simulator (random start position, fixed
  read length, sampled until the target depth is reached) reproducing
  Grinder's default shotgun behavior for a single-genome input.
- Evaluation: the original papers used EMBOSS `needle` (Needleman-Wunsch).
  We tried it (invoked directly via the `_needle` binary, since the `needle`
  wrapper script shipped by the bioconda package is broken) but its
  full-quadratic-space implementation took 30-90s and ~4GB RAM per pair for
  ~16-17kb mtDNA sequences, which is not affordable across the grid's
  thousands of evaluation calls. We use skbio's `pair_align_nucl(mode=
  "global")` instead, retaining its default `free_ends=True`. This runs the
  global dynamic-programming kernel with unpenalized terminal gaps (an overlap
  or semi-global alignment), and was ~10-30x faster in practice. It is not a
  strict end-to-end reproduction of EMBOSS `needle`.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

import parasail
from skbio.alignment import pair_align_nucl
from skbio.sequence import DNA

_SW_MATRIX = parasail.matrix_create("ACGT", 2, -3)  # match=2, mismatch=-3: same defaults as before
_CIRCULAR_ANCHOR_MATRIX = parasail.matrix_create("ACGTN", 2, -3)
_SW_GAP_OPEN = 5
_SW_GAP_EXTEND = 2


def read_fasta(path: str) -> tuple[str, str]:
    """Read a single-record FASTA file. Returns (header, sequence)."""
    header = None
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
            elif line:
                parts.append(line)
    if header is None:
        raise ValueError(f"no FASTA record found in {path}")
    return header, "".join(parts).upper()


def write_fasta(path: str, header: str, sequence: str) -> None:
    """Write one FASTA record with deterministic 80-base wrapping."""
    with open(path, "w") as handle:
        handle.write(f">{header}\n")
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start : start + 80] + "\n")


def stable_seed(*parts: object) -> int:
    """Deterministic seed derived from `parts`, stable across processes and
    machines. Do NOT use Python's built-in hash() for this: string hashing
    is randomized per-process by default (PYTHONHASHSEED), so hash((a, b))
    gives a different value every time the interpreter starts -- it is only
    self-consistent within one process (and its fork()'d children, which is
    why this bug wasn't obvious inside a single grid run using
    ProcessPoolExecutor, only across separate runs of the script)."""
    key = "|".join(str(p) for p in parts).encode()
    return zlib.crc32(key)


def simulate_shotgun_reads(
    sequence: str,
    depth: float,
    read_length: int = 150,
    seed: int | None = None,
) -> list[str]:
    """Uniform-random single-genome shotgun read simulation.

    Samples reads of `read_length` at uniformly random start positions
    (linear, non-circular) until the total simulated bases reach
    `depth * len(sequence)`. This reproduces Grinder's default shotgun mode
    behavior for a single reference (see module docstring).
    """
    rng = random.Random(seed)
    n = len(sequence)
    target_bases = int(round(depth * n))
    reads: list[str] = []
    total = 0
    max_start = max(n - read_length, 0)
    while total < target_bases:
        start = rng.randint(0, max_start)
        read = sequence[start : start + read_length]
        reads.append(read)
        total += len(read)
    return reads


def simulate_shotgun_reads_variable_length(
    sequence: str,
    depth: float,
    seed: int | None = None,
    max_length: int = 150,
    min_length: int = 36,
    p_full_length: float = 0.7,
) -> list[str]:
    """Shotgun read simulation with a realistic length *distribution*
    instead of a single fixed length -- models real post-sequencing
    quality/adapter trimming (e.g. Trimmomatic-style pipelines), where most
    reads survive at the sequencer's native length but a minority get
    trimmed shorter (never longer). Each read independently: with
    probability `p_full_length` keeps `max_length`; otherwise its length is
    drawn uniformly from [min_length, max_length). `min_length=36` matches
    a common Trimmomatic MINLEN default (reads trimmed shorter than this are
    discarded entirely in real pipelines, so nothing shorter is generated
    here either). Read start positions are sampled the same way as
    `simulate_shotgun_reads`; only the per-read length differs.
    """
    rng = random.Random(seed)
    n = len(sequence)
    target_bases = int(round(depth * n))
    reads: list[str] = []
    total = 0
    while total < target_bases:
        read_length = max_length if rng.random() < p_full_length else rng.randint(min_length, max_length - 1)
        max_start = max(n - read_length, 0)
        start = rng.randint(0, max_start)
        read = sequence[start : start + read_length]
        reads.append(read)
        total += len(read)
    return reads


@dataclass
class LocalAlignment:
    score: float
    ref_start: int  # 0-indexed offset into the reference where alignment begins
    read_start: int
    ref_aligned: str  # gapped, aligned-region only
    read_aligned: str


def align_local(read: str, reference: str) -> LocalAlignment | None:
    """Local Striped Smith-Waterman alignment of `read` onto `reference`
    (parasail, Farrar 2007 -- see module docstring)."""
    res = parasail.sw_trace_striped_16(read, reference, _SW_GAP_OPEN, _SW_GAP_EXTEND, _SW_MATRIX)
    if res.score <= 0:
        return None
    tb = res.get_traceback("-")
    ref_aligned = tb.ref
    read_aligned = tb.query
    ref_ungapped_len = len(ref_aligned) - ref_aligned.count("-")
    read_ungapped_len = len(read_aligned) - read_aligned.count("-")
    ref_start = res.end_ref - ref_ungapped_len + 1
    read_start = res.end_query - read_ungapped_len + 1
    return LocalAlignment(
        score=float(res.score),
        ref_start=int(ref_start),
        read_start=int(read_start),
        ref_aligned=ref_aligned,
        read_aligned=read_aligned,
    )


def placement_observations(
    aln: LocalAlignment,
) -> tuple[dict[int, str], dict[int, str], int]:
    """Return reference-column calls, insertion-slot sequences, and ref end.

    Insertion slot ``p`` is immediately before reference coordinate ``p``;
    slot 0 precedes the reference and slot ``len(reference)`` follows it.
    Consecutive read bases aligned to reference gaps are kept in their aligned
    order so the structural update can construct combined-alignment columns.
    """
    calls: dict[int, str] = {}
    insertions: dict[int, list[str]] = {}
    ref_pos = aln.ref_start
    for ref_ch, read_ch in zip(aln.ref_aligned, aln.read_aligned):
        if ref_ch == "-":
            if read_ch != "-":
                insertions.setdefault(ref_pos, []).append(read_ch)
            continue
        calls[ref_pos] = read_ch  # read_ch may be "-" (deletion vote)
        ref_pos += 1
    return calls, {slot: "".join(seq) for slot, seq in insertions.items()}, ref_pos


def placement_base_calls(aln: LocalAlignment) -> dict[int, str]:
    """Map an alignment's calls onto existing reference coordinates only."""
    calls, _, _ = placement_observations(aln)
    return calls


def _evaluate_linear_theta_vs_target(theta: str, target: str) -> dict:
    """Evaluate one fixed orientation and linear origin."""
    result = pair_align_nucl(DNA(theta), DNA(target), mode="global")
    path = result.paths[0]
    aligned_theta, aligned_target = path.to_aligned([DNA(theta), DNA(target)])
    aligned_theta, aligned_target = str(aligned_theta), str(aligned_target)

    c_theta = w_theta = u_t = identical = aligned_len = 0
    for a_ch, b_ch in zip(aligned_theta, aligned_target):
        aligned_len += 1
        if a_ch == "-":
            u_t += 1
        elif b_ch == "-":
            pass
        elif a_ch == b_ch:
            c_theta += 1
            identical += 1
        else:
            w_theta += 1

    recall = c_theta / (c_theta + u_t) if (c_theta + u_t) else 0.0
    precision = c_theta / (c_theta + w_theta) if (c_theta + w_theta) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "identity_pct": 100.0 * identical / aligned_len if aligned_len else 0.0,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "c_theta": c_theta,
        "w_theta": w_theta,
        "u_t": u_t,
    }


def _anchor_transform(theta: str, reference: str) -> tuple[int, int]:
    """Return the inferred theta-to-reference offset and alignment score.

    Subtracting the aligned theta start accounts for incomplete or noisy
    leading sequence.  The 32-bit kernel avoids score saturation for complete
    mitochondrial genomes.
    """
    if not theta or not reference or set(theta + reference) - set("ACGTN"):
        return 0, 0
    result = parasail.sw_trace_striped_32(
        theta,
        reference,
        _SW_GAP_OPEN,
        _SW_GAP_EXTEND,
        _CIRCULAR_ANCHOR_MATRIX,
    )
    if result.score <= 0:
        return 0, 0
    traceback = result.get_traceback("-")
    ref_length = len(traceback.ref) - traceback.ref.count("-")
    theta_length = len(traceback.query) - traceback.query.count("-")
    ref_start = result.end_ref - ref_length + 1
    theta_start = result.end_query - theta_length + 1
    return ref_start - theta_start, int(result.score)


def _rotation_offset_and_score(theta: str, target: str) -> tuple[int, int]:
    """Infer the circular target origin corresponding to theta coordinate 0."""
    offset, score = _anchor_transform(theta, target + target)
    return offset % len(target) if target else 0, score


def evaluate_theta_vs_target(
    theta: str,
    target: str,
    *,
    normalize_strand: bool = False,
    normalize_circular_origin: bool = False,
) -> dict:
    """Align estimated Theta to the target and compute the published metrics.

    The default preserves the historical fixed-orientation, fixed-origin
    evaluator.  External organelle assemblers must set both normalization
    flags because a biologically equivalent circular assembly may be emitted
    on either strand and from any circular origin. Candidate transforms are
    chosen by local-alignment score before reporting metrics are calculated;
    the metric definitions remain unchanged.

    Metrics follow the original papers:

        c_theta = correctly assembled positions of Theta (matches)
        w_theta = incorrectly assembled positions of Theta (mismatches)
        u_t     = unassembled positions of target (gaps in Theta's alignment,
                  i.e. target bases Theta failed to reproduce)
        recall    = c_theta / (c_theta + u_t)
        precision = c_theta / (c_theta + w_theta)
        F1        = harmonic mean of precision and recall
        identity  = percent identical positions over the aligned length
    """
    if not theta or not target:
        return _evaluate_linear_theta_vs_target(theta, target)

    orientations = [theta]
    if normalize_strand:
        reverse = str(DNA(theta).reverse_complement())
        if reverse != theta:
            orientations.append(reverse)

    candidates = []
    for oriented_theta in orientations:
        if normalize_circular_origin:
            offset, transform_score = _rotation_offset_and_score(oriented_theta, target)
        elif normalize_strand:
            _, transform_score = _anchor_transform(oriented_theta, target)
            offset = 0
        else:
            transform_score = 0
            offset = 0
        rotated_target = target[offset:] + target[:offset]
        metrics = _evaluate_linear_theta_vs_target(oriented_theta, rotated_target)
        candidates.append((transform_score, metrics))

    return max(candidates, key=lambda candidate: candidate[0])[1]
