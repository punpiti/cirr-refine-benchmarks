"""Baseline 2: OJEMB ASEM, adds recursive partial alignment
(archive/manuscripts/ojemb/original_working_tree/genome_ojemb.tex,
Section "A Case of Divergent Reference Sequence" / sec:recursive_alignment,
Algorithm "VarEstimation").

Everything else (EM loop, match-state-only consensus, stop condition) is
identical to Baseline 1 and lives in asem_core.py; this file only adds the
recursive alignment rule on top of Baseline 1's single-shot one.

What recursive partial alignment does, and why: a read spanning a region
where the reference is locally very divergent tends to get clipped short by
Smith-Waterman rather than pushed through with a series of mismatches/gaps,
because SW stops extending once the running score would go negative. The
part of the read past that clip point is thrown away by Baseline 1 even
though it may still carry a usable, separately-alignable signal (e.g. if
that clipped-off part actually matches a DIFFERENT nearby region well, which
happens when the divergence is a genuinely different sub-sequence rather
than scattered point mutations). OJEMB's fix: after the primary alignment,
take whatever the read has left over past either end of what got aligned
and -- if it's long enough to be worth trying (>= a length threshold ℓ) --
recursively align that leftover fragment against theta too, exactly like a
brand new read. The paper's pseudocode (Algorithm alg:alignread) writes this
as a single "UnalignedPart"; a read locally aligned via Smith-Waterman can be
clipped on both ends at once (a prefix before the aligned region and a
suffix after it), so this implementation recurses on each flank
independently -- an explicit design choice the paper's pseudocode does not
spell out.

Additional unspecified-in-the-paper hyperparameter beyond Baseline 1's
tau/w:
- min_leftover_len (the paper's ℓ): minimum length, in bases, for a
  leftover flank to be worth recursively re-aligning. Default 30.
"""

from __future__ import annotations

import functools

from asem_core import AsemResult, run_asem_em_loop
from common import LocalAlignment, align_local

MATCH_SCORE = 2.0  # must match common.align_local's parasail matrix


def _align_read_ojemb(read: str, theta: str, tau: float, min_leftover_len: int) -> list[LocalAlignment]:
    accepted: list[LocalAlignment] = []
    _recursive_align(read, theta, tau, min_leftover_len, accepted)
    return accepted


def _recursive_align(
    read: str, theta: str, tau: float, min_leftover_len: int, accepted: list[LocalAlignment]
) -> None:
    if not read:
        return
    aln = align_local(read, theta)
    if aln is None:
        return

    if aln.score >= tau * MATCH_SCORE * len(read):
        accepted.append(aln)

    read_ungapped_len = len(aln.read_aligned) - aln.read_aligned.count("-")
    prefix = read[: aln.read_start]
    suffix = read[aln.read_start + read_ungapped_len :]

    if len(prefix) >= min_leftover_len:
        _recursive_align(prefix, theta, tau, min_leftover_len, accepted)
    if len(suffix) >= min_leftover_len:
        _recursive_align(suffix, theta, tau, min_leftover_len, accepted)


def run_asem(
    theta_init: str,
    reads: list[str],
    tau: float = 0.5,
    w: float = 0.1,
    min_leftover_len: int = 30,
    max_iterations: int = 6,
    n_workers: int = 1,
) -> AsemResult:
    align_read_fn = functools.partial(_align_read_ojemb, tau=tau, min_leftover_len=min_leftover_len)
    return run_asem_em_loop(
        theta_init=theta_init,
        reads=reads,
        align_read_fn=align_read_fn,
        w=w,
        max_iterations=max_iterations,
        n_workers=n_workers,
    )
