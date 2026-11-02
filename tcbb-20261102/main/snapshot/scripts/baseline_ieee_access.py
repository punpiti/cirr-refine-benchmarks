"""Baseline 1: IEEE Access ASEM (EM / match-state-only Profile HMM), no
recursive partial alignment.

Faithful reimplementation of the algorithm described in
archive/manuscripts/ieee_access/genome_ieeeaccess_submitted.pdf and
archive/manuscripts/ojemb/original_working_tree/genome_ojemb.tex
(sections up to, but excluding, sec:recursive_alignment). See
asem_core.py for the EM loop shared with Baseline 2 (OJEMB); this file
only supplies the per-read alignment rule.

Unspecified-in-the-paper hyperparameters (documented here since the papers
never state numeric values for them):
- tau: alignment acceptance threshold, interpreted as
  (alignment score) / (match_score * len(read)), i.e. fraction of the best
  possible score for a read of that length. Default 0.5.
- w: minimum-confidence weight in the emission-probability rule
  (d_k > w * D). Default 0.1 (a column needs at least 10% of the
  mean per-column depth to be treated as well-supported).

These are Layer-1 engineering hyperparameters per
algorithm_sketch_iterative_refinement.md and are not tuned here; Phase 1's
goal is a working, faithful reproduction, not optimal parameters.
"""

from __future__ import annotations

import functools

from asem_core import AsemResult, run_asem_em_loop
from common import LocalAlignment, align_local

MATCH_SCORE = 2.0  # must match common.align_local's parasail matrix


def _align_read_ieee_access(read: str, theta: str, tau: float) -> list[LocalAlignment]:
    """One local alignment attempt per read; accepted or discarded whole,
    no recursion on the unaligned remainder (that is OJEMB's addition)."""
    aln = align_local(read, theta)
    if aln is not None and aln.score >= tau * MATCH_SCORE * len(read):
        return [aln]
    return []


def run_asem(
    theta_init: str,
    reads: list[str],
    tau: float = 0.5,
    w: float = 0.1,
    max_iterations: int = 6,
    n_workers: int = 1,
) -> AsemResult:
    align_read_fn = functools.partial(_align_read_ieee_access, tau=tau)
    return run_asem_em_loop(
        theta_init=theta_init,
        reads=reads,
        align_read_fn=align_read_fn,
        w=w,
        max_iterations=max_iterations,
        n_workers=n_workers,
    )
