"""Testable helper implementation for boundary-recruited ASEM-Hybrid.

This module implements the boundary-recruited variant independently for
focused tests.  When a zero-depth island
is detected, the assembly pool contains the unplaced reads plus only placed
reads whose reference alignment overlaps a short flank immediately beside an
island.  The placed reads supply a coordinate-linked bridge at each boundary;
they are *not* drawn from the whole reference.

No read-end labels are inferred from sequence.  Membership in a left or right
flank is determined from the accepted alignment's reference coordinates;
the overlap-layout assembler then determines compatible suffix--prefix joins
from the read sequences themselves.  The paper's reported unplaced-only
algorithm remains available by passing ``boundary_flank=None`` to
``asem_hybrid.run_asem_hybrid_loop``.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from asem_core import CombinedAlignmentCounts, accumulate_alignment, build_theta
from asem_hybrid import anchor_contigs, assemble_overlap_contigs
from common import LocalAlignment

AlignReadFn = Callable[[str, str], list[LocalAlignment]]


@dataclass
class BoundaryHybridIterationStats:
    iteration: int
    n_placed: int
    n_unplaced: int
    n_boundary_reads: int
    theta_changed: bool
    n_contigs: int
    n_contigs_anchored: int
    contig_lengths_anchored: list[int] = field(default_factory=list)


@dataclass
class AsemBoundaryHybridResult:
    theta: str
    history: list[BoundaryHybridIterationStats] = field(default_factory=list)
    theta_by_iteration: list[str] = field(default_factory=list)


def uncovered_gaps(depths: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return half-open zero-depth intervals ``[start, end)`` of sufficient length."""
    if min_len <= 0:
        min_len = 1
    zero = depths == 0
    if not zero.any():
        return []
    padded = np.concatenate(([False], zero, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= min_len]


def alignment_ref_interval(aln: LocalAlignment) -> tuple[int, int]:
    """Return the half-open reference interval consumed by an alignment."""
    ref_len = sum(base != "-" for base in aln.ref_aligned)
    return aln.ref_start, aln.ref_start + ref_len


def recruit_boundary_reads(
    placed: list[tuple[str, list[LocalAlignment]]],
    gaps: list[tuple[int, int]],
    flank: int,
) -> list[str]:
    """Select placed reads overlapping either reference-coordinate gap flank.

    A read is included once even when it has multiple accepted alignments or
    touches both flanks.  ``flank=150`` means that for a gap ``[a,b)`` we use
    placed alignments overlapping ``[a-150,a)`` or ``[b,b+150)``.  This avoids
    global reassembly of all placed reads while retaining coordinate-supported
    boundary sequence for overlap assembly.
    """
    if flank <= 0 or not gaps:
        return []
    boundary_reads: list[str] = []
    for read, alns in placed:
        for aln in alns:
            start, end = alignment_ref_interval(aln)
            if any(
                (start < gap_start and end > gap_start - flank)
                or (start < gap_end + flank and end > gap_end)
                for gap_start, gap_end in gaps
            ):
                boundary_reads.append(read)
                break
    return boundary_reads


def run_asem_boundary_hybrid_loop(
    theta_init: str,
    reads: list[str],
    align_read_fn: AlignReadFn,
    contig_tau: float = 0.25,
    min_anchor_len: int = 300,
    boundary_flank: int = 150,
    w: float = 0.1,
    max_iterations: int = 6,
    min_overlap: int = 20,
    n_workers: int = 1,
) -> AsemBoundaryHybridResult:
    """Run the boundary-recruited experimental variant.

    Ordinary placed-read voting is unchanged.  The extra boundary reads enter
    only the temporary overlap assembly pool; accepted contigs still enter the
    existing position-counting update exactly as in ordinary ASEM-Hybrid.
    """
    theta = theta_init
    history: list[BoundaryHybridIterationStats] = []
    theta_by_iteration: list[str] = []
    prev_unplaced_count: int | None = None

    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for it in range(1, max_iterations + 1):
            theta_prev = theta
            n = len(theta)
            counts = CombinedAlignmentCounts.empty(n)
            n_placed = 0
            unplaced_reads: list[str] = []
            placed: list[tuple[str, list[LocalAlignment]]] = []

            if pool is not None:
                results = pool.map(align_read_fn, reads, [theta] * len(reads), chunksize=32)
            else:
                results = (align_read_fn(read, theta) for read in reads)
            for read, alns in zip(reads, results):
                if alns:
                    n_placed += 1
                    placed.append((read, alns))
                    for aln in alns:
                        accumulate_alignment(counts, aln, n)
                else:
                    unplaced_reads.append(read)

            gaps = uncovered_gaps(counts.reference_depths(), min_anchor_len)
            boundary_reads = recruit_boundary_reads(placed, gaps, boundary_flank)
            if gaps:
                contigs = assemble_overlap_contigs(
                    unplaced_reads + boundary_reads, min_overlap=min_overlap
                )
                anchored = anchor_contigs(contigs, theta, contig_tau, min_anchor_len)
                for _, aln in anchored:
                    accumulate_alignment(counts, aln, n)
            else:
                contigs, anchored = [], []

            theta = build_theta(theta_prev, counts, w)
            theta_by_iteration.append(theta)
            changed = theta != theta_prev
            n_unplaced = len(unplaced_reads)
            history.append(
                BoundaryHybridIterationStats(
                    iteration=it,
                    n_placed=n_placed,
                    n_unplaced=n_unplaced,
                    n_boundary_reads=len(boundary_reads),
                    theta_changed=changed,
                    n_contigs=len(contigs),
                    n_contigs_anchored=len(anchored),
                    contig_lengths_anchored=[len(contig) for contig, _ in anchored],
                )
            )
            if not changed and (
                prev_unplaced_count is None or n_unplaced >= prev_unplaced_count
            ):
                break
            prev_unplaced_count = n_unplaced
    finally:
        if pool is not None:
            pool.shutdown()

    return AsemBoundaryHybridResult(theta=theta, history=history, theta_by_iteration=theta_by_iteration)
