"""Shared EM loop for the ASEM family of baselines (IEEE Access and OJEMB;
the two are called "ASEM" as a pair, distinct from ECTI-CON's unrelated
LCS/splitting method).

The two baselines differ only in how a single read is turned into zero or
more accepted local alignments (IEEE Access: one alignment or none; OJEMB:
also recursively re-aligns the unaligned flanks of a read, so one read can
contribute several disjoint sub-alignments -- see baseline_ojemb.py). Every
other part of the algorithm (position/symbol tallying, the match-state-only
consensus rule, the iterate-until-stable loop) is identical between them and
lives here once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from common import LocalAlignment, placement_observations

# Symbol -> row index into the (5, n) count array used by build_theta.
# A numpy count array (not one Counter object per reference position) is
# used so this stays cheap at real-genome scale (a chromosome-length
# reference would mean tens/hundreds of millions of positions -- that many
# separate Counter objects would be prohibitively slow and memory-heavy;
# a contiguous (5, n) int array is not).
SYMBOLS = "ACGT-"
SYMBOL_IDX = {s: i for i, s in enumerate(SYMBOLS)}
BASES = SYMBOLS[:-1]
BASE_IDX = {s: i for i, s in enumerate(BASES)}

# AlignReadFn: (read, theta) -> accepted local alignments for that read
# (already threshold-filtered by the caller; empty list means unplaced).
AlignReadFn = Callable[[str, str], list[LocalAlignment]]


@dataclass
class IterationStats:
    iteration: int
    n_placed: int
    n_unplaced: int
    theta_changed: bool


@dataclass
class AsemResult:
    theta: str
    history: list[IterationStats] = field(default_factory=list)
    theta_by_iteration: list[str] = field(default_factory=list)


@dataclass
class CombinedAlignmentCounts:
    """Streaming counts for reference columns and between-column insertions."""

    reference: np.ndarray
    insertion_bases: dict[int, np.ndarray]
    insertion_spans: np.ndarray

    @classmethod
    def empty(cls, reference_length: int) -> "CombinedAlignmentCounts":
        return cls(
            reference=np.zeros((len(SYMBOLS), reference_length), dtype=np.int64),
            insertion_bases={},
            insertion_spans=np.zeros(reference_length + 1, dtype=np.int64),
        )

    def reference_depths(self) -> np.ndarray:
        return self.reference.sum(axis=0)


def run_asem_em_loop(
    theta_init: str,
    reads: list[str],
    align_read_fn: AlignReadFn,
    w: float = 0.1,
    max_iterations: int = 6,
    n_workers: int = 1,
) -> AsemResult:
    theta = theta_init
    history: list[IterationStats] = []
    theta_by_iteration: list[str] = []
    prev_unplaced_count: int | None = None

    # n_workers<=1 (the common case when this run itself is one job in an
    # outer parallel grid) skips ProcessPoolExecutor entirely: spawning a
    # subprocess pool per run call turned out to dominate runtime (each
    # spawned worker re-imports scipy/pandas/skbio from scratch).
    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for it in range(1, max_iterations + 1):
            theta_prev = theta
            n = len(theta)
            counts = CombinedAlignmentCounts.empty(n)
            n_placed = 0
            n_unplaced = 0

            # Stream alignment results into `counts` as they arrive instead
            # of collecting every LocalAlignment in a list first: at WGS
            # read counts (millions), holding all of them in memory at once
            # is the real cost, not the alignment computation itself.
            if pool is not None:
                results = pool.map(align_read_fn, reads, [theta] * len(reads), chunksize=32)
            else:
                results = (align_read_fn(r, theta) for r in reads)
            for alns in results:
                if alns:
                    n_placed += 1
                    for aln in alns:
                        accumulate_alignment(counts, aln, n)
                else:
                    n_unplaced += 1

            theta = build_theta(theta_prev, counts, w)
            theta_by_iteration.append(theta)

            changed = theta != theta_prev
            history.append(IterationStats(it, n_placed, n_unplaced, changed))

            if not changed and (
                prev_unplaced_count is None or n_unplaced >= prev_unplaced_count
            ):
                break
            prev_unplaced_count = n_unplaced
    finally:
        if pool is not None:
            pool.shutdown()

    return AsemResult(theta=theta, history=history, theta_by_iteration=theta_by_iteration)


def accumulate_alignment(
    counts: CombinedAlignmentCounts, aln: LocalAlignment, n: int | None = None
) -> None:
    """Stream one gapped pairwise alignment into combined-alignment counts."""
    reference_length = counts.reference.shape[1]
    if n is not None and n != reference_length:
        raise ValueError("count layout length does not match reference length")

    calls, insertions, ref_end = placement_observations(aln)
    for pos, base in calls.items():
        if 0 <= pos < reference_length and base in SYMBOL_IDX:
            counts.reference[SYMBOL_IDX[base], pos] += 1

    first_internal_slot = max(aln.ref_start + 1, 1)
    last_internal_slot = min(ref_end, reference_length)
    if first_internal_slot < last_internal_slot:
        counts.insertion_spans[first_internal_slot:last_internal_slot] += 1

    for slot, sequence in insertions.items():
        if not 0 <= slot <= reference_length:
            continue
        if slot == 0 or slot == reference_length:
            counts.insertion_spans[slot] += 1
        matrix = counts.insertion_bases.get(slot)
        if matrix is None:
            matrix = np.zeros((len(BASES), 0), dtype=np.int64)
        if matrix.shape[1] < len(sequence):
            matrix = np.pad(matrix, ((0, 0), (0, len(sequence) - matrix.shape[1])))
            counts.insertion_bases[slot] = matrix
        for offset, base in enumerate(sequence):
            if base in BASE_IDX:
                matrix[BASE_IDX[base], offset] += 1



def tie_break(winners: np.ndarray, theta_prev: str, slot: int, offset: int) -> str:
    """Return the single winner, or pick one of several tied winners.

    The pick is uniform over the tied nucleotides but deterministic: it hashes
    the column coordinate (slot, insertion offset) and the current bases
    around it, so the same input always gives the same output and no base is
    favored, unlike the former lexicographic A/C/G/T order.
    """
    if len(winners) == 1:
        return BASES[int(winners[0])]
    context = theta_prev[max(0, slot - 10):slot + 10]
    digest = hashlib.blake2b(f"{slot}:{offset}:{context}".encode(), digest_size=8).digest()
    return BASES[int(winners[int.from_bytes(digest, "big") % len(winners)])]

def build_theta(theta_prev: str, counts: CombinedAlignmentCounts, w: float) -> str:
    n = len(theta_prev)
    reference_depths = counts.reference_depths()
    combined_depths = list(reference_depths)
    for slot, matrix in counts.insertion_bases.items():
        combined_depths.extend([counts.insertion_spans[slot]] * matrix.shape[1])
    mean_depth = float(np.mean(combined_depths)) if combined_depths else 0.0
    min_support = w * mean_depth

    output: list[str] = []
    for slot in range(n + 1):
        insertion_depth = int(counts.insertion_spans[slot])
        insertion_matrix = counts.insertion_bases.get(slot)
        if insertion_matrix is not None and insertion_depth > min_support:
            for offset in range(insertion_matrix.shape[1]):
                nucleotide_counts = insertion_matrix[:, offset]
                gap_count = insertion_depth - int(nucleotide_counts.sum())
                max_count = int(nucleotide_counts.max())
                winners = np.flatnonzero(nucleotide_counts == max_count)
                # An insertion has no prior base, so a nucleotide tie is
                # broken among the tied winners by tie_break.
                if 2 * gap_count <= insertion_depth:
                    output.append(tie_break(winners, theta_prev, slot, offset + 1))

        if slot == n:
            continue
        depth = int(reference_depths[slot])
        if depth <= min_support:
            output.append(theta_prev[slot])
            continue
        if 2 * int(counts.reference[SYMBOL_IDX["-"], slot]) > depth:
            continue
        nucleotide_counts = counts.reference[: len(BASES), slot]
        max_count = int(nucleotide_counts.max())
        winners = np.flatnonzero(nucleotide_counts == max_count)
        previous_base = theta_prev[slot]
        # A unique plurality updates the column. A tie that includes the
        # current base keeps it (the current sequence acts as the prior);
        # any other tie is broken among the tied winners by tie_break.
        if len(winners) > 1 and SYMBOL_IDX.get(previous_base, -1) in winners:
            output.append(previous_base)
        else:
            output.append(tie_break(winners, theta_prev, slot, 0))

    return "".join(output)
