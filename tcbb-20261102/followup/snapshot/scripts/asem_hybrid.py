"""Hybrid ASEM: reference-guided EM plus a local de novo gap-fill step.

Reference-guided placement (asem_core.py) leaves a genome's most divergent
regions untouched -- no read ever clears the alignment threshold there, so
those positions keep whatever base theta started with, forever (at the
hardest tested pair, 65.2% of the reference is never touched by a single
placed read).
But the reads that *originate* from those regions still overlap each other
exactly even when none of them overlaps theta, because they all come from
the same real target sequence.

This module adds one step to every EM iteration: assemble the pool of
reads that fail placement this round plus placed reads immediately adjacent
to a long zero-depth island via read-read overlap, then try to anchor each
resulting contig back onto theta with the same score-threshold rule used for
reads. The boundary reads are selected from their accepted reference
coordinates; the overlap assembler itself still decides sequence joins. An anchored
contig is fed into the same position/symbol vote-counting machinery as an
ordinary read (asem_core.accumulate_alignment), so a well-supported contig
can pull a previously-untouched span of theta straight to the true
sequence in one shot. Because next iteration's E-step re-aligns every
read -- including previously-unplaced ones -- against the patched theta,
reads that couldn't place before may now clear the threshold near a
contig's edges, shrinking the unplaced pool and letting contigs grow
further each round.

Validated on the single hardest-pair island before writing this: the 109
reads whose true origin fell in the largest untouched island assembled
into 2 contigs (1290bp + 781bp), both 100% identical to the true target
over their full length.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from asem_core import CombinedAlignmentCounts, accumulate_alignment, build_theta
from common import LocalAlignment, align_local, reverse_complement

MATCH_SCORE = 2.0  # must match common.align_local's parasail matrix

AlignReadFn = Callable[[str, str], list[LocalAlignment]]


@dataclass
class HybridIterationStats:
    iteration: int
    n_placed: int
    n_unplaced: int
    theta_changed: bool
    n_contigs: int
    n_contigs_anchored: int
    n_boundary_reads: int = 0
    contig_lengths_anchored: list[int] = field(default_factory=list)


@dataclass
class AsemHybridResult:
    theta: str
    history: list[HybridIterationStats] = field(default_factory=list)
    theta_by_iteration: list[str] = field(default_factory=list)


def suffix_prefix_overlap(a: str, b: str, min_overlap: int) -> int:
    """Longest suffix of `a` that exactly matches a prefix of `b`
    (>= min_overlap), or 0 if none reaches that length."""
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, min_overlap - 1, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0


def assemble_overlap_contigs(
    reads: list[str], min_overlap: int = 20, seed_k: int = 20
) -> list[str]:
    """Greedy overlap-layout assembly of `reads` by exact suffix-prefix
    overlap, on both strands. No reference is involved -- this only uses
    read-read overlap, which is why it can recover sequence theta can't reach.

    Raw FASTQ mixes both strands, so each read may join in either
    orientation. The layout is bidirected: a read has a left and a right end,
    an overlap consumes one end of each read, every end is used at most once,
    and a merge that would close a cycle is rejected. Each read therefore
    appears in exactly one contig, in one orientation, and contributes
    evidence once. A join (i, s) -> (j, t) between oriented reads is the same
    join as its mirror (j, ~t) -> (i, ~s); only one form is kept, preferring a
    forward left read. Each contig is reported in the orientation that keeps
    most of its reads as given (ties: the lexicographically smaller strand).
    On a single-orientation pool this reproduces forward-only assembly unless
    a reverse-complement overlap occurs by chance.

    Candidate pairs are pruned with a k-mer seed index (share a `seed_k`-mer
    at the suffix/prefix boundary) instead of checked exhaustively, since
    reads from unrelated regions of the genome essentially never share one
    by chance. This keeps assembly near-linear in read count rather than
    the O(n^3) cost of an all-pairs brute-force greedy merge.
    """
    n = len(reads)
    if n == 0:
        return []
    oriented = [reads, [reverse_complement(r) for r in reads]]

    # Anchor every oriented read by its first seed_k bases. A hit at offset p
    # of the left read implies a candidate overlap of length len(left) - p,
    # verified below.
    prefix_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s in (0, 1):
        for j, r in enumerate(oriented[s]):
            if len(r) >= seed_k:
                prefix_index[r[:seed_k]].append((j, s))

    # (overlap_len, i, j, s, t): oriented read (i, s)'s suffix overlaps
    # oriented read (j, t)'s prefix.
    seen: set[tuple[int, int, int, int]] = set()
    candidates: list[tuple[int, int, int, int, int]] = []
    for s in (0, 1):
        for i, r in enumerate(oriented[s]):
            for p in range(0, len(r) - seed_k + 1):
                for j, t in prefix_index.get(r[p : p + seed_k], ()):
                    if i == j:
                        continue
                    form = (s, i, j, t)
                    mirror = (1 - t, j, i, 1 - s)
                    key = min(form, mirror)
                    if key in seen:
                        continue
                    seen.add(key)
                    ov = suffix_prefix_overlap(r, oriented[t][j], min_overlap)
                    if ov > 0:
                        ks, ki, kj, kt = key
                        candidates.append((ov, ki, kj, ks, kt))
    # Longest overlaps first; ties in the forward-only assembler's (i, j) order.
    candidates.sort(key=lambda c: (c[0], c[1], c[2], -c[3], -c[4]), reverse=True)

    # End 1 is a read's right end in forward orientation, end 0 its left end.
    # Leaving (i, s) uses end 1 when s == 0; entering (j, t) uses end 0 when
    # t == 0.
    link: dict[tuple[int, int], tuple[int, int, int]] = {}
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ov, i, j, s, t in candidates:
        end_i = 1 if s == 0 else 0
        end_j = 0 if t == 0 else 1
        if (i, end_i) in link or (j, end_j) in link:
            continue
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue  # would close a cycle
        parent[root_i] = root_j
        link[(i, end_i)] = (j, end_j, ov)
        link[(j, end_j)] = (i, end_i, ov)

    contigs: list[str] = []
    visited: set[int] = set()
    for start in range(n):
        if start in visited or ((start, 0) in link and (start, 1) in link):
            continue  # already emitted, or interior to a path
        # Walk away from the free end: forward if the left end is free.
        cur, s = start, (0 if (start, 0) not in link else 1)
        seq = oriented[s][cur]
        n_reads, n_forward = 1, 1 - s
        visited.add(cur)
        while (cur, 1 if s == 0 else 0) in link:
            j, end_j, ov = link[(cur, 1 if s == 0 else 0)]
            t = 0 if end_j == 0 else 1
            seq += oriented[t][j][ov:]
            n_reads += 1
            n_forward += 1 - t
            visited.add(j)
            cur, s = j, t
        rc = reverse_complement(seq)
        if 2 * n_forward > n_reads:
            contigs.append(seq)
        elif 2 * n_forward < n_reads:
            contigs.append(rc)
        else:
            contigs.append(min(seq, rc))
    return contigs


def anchor_contigs(
    contigs: list[str], theta: str, contig_tau: float, min_anchor_len: int
) -> list[tuple[str, LocalAlignment]]:
    """Locally align each contig back onto theta and keep the ones with a
    long, reasonably-identical anchor window.

    This deliberately does NOT reuse the read-level tau/length rule (score
    >= tau * MATCH_SCORE * len(contig)). Empirically, contigs assembled
    from an unplaced-read pool anchor to theta at ~73-80% identity over
    their *aligned span* -- exactly the divergence level that made the
    underlying 150bp reads fail the read-level threshold in the first
    place -- and parasail's local alignment often can't sustain that rate
    across a contig's full length, so it returns the best-scoring
    *sub*-window rather than an end-to-end match. Judging that against a
    bar sized for the whole contig punishes exactly the long, high-value
    contigs this step exists to anchor.

    contig_tau=0.25 is a score-normalized threshold, not identity
    directly: with this scoring scheme (match=+2, mismatch=-3, no gaps),
    score >= contig_tau * MATCH_SCORE * L works out to an identity floor
    of exactly 70% (solve 2p - 3(1-p) >= 0.5 for p). min_anchor_len=300
    is an additional, independent floor on top of that -- NOT because
    70%-identity matches are common at 150bp against random sequence
    (they aren't: a purely-random-DNA negative control, contigs generated
    from `random.choice('ACGT')` with no relation to theta, produced 0
    false anchors in 2000 trials at each of the 150bp and 300bp length
    floors, so contig_tau=0.25 alone already rejects unrelated sequence
    reliably against that null model). The 300bp floor is a margin against
    a different, untested risk: real non-homologous genomic sequence is
    not perfectly uniform-random (GC bias, low-complexity/repetitive
    stretches), which could in principle score better against theta than
    the idealized random-DNA control above. This margin has not been
    validated against structured/repeat-containing negative controls --
    flagged here as an open gap, not a confirmed safety property.
    """
    anchored = []
    for contig in contigs:
        aln = align_local(contig, theta, try_reverse_complement=True)
        if aln is None:
            continue
        aligned_len = len(aln.read_aligned) - aln.read_aligned.count("-")
        if aligned_len < min_anchor_len:
            continue
        if aln.score >= contig_tau * MATCH_SCORE * aligned_len:
            anchored.append((contig, aln))
    return anchored


def has_uncovered_gap(depths: np.ndarray, min_len: int) -> bool:
    """True if `depths` (per-position placed-read support count) contains a
    contiguous run of zero-depth positions at least `min_len` long.

    Used to gate the de novo assembly/anchoring step: it is only worth
    attempting when theta actually has a hole big enough for a contig to
    plausibly fill (anchor_contigs requires an aligned span >= min_anchor_len
    anyway, so a hole shorter than that could never be usefully filled).
    This is a per-iteration cost-saving gate only, not a positional one:
    it controls whether the expensive assembly/anchoring step runs *this
    round* at all, not which theta positions an anchored contig is allowed
    to vote at. A contig's own alignment can (and empirically does) extend
    beyond the gap into positions ordinary reads already cover -- see the
    Results discussion of the resulting small recall cost on some jobs
    (Section~sec:hybrid-results). Don't describe this gate as restricting
    contig votes to gap positions; it doesn't."""
    if min_len <= 0:
        return True
    zero = depths == 0
    if not zero.any():
        return False
    padded = np.concatenate(([False], zero, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return bool(((ends - starts) >= min_len).any())


def run_asem_hybrid_loop(
    theta_init: str,
    reads: list[str],
    align_read_fn: AlignReadFn,
    contig_tau: float = 0.25,
    min_anchor_len: int = 300,
    boundary_flank: int | None = 150,
    w: float = 0.1,
    max_iterations: int = 6,
    min_overlap: int = 20,
    n_workers: int = 1,
) -> AsemHybridResult:
    """Run ASEM-Hybrid with boundary-recruited local assembly by default.

    ``boundary_flank`` selects placed reads whose accepted alignment overlaps
    a flank of this many reference bases adjacent to a long uncovered gap.
    Pass ``None`` to reproduce the original unplaced-only ASEM-Hybrid used
    for the published grid and manuscript results.
    """
    theta = theta_init
    history: list[HybridIterationStats] = []
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
            placed_reads: list[tuple[str, list[LocalAlignment]]] = []

            if pool is not None:
                results = pool.map(align_read_fn, reads, [theta] * len(reads), chunksize=32)
            else:
                results = (align_read_fn(r, theta) for r in reads)
            for read, alns in zip(reads, results):
                if alns:
                    n_placed += 1
                    placed_reads.append((read, alns))
                    for aln in alns:
                        accumulate_alignment(counts, aln, n)
                else:
                    unplaced_reads.append(read)

            depths = counts.reference_depths()
            boundary_reads: list[str] = []
            if has_uncovered_gap(depths, min_anchor_len):
                if boundary_flank is None:
                    assembly_reads = unplaced_reads
                else:
                    # Imported here to keep the experimental helper module
                    # independently testable while this public loop owns the
                    # default algorithm.  The import is safe after this module
                    # has fully initialized.
                    from asem_hybrid_boundary import uncovered_gaps, recruit_boundary_reads

                    boundary_reads = recruit_boundary_reads(
                        placed_reads, uncovered_gaps(depths, min_anchor_len), boundary_flank
                    )
                    assembly_reads = unplaced_reads + boundary_reads
                contigs = assemble_overlap_contigs(assembly_reads, min_overlap=min_overlap)
                anchored = anchor_contigs(contigs, theta, contig_tau, min_anchor_len)
                for contig, aln in anchored:
                    accumulate_alignment(counts, aln, n)
            else:
                contigs, anchored = [], []

            theta = build_theta(theta_prev, counts, w)
            theta_by_iteration.append(theta)

            changed = theta != theta_prev
            n_unplaced = len(unplaced_reads)
            history.append(
                HybridIterationStats(
                    iteration=it,
                    n_placed=n_placed,
                    n_unplaced=n_unplaced,
                    theta_changed=changed,
                    n_contigs=len(contigs),
                    n_contigs_anchored=len(anchored),
                    n_boundary_reads=len(boundary_reads),
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

    return AsemHybridResult(theta=theta, history=history, theta_by_iteration=theta_by_iteration)
