"""Baseline 3: ECTI-CON (Benchaphattharaworakul, Phisanbut, Srikulnath,
Piamsa-nga, ECTI-CON 2021, doi:10.1109/ECTI-CON51831.2021.9454937).

Structurally unrelated to the ASEM family (baseline_ieee_access.py /
baseline_ojemb.py / asem_core.py): a single-pass, non-iterative,
LCS-based split-align-assemble-merge pipeline, not an EM/Profile-HMM
consensus loop. Read directly from the paper's page images (pdftotext could
not extract this PDF's body text -- it appears to be scan/image-based).

Confirmed from the paper (Section II, Fig. 1):
1. Split the reference into M overlapping sub-references, each longer than
   the max read length, with m bp of overlap between consecutive chunks.
2. Locally align every read to every sub-reference via the longest common
   substring (LCS) -- not Smith-Waterman. A read need not align end-to-end,
   may match more than one sub-reference, and is dropped only if its best
   LCS match falls below a minimum confidence length. This paper's dataset
   is the same 6-species Phase-1 set already used here (confirmed by
   reading Section III: "Homo sapiens (NC_012920.1), Gorilla gorilla
   (NC_001645.1), Saimiri boliviensis (NC_021966.1), Aotus azarai
   (NC_021939.1), Varecia variegata (NC_012773.1), Saimiri sciureus
   (NC_012775.1)").
3. Reads matched to a sub-reference are assembled independently into a
   contig for that sub-reference (majority vote per position; positions
   with no read support keep the sub-reference's own base), keeping a
   flanking margin on each contig end specifically to support step 4's
   overlap test.
4. Contigs are merged by highest pairwise intersection: repeatedly find the
   pair of contigs (in either orientation) whose end-flanks share the
   longest common substring, merge them, and return the merged contig to
   the pool, until no pair clears the merge threshold. Because merging is
   driven by sequence overlap rather than original chunk position, this
   step is what gives the method its claimed tolerance to
   insertion/deletion/duplication/reordering between sub-reference-derived
   contigs.

Unspecified-in-the-paper hyperparameters (the paper never states numeric
values for these either, matching the pattern in baseline_ieee_access.py /
baseline_ojemb.py):
- sub_ref_len (L): sub-reference length, must exceed max read length.
  Default 600 (4x the 150bp read length).
- overlap (m): overlap between consecutive sub-references. Default 150,
  raised from an initial naive default of 100 after a targeted diagnostic
  (not part of the main grid) surfaced a read-length-dependent weakness:
  at overlap=100, the same-family same-pair F1 degrades sharply across
  roughly the 100-250bp read-length band (worst observed: 0.489 at 125bp,
  vs. 0.963-0.968 at 50-75bp), because `_try_merge`'s flank-overlap LCS
  (which defaults to width `overlap`) needs enough room to find an
  uninterrupted matching run despite same-family-level substitution noise;
  when a read is comparable in length to that flank, per-position vote
  accuracy right at the chunk boundary degrades and merges fail more
  often, and `_merge_by_overlap` silently drops every contig except the
  single largest survivor when a merge can't be found. A controlled
  ablation (single replicate, same simulated reads at each length, only
  `overlap` varied) confirmed this is causal, not incidental: raising
  overlap to 150-200 substantially reduces the degradation across the
  full 50-300bp range tested (that single-replicate ablation showed F1
  staying at 0.93-0.97 throughout). This is a post-hoc parameter
  adjustment, made after this same weakness surfaced in our own
  experiments -- named as such rather than described as a value chosen
  independently of the data. It does NOT fully eliminate the failure
  mode: the actual 3-replicate confirmatory run (run_read_length_experiment.py,
  not this single-replicate diagnostic) still shows occasional merge
  failures at 250-300bp even at overlap=150 -- e.g. one of three
  replicates at 250bp lands at F1=0.729 against ~0.935 for the other two,
  and two of three replicates at 300bp land at 0.805. 150 and 200
  performed within noise of each other in the single-replicate ablation
  (worst-case F1 0.9326 vs. 0.9317, both at 300bp); 150 was kept as the
  smaller of the two since neither offered a further benefit there. This
  choice trades off a small increase in chunk count (and thus runtime)
  for reduced (not eliminated) read-length fragility; it was not tuned to
  maximize any specific benchmark pair's score, but it is tuning -- chosen
  post-hoc from observing this project's own read-length experiment, not
  a value fixed in advance or proven optimal in general.
- min_match_frac: minimum LCS match length to accept a read-to-sub-reference
  alignment, as a fraction of read length. Default 0.25 (38bp of 150bp).
  Swept 0.5 down to 0.1 on the hardest pair in the Phase-1 set (Homo vs
  Varecia, 30X): safe and mildly beneficial from ~0.25-0.5 (identity stays
  at or slightly above the no-refinement baseline, 67.5%), but degrades
  sharply and unpredictably below ~0.2 (one run at frac=0.17 crashed
  identity from 67.5% to 45.8%) -- once the minimum match length gets
  short enough that a handful of *coincidental* exact matches slip through,
  each one anchors a whole 150bp read at the wrong offset and votes
  garbage across that whole span (see the offset-projection note on
  `_build_contig` below), which is far more damaging than just leaving a
  read unplaced. 0.25 sits in the safe region with a margin, not chosen to
  maximize the score on this one pair.
- flank: length of the end-flank compared between contigs during merging.
  Default None -> equals `overlap` (150), since that is exactly how much
  true overlap two originally-adjacent contigs should share.
- merge_min_len: minimum flank-overlap length to accept a merge. Default 30.
"""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass


def split_reference(ref: str, sub_ref_len: int, overlap: int) -> list[tuple[int, str]]:
    """Overlapping sub-references: consecutive chunks share `overlap` bases."""
    step = sub_ref_len - overlap
    chunks: list[tuple[int, str]] = []
    start = 0
    n = len(ref)
    while True:
        end = min(start + sub_ref_len, n)
        chunks.append((start, ref[start:end]))
        if end >= n:
            break
        start += step
    return chunks


def _lcs(a: str, b: str) -> tuple[int, int, int]:
    """Longest common substring. Returns (start_in_a, start_in_b, length)."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = sm.find_longest_match(0, len(a), 0, len(b))
    return match.a, match.b, match.size


@dataclass
class EctiConResult:
    theta: str
    n_chunks: int
    n_placed: int
    n_unplaced: int
    n_contigs_before_merge: int
    n_contigs_after_merge: int


def _seed_hit(seq: str, chunk: str, seed_len: int) -> bool:
    """Cheap pre-filter: does any non-overlapping seed_len-mer tile of `seq`
    occur as an exact substring of `chunk`? Uses Python's C-optimized
    str.find rather than difflib's SequenceMatcher, which is orders of
    magnitude cheaper. Only chunks that pass this filter get the expensive
    full LCS computation below -- otherwise testing every (read, chunk)
    pair with full LCS does not scale (with ~30 chunks per reference and
    hundreds of reads, that is thousands of difflib calls per run, most of
    them on chunks with no real relationship to the read at all).

    This is a documented approximation, not part of the original paper: a
    read whose only true long match to a chunk happens to fall between two
    seed tile boundaries could be missed. seed_len is kept short (15bp)
    with tiling across the whole read to keep that false-negative rate low
    in practice.
    """
    for i in range(0, len(seq) - seed_len + 1, seed_len):
        if chunk.find(seq[i : i + seed_len]) != -1:
            return True
    return False


def run_ecticon(
    reference: str,
    reads: list[str],
    sub_ref_len: int = 600,
    overlap: int = 150,
    min_match_frac: float = 0.25,
    flank: int | None = None,
    merge_min_len: int = 30,
    seed_len: int = 15,
) -> EctiConResult:
    if flank is None:
        flank = overlap

    chunks = split_reference(reference, sub_ref_len, overlap)
    chunk_matches: list[list[tuple[str, int, int, int]]] = [[] for _ in chunks]

    n_placed = 0
    n_unplaced = 0
    for read in reads:
        min_len = min_match_frac * len(read)
        placed_this_read = False
        for ci, (_, chunk_seq) in enumerate(chunks):
            if not _seed_hit(read, chunk_seq, seed_len):
                continue
            read_pos, chunk_pos, length = _lcs(read, chunk_seq)
            if length >= min_len:
                chunk_matches[ci].append((read, read_pos, chunk_pos, length))
                placed_this_read = True
        if placed_this_read:
            n_placed += 1
        else:
            n_unplaced += 1

    contigs = [
        _build_contig(chunk_seq, chunk_matches[ci])
        for ci, (_, chunk_seq) in enumerate(chunks)
    ]

    merged_pool, n_after = _merge_by_overlap(contigs, flank, merge_min_len)
    theta = max(merged_pool, key=len)

    return EctiConResult(
        theta=theta,
        n_chunks=len(chunks),
        n_placed=n_placed,
        n_unplaced=n_unplaced,
        n_contigs_before_merge=len(contigs),
        n_contigs_after_merge=n_after,
    )


def _build_contig(chunk_seq: str, matches: list[tuple[str, int, int, int]]) -> str:
    """Majority-vote consensus per position from all reads matched to this
    sub-reference; positions with no read support keep the sub-reference's
    own base (there is no prior-model fallback threshold here the way
    asem_core.build_theta has `w` -- ECTI-CON is a single non-iterative
    pass, so "any support beats none" is the natural rule).

    The LCS match only identifies where a read anchors onto the chunk (an
    exact-match core, by definition of "longest common substring" -- read
    and chunk necessarily already agree there). Voting using only that core
    span is a tautology that can never correct anything, since it can only
    ever reproduce bases the chunk already has. The LCS core is used here
    only to compute the read's alignment offset onto the chunk; every
    position of the FULL read is then projected onto the chunk via that
    offset and contributes a vote, including the parts outside the exact
    core, which is where the read's information about how the chunk should
    differ from the reference actually lives."""
    n = len(chunk_seq)
    counts: list[Counter] = [Counter() for _ in range(n)]
    for read, read_pos, chunk_pos, _length in matches:
        offset = chunk_pos - read_pos  # read index k -> chunk position offset+k
        for k, base in enumerate(read):
            pos = offset + k
            if 0 <= pos < n:
                counts[pos][base] += 1

    out = []
    for i in range(n):
        if counts[i]:
            base, _ = counts[i].most_common(1)[0]
        else:
            base = chunk_seq[i]
        out.append(base)
    return "".join(out)


def _merge_by_overlap(contigs: list[str], flank: int, merge_min_len: int) -> tuple[list[str], int]:
    """Iteratively merge the pair of contigs (checked in both orientations)
    whose end-flanks share the longest common substring, until no pair
    clears merge_min_len or only one contig remains."""
    pool = list(contigs)
    n_after_first_merge = len(pool)

    while len(pool) > 1:
        best = None  # (length, i, j, merged_seq)
        for i in range(len(pool)):
            for j in range(len(pool)):
                if i == j:
                    continue
                merged = _try_merge(pool[i], pool[j], flank)
                if merged is None:
                    continue
                length, seq = merged
                if best is None or length > best[0]:
                    best = (length, i, j, seq)

        if best is None or best[0] < merge_min_len:
            break

        _, i, j, seq = best
        pool = [p for k, p in enumerate(pool) if k not in (i, j)]
        pool.append(seq)
        n_after_first_merge = len(pool)

    return pool, n_after_first_merge


def _try_merge(a: str, b: str, flank: int) -> tuple[int, str] | None:
    """Try merging a followed by b: does a's suffix (last `flank` bases)
    share a long common substring with b's prefix (first `flank` bases)?
    Returns (overlap_length, merged_sequence) or None."""
    a_flank_start = max(0, len(a) - flank)
    a_flank = a[a_flank_start:]
    b_flank = b[:flank]
    sa, sb, length = _lcs(a_flank, b_flank)
    if length == 0:
        return None
    overlap_start_in_a = a_flank_start + sa
    overlap_start_in_b = sb
    merged = a[:overlap_start_in_a] + b[overlap_start_in_b:]
    return length, merged
