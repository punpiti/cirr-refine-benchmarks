"""N-neutral alignment scoring for the benchmark evaluator.

Defect (found 2026-10-28): common._evaluate_linear_theta_vs_target aligns Theta
to the target with scikit-bio pair_align_nucl (BLASTN scores: match 2,
mismatch -3, gap 5/2, free end gaps). An ambiguous base N in Theta is scored
as a -3 mismatch. MIA emits N at weakly supported columns (median 6,276 N per
same-order output); when a large share of Theta is N, the optimal free-end
alignment can be the empty one (score 0), so every target base is counted as
unassembled and F1 = identity = 0 although Theta contains thousands of
correctly placed bases. Example: Homo_sapiens -> Varecia_variegata, 4X, r0:
recorded F1 0; same sequence with N scored 0 during alignment: F1 0.727.

Fix: score N against any symbol as 0 *during alignment only*. The metric
definitions are unchanged: an N column still counts as an incorrect Theta
position (w_theta), never as correct. For sequences without N the
substitution scores are identical to the original, and the evaluator output
was verified to be identical (verify_unchanged.py).
"""

from __future__ import annotations

import os
import sys

BENCH_SCRIPTS = os.environ.get("BENCH_SCRIPTS", os.path.expanduser("~/plos-reruns/bench-v1.3.0/scripts"))
if BENCH_SCRIPTS not in sys.path:
    sys.path.insert(0, BENCH_SCRIPTS)

import common  # noqa: E402
from skbio.alignment import pair_align_nucl as _original_pair_align_nucl  # noqa: E402
from skbio.sequence import SubstitutionMatrix  # noqa: E402

_ALPHABET = "ACGTN"
N_NEUTRAL_MATRIX = SubstitutionMatrix.from_dict({
    a: {b: (0.0 if "N" in (a, b) else (2.0 if a == b else -3.0)) for b in _ALPHABET}
    for a in _ALPHABET
})


def _to_acgtn(seq) -> str:
    # Any other IUPAC ambiguity code is treated like N for alignment scoring.
    return "".join(ch if ch in "ACGT" else "N" for ch in str(seq).upper())


def pair_align_nucl_nsafe(seq1, seq2, **kwargs):
    from skbio import DNA
    a, b = str(seq1), str(seq2)
    if set(a) <= set("ACGT") and set(b) <= set("ACGT"):
        return _original_pair_align_nucl(seq1, seq2, **kwargs)
    kwargs.setdefault("sub_score", N_NEUTRAL_MATRIX)
    return _original_pair_align_nucl(DNA(_to_acgtn(a)), DNA(_to_acgtn(b)), **kwargs)


def install(*modules) -> None:
    """Route common's evaluator (and any module that imported pair_align_nucl) through the fix."""
    common.pair_align_nucl = pair_align_nucl_nsafe
    for module in modules:
        if hasattr(module, "pair_align_nucl"):
            module.pair_align_nucl = pair_align_nucl_nsafe
