"""Strand handling in the benchmark harness matches the asem-refine package."""

from __future__ import annotations

import random
import unittest

from asem_hybrid import assemble_overlap_contigs
from baseline_ieee_access import _align_read_ieee_access
from baseline_ojemb import _align_read_ojemb
from common import align_local, reverse_complement


def _rand_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


class StrandHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(7)
        self.ref = _rand_seq(2000, self.rng)

    def test_align_local_default_is_forward_only(self) -> None:
        rev = reverse_complement(self.ref[900:1050])
        fwd_only = align_local(rev, self.ref)
        both = align_local(rev, self.ref, try_reverse_complement=True)
        assert both is not None
        self.assertEqual(both.ref_start, 900)
        self.assertTrue(fwd_only is None or fwd_only.score < both.score)

    def test_cirr_places_reverse_strand_read(self) -> None:
        rev = reverse_complement(self.ref[400:550])
        alns = _align_read_ieee_access(rev, self.ref, tau=0.5)
        self.assertEqual([a.ref_start for a in alns], [400])

    def test_recursive_cirr_places_reverse_strand_read(self) -> None:
        rev = reverse_complement(self.ref[400:550])
        alns = _align_read_ojemb(rev, self.ref, tau=0.5, min_leftover_len=30)
        self.assertIn(400, [a.ref_start for a in alns])

    def test_forward_read_placement_unchanged(self) -> None:
        read = self.ref[1200:1350]
        self.assertEqual(_align_read_ieee_access(read, self.ref, tau=0.5)[0].ref_start, 1200)

    def test_hybrid_assembles_mixed_orientation_pool(self) -> None:
        genome = _rand_seq(3000, self.rng)
        reads = [genome[p : p + 150] for p in range(0, 2851, 50)]
        mixed = [reverse_complement(r) if self.rng.random() < 0.5 else r for r in reads]
        contigs = assemble_overlap_contigs(mixed)
        self.assertEqual(len(contigs), 1)
        self.assertIn(contigs[0], (genome, reverse_complement(genome)))
        self.assertEqual(assemble_overlap_contigs(reads), [genome])


if __name__ == "__main__":
    unittest.main()
