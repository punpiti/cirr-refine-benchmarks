"""Regression tests for strand-aware (bidirected) Hybrid overlap assembly."""

from __future__ import annotations

import random
import unittest

from asem_refine.align import reverse_complement
from asem_refine.hybrid import assemble_overlap_contigs


def _rand_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _tiled_reads(genome: str, read_len: int, step: int) -> list[str]:
    return [genome[p : p + read_len] for p in range(0, len(genome) - read_len + 1, step)]


class BidirectedAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(11)
        self.genome = _rand_seq(3000, self.rng)
        self.reads = _tiled_reads(self.genome, 150, 50)

    def test_forward_pool_reassembles_genome(self) -> None:
        reads = self.reads[:]
        self.rng.shuffle(reads)
        self.assertEqual(assemble_overlap_contigs(reads), [self.genome])

    def test_mixed_orientation_pool_reassembles_genome(self) -> None:
        mixed = [reverse_complement(r) if self.rng.random() < 0.5 else r for r in self.reads]
        contigs = assemble_overlap_contigs(mixed)
        self.assertEqual(len(contigs), 1)
        self.assertIn(contigs[0], (self.genome, reverse_complement(self.genome)))

    def test_mostly_forward_pool_reports_forward_strand(self) -> None:
        mixed = [reverse_complement(r) if i % 4 == 0 else r for i, r in enumerate(self.reads)]
        self.assertEqual(assemble_overlap_contigs(mixed), [self.genome])

    def test_all_reverse_pool_reports_given_strand(self) -> None:
        reverse = [reverse_complement(r) for r in self.reads]
        self.assertEqual(assemble_overlap_contigs(reverse), [reverse_complement(self.genome)])

    def test_each_read_used_once_across_contigs(self) -> None:
        # Two unrelated loci, each sampled on both strands, plus a singleton.
        other = _rand_seq(1200, self.rng)
        pool = (self.reads + [reverse_complement(r) for r in _tiled_reads(other, 150, 50)]
                + [_rand_seq(150, self.rng)])
        contigs = assemble_overlap_contigs(pool)
        self.assertEqual(len(contigs), 3)
        total = sum(len(c) for c in contigs)
        self.assertEqual(total, len(self.genome) + len(other) + 150)

    def test_read_and_its_reverse_complement_do_not_double_count(self) -> None:
        pool = self.reads + [reverse_complement(r) for r in self.reads]
        contigs = assemble_overlap_contigs(pool)
        # Each locus copy assembles once per strand copy at most; no contig
        # may exceed the genome length by reusing a read in both orientations.
        self.assertTrue(all(len(c) <= len(self.genome) for c in contigs))
        self.assertIn(self.genome, contigs + [reverse_complement(c) for c in contigs])

    def test_empty_pool(self) -> None:
        self.assertEqual(assemble_overlap_contigs([]), [])


if __name__ == "__main__":
    unittest.main()
