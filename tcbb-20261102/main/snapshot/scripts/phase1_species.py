"""Phase-1 6-species mtDNA set (from algorithm_sketch_iterative_refinement.md)
with taxonomic metadata for same-genus/family/order pair categorization,
matching the original paper's divergence-level grouping."""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phase1_reproduction")

# name -> (fasta filename, genus, family, order)
SPECIES = {
    "Homo_sapiens": ("Homo_sapiens_NC_012920.fasta", "Homo", "Hominidae", "Primates"),
    "Gorilla_gorilla": ("Gorilla_gorilla_NC_001645.fasta", "Gorilla", "Hominidae", "Primates"),
    "Saimiri_boliviensis": ("Saimiri_boliviensis_NC_021966.fasta", "Saimiri", "Cebidae", "Primates"),
    "Saimiri_sciureus": ("Saimiri_sciureus_NC_012775.fasta", "Saimiri", "Cebidae", "Primates"),
    "Aotus_azarai": ("Aotus_azarai_NC_021939.fasta", "Aotus", "Aotidae", "Primates"),
    "Varecia_variegata": ("Varecia_variegata_NC_012773.fasta", "Varecia", "Lemuridae", "Primates"),
}


# Supplementary verified same-genus pair (added in revision): the only
# Phase-1 same-genus pair, Saimiri boliviensis NC_021966 vs "S. sciureus"
# NC_012775, is 99.87% identical and NC_012775 has been argued to carry an
# S. b. boliviensis haplotype (Chiou et al. 2011). Select with
# ASEM_PAIR_SET=macaca_same_genus and run in a separate results tree.
PAIR_SET = os.environ.get("ASEM_PAIR_SET", "phase1").strip().lower()
if PAIR_SET == "macaca_same_genus":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ieee2021_mtdna_panel")
    SPECIES = {
        "Macaca_silenus": ("17_Macaca_silenus_NC_025221.fasta", "Macaca", "Cercopithecidae", "Primates"),
        "Macaca_tonkeana": ("19_Macaca_tonkeana_NC_025222.fasta", "Macaca", "Cercopithecidae", "Primates"),
    }
elif PAIR_SET != "phase1":
    raise ValueError(f"unknown ASEM_PAIR_SET: {PAIR_SET}")


def fasta_path(name: str) -> str:
    return os.path.join(DATA_DIR, SPECIES[name][0])


def divergence_level(ref_name: str, target_name: str) -> str:
    """same-genus < same-family < same-order, matching the original paper's
    reference-target divergence categories."""
    _, rg, rf, ro = SPECIES[ref_name]
    _, tg, tf, to = SPECIES[target_name]
    if rg == tg:
        return "same-genus"
    if rf == tf:
        return "same-family"
    if ro == to:
        return "same-order"
    return "different-order"


def ordered_pairs():
    """All (reference, target) pairs with reference != target."""
    names = list(SPECIES.keys())
    return [(r, t) for r in names for t in names if r != t]
