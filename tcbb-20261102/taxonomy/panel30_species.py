"""The 30-species mammalian mtDNA panel of the original study (IEEE Access 2021),
used as the single simulation grid of the PLOS ONE manuscript.

Taxonomy follows the original panel, with Mus musculus corrected to family
Muridae (Muroidea is its superfamily). This correction leaves all pair tiers
unchanged because Mus is the only rodent in this panel.
All 870 ordered reference-target pairs are used; pairs from different orders
form a fourth tier, "different-order".
"""

from __future__ import annotations

import os

DATA_DIR = os.environ.get(
    "PANEL30_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "asem-refine-benchmarks", "data", "ieee2021_mtdna_panel"),
)

# name -> (fasta filename, genus, family, order)
SPECIES = {
    "Lagothrix_lagotricha": ("1_Lagothrix_lagotricha_NC_021951.fasta", "Lagothrix", "Atelidae", "Primates"),
    "Ateles_belzebuth": ("2_Ateles_belzebuth_NC_019800.fasta", "Ateles", "Atelidae", "Primates"),
    "Alouatta_seniculus": ("3_Alouatta_seniculus_NC_027825.fasta", "Alouatta", "Atelidae", "Primates"),
    "Aotus_azarai": ("4_Aotus_azarai_NC_021939.fasta", "Aotus", "Aotidae", "Primates"),
    "Callithrix_penicillata": ("5_Callithrix_penicillata_NC_030788.fasta", "Callithrix", "Callitrichidae", "Primates"),
    "Callimico_goeldii": ("6_Callimico_goeldii_NC_024628.fasta", "Callimico", "Callitrichidae", "Primates"),
    "Leontopithecus_rosalia": ("7_Leontopithecus_rosalia_NC_021952.fasta", "Leontopithecus", "Callitrichidae", "Primates"),
    "Saguinus_oedipus": ("8_Saguinus_oedipus_NC_021960.fasta", "Saguinus", "Callitrichidae", "Primates"),
    "Saimiri_boliviensis": ("9_Saimiri_boliviensis_NC_021966.fasta", "Saimiri", "Cebidae", "Primates"),
    "Pygathrix_cinerea": ("10_Pygathrix_cinerea_NC_018063.fasta", "Pygathrix", "Cercopithecidae", "Primates"),
    "Procolobus_verus": ("11_Procolobus_verus_NC_020666.fasta", "Procolobus", "Cercopithecidae", "Primates"),
    "Rhinopithecus_strykeri": ("12_Rhinopithecus_strykeri_NC_018059.fasta", "Rhinopithecus", "Cercopithecidae", "Primates"),
    "Rhinopithecus_brelichi": ("13_Rhinopithecus_brelichi_NC_018057.fasta", "Rhinopithecus", "Cercopithecidae", "Primates"),
    "Simias_concolor": ("14_Simias_concolor_NC_020667.fasta", "Simias", "Cercopithecidae", "Primates"),
    "Cercocebus_torquatus": ("15_Cercocebus_torquatus_NC_023964.fasta", "Cercocebus", "Cercopithecidae", "Primates"),
    "Allenopithecus_nigroviridis": ("16_Allenopithecus_nigroviridis_NC_023965.fasta", "Allenopithecus", "Cercopithecidae", "Primates"),
    "Macaca_silenus": ("17_Macaca_silenus_NC_025221.fasta", "Macaca", "Cercopithecidae", "Primates"),
    "Macaca_arctoides": ("18_Macaca_arctoides_NC_025201.fasta", "Macaca", "Cercopithecidae", "Primates"),
    "Macaca_tonkeana": ("19_Macaca_tonkeana_NC_025222.fasta", "Macaca", "Cercopithecidae", "Primates"),
    "Papio_papio": ("20_Papio_papio_NC_020009.fasta", "Papio", "Cercopithecidae", "Primates"),
    "Homo_sapiens": ("21_Homo_sapiens_NC_012920.fasta", "Homo", "Hominidae", "Primates"),
    "Gorilla_gorilla": ("22_Gorilla_gorilla_NC_001645.fasta", "Gorilla", "Hominidae", "Primates"),
    "Cephalopachus_bancanus": ("23_Cephalopachus_bancanus_NC_002811.fasta", "Cephalopachus", "Tarsiidae", "Primates"),
    "Varecia_variegata": ("24_Varecia_variegata_NC_012773.fasta", "Varecia", "Lemuridae", "Primates"),
    "Lemur_catta": ("25_Lemur_catta_NC_004025.fasta", "Lemur", "Lemuridae", "Primates"),
    "Galeopterus_variegatus": ("26_Galeopterus_variegatus_NC_004031.fasta", "Galeopterus", "Cynocephalidae", "Dermoptera"),
    "Oryctolagus_cuniculus": ("27_Oryctolagus_cuniculus_NC_001913.fasta", "Oryctolagus", "Leporidae", "Lagomorpha"),
    "Mus_musculus": ("28_Mus_musculus_KF937876.fasta", "Mus", "Muridae", "Rodentia"),
    "Tupaia_belangeri": ("29_Tupaia_belangeri_AF217811.fasta", "Tupaia", "Tupaiidae", "Scandentia"),
    "Sphaerias_blanfordi": ("30_Sphaerias_blanfordi_NC_046933.fasta", "Sphaerias", "Pteropodidae", "Chiroptera"),
}

TIERS = ["same-genus", "same-family", "same-order", "different-order"]


def fasta_path(name: str) -> str:
    return os.path.join(DATA_DIR, SPECIES[name][0])


def divergence_level(ref_name: str, target_name: str) -> str:
    _, rg, rf, ro = SPECIES[ref_name]
    _, tg, tf, to = SPECIES[target_name]
    if rg == tg:
        return "same-genus"
    if rf == tf:
        return "same-family"
    if ro == to:
        return "same-order"
    return "different-order"


def ordered_pairs() -> list[tuple[str, str]]:
    names = list(SPECIES)
    return [(r, t) for r in names for t in names if r != t]
