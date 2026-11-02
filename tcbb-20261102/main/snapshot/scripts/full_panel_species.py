"""The full original 30-species mtDNA panel (bmc/data/ieee2021_mtdna_panel/),
for full-scale visualization (e.g. the all-species phylogeny-clustered
heatmap) as opposed to phase1_species.py's 6-species subset used for the
actual iterative-refinement experiment grid (tractable compute budget)."""

import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ieee2021_mtdna_panel")

_FILENAME_RE = re.compile(r"^\d+_(.+)_[A-Z]{1,2}_?\d+\.fasta$")


def _species_name_from_filename(fname: str) -> str:
    m = _FILENAME_RE.match(fname)
    if not m:
        raise ValueError(f"unexpected filename format: {fname}")
    return m.group(1)


def list_species() -> dict[str, str]:
    """Returns {species_name: fasta_path} for all 30 species, sorted by the
    original panel's index (the filename prefix)."""
    files = sorted(
        (f for f in os.listdir(DATA_DIR) if f.endswith(".fasta")),
        key=lambda f: int(f.split("_", 1)[0]),
    )
    return {_species_name_from_filename(f): os.path.join(DATA_DIR, f) for f in files}
