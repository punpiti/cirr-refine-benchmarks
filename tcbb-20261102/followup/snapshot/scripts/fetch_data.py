"""Download the reference FASTA files these benchmarks expect, by NCBI
accession, instead of shipping copies of public NCBI records in this repo.

Every filename phase1_species.py / full_panel_species.py expects already
encodes its NCBI accession (e.g. "Homo_sapiens_NC_012920.fasta",
"21_Homo_sapiens_NC_012920.fasta"); this script just extracts it and
fetches that accession via NCBI's efetch, saving it under the exact
filename the rest of the code already looks for. Safe to re-run: an
existing, non-empty file is left alone.

Usage:
    python3 fetch_data.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ACCESSION_RE = re.compile(r"_([A-Za-z]{1,2}_?[0-9]{5,8})\.fasta$")

MANIFESTS = {
    "phase1_reproduction": [
        "Homo_sapiens_NC_012920.fasta",
        "Gorilla_gorilla_NC_001645.fasta",
        "Saimiri_boliviensis_NC_021966.fasta",
        "Saimiri_sciureus_NC_012775.fasta",
        "Aotus_azarai_NC_021939.fasta",
        "Varecia_variegata_NC_012773.fasta",
    ],
    "ieee2021_mtdna_panel": [
        "1_Lagothrix_lagotricha_NC_021951.fasta",
        "2_Ateles_belzebuth_NC_019800.fasta",
        "3_Alouatta_seniculus_NC_027825.fasta",
        "4_Aotus_azarai_NC_021939.fasta",
        "5_Callithrix_penicillata_NC_030788.fasta",
        "6_Callimico_goeldii_NC_024628.fasta",
        "7_Leontopithecus_rosalia_NC_021952.fasta",
        "8_Saguinus_oedipus_NC_021960.fasta",
        "9_Saimiri_boliviensis_NC_021966.fasta",
        "10_Pygathrix_cinerea_NC_018063.fasta",
        "11_Procolobus_verus_NC_020666.fasta",
        "12_Rhinopithecus_strykeri_NC_018059.fasta",
        "13_Rhinopithecus_brelichi_NC_018057.fasta",
        "14_Simias_concolor_NC_020667.fasta",
        "15_Cercocebus_torquatus_NC_023964.fasta",
        "16_Allenopithecus_nigroviridis_NC_023965.fasta",
        "17_Macaca_silenus_NC_025221.fasta",
        "18_Macaca_arctoides_NC_025201.fasta",
        "19_Macaca_tonkeana_NC_025222.fasta",
        "20_Papio_papio_NC_020009.fasta",
        "21_Homo_sapiens_NC_012920.fasta",
        "22_Gorilla_gorilla_NC_001645.fasta",
        "23_Cephalopachus_bancanus_NC_002811.fasta",
        "24_Varecia_variegata_NC_012773.fasta",
        "25_Lemur_catta_NC_004025.fasta",
        "26_Galeopterus_variegatus_NC_004031.fasta",
        "27_Oryctolagus_cuniculus_NC_001913.fasta",
        "28_Mus_musculus_KF937876.fasta",
        "29_Tupaia_belangeri_AF217811.fasta",
        "30_Sphaerias_blanfordi_NC_046933.fasta",
    ],
    # Real-WGS Section 3.8 references: fetched the same way, into refs/,
    # matching run_real_wgs*.py's/summarize_real_wgs_hybrid.py's expected
    # paths. The NA07000 read pool itself is not fetched here -- see
    # extract_real_wgs_reads.sh for that (it requires downloading and
    # samtools-filtering the full WGS BAM, not a single small accession).
    "refs": [
        "chimp_NC_001643.fasta",  # accession embedded in filename
        "chrM_rCRS.fasta",  # accession looked up via EXPLICIT_ACCESSIONS below
    ],
}

# A few expected filenames don't embed their accession (chrM_rCRS.fasta is
# named for readability, not by accession) -- handled explicitly instead
# of by the regex above.
EXPLICIT_ACCESSIONS = {
    os.path.join("refs", "chrM_rCRS.fasta"): "NC_012920",
}


def fetch_accession(accession: str, dest_path: str, timeout: float = 30.0) -> None:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  already have {dest_path}")
        return
    params = urllib.parse.urlencode(
        {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"}
    )
    url = f"{EUTILS_BASE}?{params}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    if not text.startswith(">"):
        raise ValueError(f"NCBI did not return a FASTA record for {accession!r}: {text[:200]!r}")
    tmp_path = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(tmp_path, "w") as fh:
        fh.write(text)
    os.replace(tmp_path, dest_path)
    print(f"  fetched {accession} -> {dest_path}")


def main() -> int:
    for subdir, filenames in MANIFESTS.items():
        out_dir = os.path.join(DATA_DIR, subdir)
        print(f"{subdir}:")
        for filename in filenames:
            rel_path = os.path.join(subdir, filename)
            accession = EXPLICIT_ACCESSIONS.get(rel_path)
            if accession is None:
                m = _ACCESSION_RE.search(filename)
                if not m:
                    print(f"  SKIP (no accession found in filename): {filename}", file=sys.stderr)
                    continue
                accession = m.group(1)
            dest_path = os.path.join(out_dir, filename)
            try:
                fetch_accession(accession, dest_path)
            except urllib.error.URLError as exc:
                print(f"  FAILED {accession} -> {filename}: {exc}", file=sys.stderr)
                return 1
            time.sleep(0.34)  # stay under NCBI's unauthenticated 3-requests/sec limit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
