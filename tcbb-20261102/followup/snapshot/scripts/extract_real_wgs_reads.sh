#!/usr/bin/env bash
# Extract the mtDNA-mapped read pool used by the paper's Section 3.8
# real-WGS validation (NA07000, ENA project PRJEB31736, run ERR3239279,
# part of the 1000 Genomes Project 30x resequencing collection).
#
# This pulls only the chrM-mapped slice from the remote, indexed CRAM via
# samtools' range-query support -- the full ~18GB genome-wide file is never
# downloaded. Downstream filtering to primary/mapped/non-duplicate,
# exact-150bp records (samtools view -F 3332, then a length filter) is
# already done inside run_real_wgs_benchmark.py / run_real_wgs_hybrid_benchmark.py
# once this BAM exists locally -- this script only produces that BAM.
#
# Usage:
#   ./extract_real_wgs_reads.sh [output_dir]
#
# Requires: samtools, curl, and a reference FASTA whose sequence name
# matches the CRAM's @SQ SN: field exactly ("chrM" for this CRAM) -- fetch
# it first with `python3 fetch_data.py`, then rename its header, or supply
# your own chrM.fasta with a matching header via CHRM_REF_FASTA below.

set -euo pipefail

OUT_DIR="${1:-$(dirname "$0")/../data}"
mkdir -p "$OUT_DIR"

CRAM_URL="https://ftp.sra.ebi.ac.uk/vol1/run/ERR323/ERR3239279/NA07000.final.cram"
CRAI_PATH="$OUT_DIR/NA07000.final.cram.crai"
CHRM_REF_FASTA="${CHRM_REF_FASTA:-$OUT_DIR/refs/chrM_for_cram.fasta}"
OUT_BAM="$OUT_DIR/NA07000_chrM.bam"
SAMTOOLS="${SAMTOOLS_BIN:-samtools}"

if [ ! -f "$CHRM_REF_FASTA" ]; then
  echo "Missing $CHRM_REF_FASTA: fetch data/refs/chrM_rCRS.fasta via" >&2
  echo "fetch_data.py, then save a copy with its header changed to" >&2
  echo "'>chrM' (matching this CRAM's @SQ SN: field) at that path." >&2
  exit 1
fi

echo "Fetching CRAM index..."
curl -sf "${CRAM_URL}.crai" -o "$CRAI_PATH"

echo "Extracting chrM-mapped reads (remote range query, full CRAM not downloaded)..."
"$SAMTOOLS" view -b --reference "$CHRM_REF_FASTA" \
  -X "$CRAM_URL" "$CRAI_PATH" chrM -o "$OUT_BAM"

echo "Wrote $OUT_BAM ($("$SAMTOOLS" view -c "$OUT_BAM") reads)"
echo "Expected (paper run, 2026-08-18): 2,254,405 reads before the -F 3332 +"
echo "exact-150bp filtering that run_real_wgs*.py applies on load (leaving"
echo "1,860,886 primary/mapped/non-duplicate 150bp records)."
