# cirr-refine-benchmarks

CIRR and CIRR-Hybrid are the current method names. Install the general-use
Python package from [cirr-refine](https://github.com/punpiti/cirr-refine).
This repository preserves the code and results that generated the archived
research outputs. Historical filenames, environment variables, raw method IDs
and frozen snapshot documentation retain their original spelling so that
recorded hashes and reproduction commands remain valid.

The frozen v1.5.0 archive is available at
[DOI 10.5281/zenodo.23874553](https://doi.org/10.5281/zenodo.23874553).
It does not contain all subsequent manuscript experiments.

Current frozen benchmark release: **v1.5.0**.

Version 1.5.0 adds `tcbb-20261102/`: the accepted 30-species mitochondrial
benchmark with eight methods and 167,040 main-grid rows, including MITObim;
the separate strand-aware follow-up source; reference baseline/net changes;
species-deletion sensitivity; and updated read-length, empirical and controlled
cost results. CIRR and CIRR-Hybrid retain their ASEM working names in source
and result tables. The main grid uses the frozen forward-only snapshot. The
strand-aware snapshot is archived separately and was used for the follow-up.

Start with [`tcbb-20261102/RELEASE.txt`](tcbb-20261102/RELEASE.txt).
The exact author-approved ZIP is a GitHub release asset with SHA-256
`5ac0804a28d239d3649bc273b45fc459ccf0966f1415b93a9e8c2f2c95a5ca8d`.
The files under `tcbb-20261102/` reproduce that ZIP without alteration.
Its README and S1 Text record the status at candidate preparation, before this
release; the deposit status is documented in RELEASE.txt. No experimental
result was changed during publication.

The preceding v1.4.0 added `plos-one/` for the earlier six-species manuscript.
The top-level `scripts/` and `plos-one/` remain historical reproduction paths.
The instructions below describe those earlier experiments; use the new release
folder for the current 30-species evidence.

Historical reproduction package for the CIRR/CIRR-Hybrid
experiments: the 720-job Phase-1 simulation grid, the
external-tool comparison, the read-length experiment, the computational-cost
profiling, parameter and circular-boundary diagnostics, MIA/Pilon comparisons,
and the real-WGS (NA07000) validation.

This repository preserves the research snapshots and raw results used to
produce the archived tables and figures. The current
[cirr-refine package](https://github.com/punpiti/cirr-refine) version 0.5.0
packages the strand-aware engine from `tcbb-20261102/followup/`, with source
fingerprints and CLI parity checks. Exact reproduction of older tables still
requires their own frozen snapshot and settings, including the separate
original forward-only main grid.

Version 1.2.0 contains the insertion-aware combined-alignment consensus used
by the revision manuscript and complete result CSVs regenerated with that
implementation. Its consensus update removes a sufficiently supported column
only under a strict gap majority. A unique nucleotide plurality is called;
tied current-reference columns retain their previous base, while tied candidate
insertions are omitted. Read-supported candidate insertion columns can be
retained between reference positions.

## Layout

- `scripts/` -- all benchmark/grid-runner code, the CIRR/CIRR-Hybrid/SRSC
  implementations as actually run, external-tool wrapper scripts, and the
  plotting/table-generation scripts that turned the raw CSVs into the
  paper's figures and tables.
- `scripts/results/` -- the raw result CSVs each grid run actually
  produced. These are the numbers the paper's tables and figures were
  built from; nothing here is synthetic or example data.
- `plos-one/` -- analyses added in v1.4.0 for the revised manuscript, with
  their result files; see `plos-one/README.md`.
- `data/` -- not checked in. Populated by `fetch_data.py` (public NCBI
  reference sequences, downloaded by accession) and, for the real-WGS
  validation only, `extract_real_wgs_reads.sh` (a public ENA read pool).
  See "Getting the data" below.

## Setup

```bash
bash setup_env.sh          # creates a micromamba env with all Python/bio dependencies
micromamba activate asem-bench
```

## Getting the data

Reference sequences are pre-existing public NCBI GenBank/RefSeq records;
this repo does not embed copies of them, only fetches them by accession
into `data/` (gitignored):

```bash
cd scripts
python3 fetch_data.py
```

This populates `data/phase1_reproduction/` (six-species Phase-1 panel),
`data/ieee2021_mtdna_panel/` (30-species divergence-landscape panel), and
`data/refs/` (human rCRS and chimpanzee references for the real-WGS
validation).

The real-WGS validation (Section 3.8) additionally needs the actual read
pool, which is a public human sequencing dataset rather than a small
reference sequence:

```bash
./extract_real_wgs_reads.sh
```

See that script's header comment for what it does and why it only
downloads the mtDNA-mapped slice (a remote-indexed range query), not the
full ~18GB whole-genome CRAM.

## Running the benchmarks

Each `run_grid_*.py` / `run_real_wgs*.py` / `run_read_length_experiment.py`
script is self-contained and writes its output CSV(s) under
`scripts/results/<name>/`, matching what's already checked in there. For
example:

```bash
cd scripts
python3 run_grid_baseline1.py   # non-recursive CIRR, full 720-job grid
python3 run_grid_baseline2.py   # recursive CIRR, full 720-job grid
python3 run_grid_baseline3.py   # SRSC, same grid
ASEM_HYBRID_VARIANT=boundary python3 run_grid_baseline_hybrid.py
```

The complete tie-rule verification can be resumed with one command from the
repository root:

```bash
bash scripts/run_post_tie_rule_grids.sh
bash scripts/check_post_tie_rule_status.sh
```

External-tool comparators (`run_grid_ext_novoplasty.py`,
`run_grid_ext_getorganelle.py`) additionally need NOVOPlasty/GetOrganelle
installed -- see `setup_env.sh`'s optional section.

MIA and Pilon default to the historical 12-job representative panel. Their
shared full-grid runs use the same 30 ordered pairs, depths 1--8X, three
replicates, and deterministic read seeds as the internal methods:

```bash
MIA_SCOPE=full MIA_TIMEOUT_S=300 python3 scripts/run_grid_ext_mia.py
PILON_SCOPE=full PILON_TIMEOUT_S=300 python3 scripts/run_grid_ext_pilon.py
```

Both runners append one row after each completed job and skip keys already in
their result CSV, so interrupted runs can be resumed with the same command.

NOVOPlasty and GetOrganelle outputs are scored after strand and
circular-origin normalization. The original wrappers did not retain
assemblies, so every originally successful job was rerun and rescored
(`normalized_successful_rerun.csv`, assemblies retained under `assemblies/`);
success/failure status still comes from the original `grid_results.csv`.
Some NOVOPlasty successes need more than the first-pass 300-s limit:

```bash
NOVOPLASTY_TIMEOUT_S=3600 python3 scripts/rerun_external_successes_normalized.py novoplasty --all-divergences
python3 scripts/rerun_external_successes_normalized.py getorganelle
python3 scripts/summarize_normalized_external.py
```

Acceptance records for the run used in the paper are in
`scripts/results/normalized_rerun_20261001/`.

The `make_*.py` scripts regenerate the paper's figures/tables from the
CSVs in `scripts/results/`.

## Notes

- Hardcoded machine-specific paths from the original research environment
  (tool binary locations, a personal data directory) have been replaced
  with environment variables that fall back to sensible defaults relative
  to this repo (`SAMTOOLS_BIN`, `NOVOPLASTY_PL`, `GETORGANELLE_ENV_BIN`,
  `GENOME_WGS_DATA_DIR`, `GENOME_ENV_PYTHON`) -- set them if your tools
  live somewhere else or aren't on `PATH`.
- `scripts/asem_hybrid.py` and `scripts/asem_hybrid_boundary.py` are the
  exact algorithm implementations used to generate the reported results,
  kept here even though `cirr-refine` now also implements the
  boundary-recruited variant, for byte-for-byte reproducibility of this
  paper's specific numbers.
- License: MIT (see `LICENSE`).
