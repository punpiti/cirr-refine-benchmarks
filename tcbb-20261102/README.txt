CIRR 30-species benchmark and strand-aware follow-up — local release candidate, 2026-11-02

Status: prepared for author review. Not published, no new DOI assigned.
The cited Zenodo benchmark v1.4.0 remains the preceding public archive.

Contents
main/: frozen forward-only main-grid source and reference inputs.
followup/: separate strand-aware source; never substitute it for main-grid code.
results/: seven-method main CSV plus MITObim CSV, together 167,040 rows.
analysis/: accepted analysis, reference baseline, net counts and species sensitivity.
accepted/: manuscript summaries, mixed-orientation controls and auxiliary job tables.
tools/: postprocessing, MITObim and follow-up orchestration source.
taxonomy/: corrected current panel (Mus musculus family Muridae), with proof that all 870 pair tiers are unchanged. Frozen snapshots retain their historical label.
supplementary/: current supplements. S3/S4 and historical S1 Text diagnostics describe earlier experiments.

Reproducibility and provenance
SOURCE_HASHES.json records every original SHA-256 and packaged SHA-256.
Frozen source algorithms are preserved. For portable public paths, UTF-8 text replaces the original personal home directory with /home/benchmark and the personal OneDrive project prefix with /workspace/genome. The exact files affected are marked path_portability_edit=true. Snapshot manifests retain original content hashes; use SOURCE_HASHES.json to check packaged bytes. No accepted numerical value is changed by these substitutions.
This candidate includes result tables, not all retained per-job assemblies, tool binaries or simulation work directories. Environment package inventories are in the snapshot manifests; configure comparator executables using the bundled environment overrides before rerunning. No rerun was performed to prepare this archive.
The empirical BAM is excluded to avoid duplicating a public input. INPUTS_NOT_BUNDLED.json gives its original hash; use the bundled extraction script and ENA PRJEB31736 input to reconstruct it. Simulated reads are generated from the bundled reference FASTAs.
The extraction script and runners retain their original command interfaces. The tools/ directory is a source collection; full original machine directory layouts required by postprocessing are not reconstructed here.

Before publication
Review this candidate, choose the release version, upload to the public benchmark repository/Zenodo, verify the deposited bytes, then update the manuscript DOI/version and S1 Text release status together. Keep pending wording until the deposit is public. Public release and journal submission are separate actions.
