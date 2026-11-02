#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

workers="${ASEM_GRID_WORKERS:-14}"
stamp="20260929"
log_dir="scripts/results/post_tie_rule_logs"
mkdir -p "$log_dir"

run_grid() {
    local label="$1"
    local output_csv="$2"
    shift 2
    echo "[$(date -Is)] starting $label -> $output_csv"
    env \
        ASEM_GRID_WORKERS="$workers" \
        ASEM_HYBRID_WORKERS="$workers" \
        ASEM_GRID_RESULTS_CSV="$output_csv" \
        "$@" 2>&1 | tee -a "$log_dir/${label}.log"
    echo "[$(date -Is)] completed $label"
}

micromamba run -n genome python -m unittest discover -s scripts -p 'test_*.py'

run_grid \
    baseline1 \
    "scripts/results/baseline1_ieee_access/grid_results.post_tie_rule_partial_${stamp}.csv" \
    micromamba run -n genome python scripts/run_grid_baseline1.py

run_grid \
    baseline2 \
    "scripts/results/baseline2_ojemb/grid_results.post_tie_rule_partial_${stamp}.csv" \
    micromamba run -n genome python scripts/run_grid_baseline2.py

run_grid \
    hybrid_boundary \
    "scripts/results/baseline_hybrid_boundary/grid_results.post_tie_rule_partial_${stamp}.csv" \
    env ASEM_HYBRID_VARIANT=boundary \
    micromamba run -n genome python scripts/run_grid_baseline_hybrid.py

micromamba run -n genome python scripts/summarize_post_tie_rule.py
