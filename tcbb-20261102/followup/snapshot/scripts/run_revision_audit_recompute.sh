#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

workers="${ASEM_EXTERNAL_WORKERS:-8}"

micromamba run -n genome python -m unittest discover -s scripts -p 'test_*.py'
micromamba run -n genome python scripts/rerun_external_successes_normalized.py novoplasty --workers "$workers"
micromamba run -n genome python scripts/rerun_external_successes_normalized.py getorganelle --workers "$workers"
micromamba run -n genome python scripts/summarize_normalized_external.py
micromamba run -n genome python -m py_compile \
  scripts/common.py \
  scripts/rerun_external_successes_normalized.py \
  scripts/summarize_normalized_external.py
