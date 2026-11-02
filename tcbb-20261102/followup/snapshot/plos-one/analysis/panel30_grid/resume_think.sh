#!/usr/bin/env bash
# Resume the frozen snapshot, preserving every completed CSV checkpoint.
set -euo pipefail
run_root="${1:-$HOME/plos-reruns/cirr-20261030}"
python_bin="$HOME/.local/share/mamba/envs/genome/bin/python"
exec "$python_bin" -I "$run_root/snapshot/plos-one/analysis/panel30_grid/run_think_pipeline.py" --root "$run_root"
