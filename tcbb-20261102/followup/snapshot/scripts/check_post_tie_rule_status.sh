#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

echo "Processes"
pgrep -af 'bash scripts/run_post_tie_rule_grids|micromamba run -n genome python scripts/run_grid_' || true

echo
echo "Completed jobs"
python3 - <<'PY'
import csv
from pathlib import Path

root = Path("scripts/results")
paths = [
    root / "baseline1_ieee_access" / "grid_results.post_tie_rule_partial_20260929.csv",
    root / "baseline2_ojemb" / "grid_results.post_tie_rule_partial_20260929.csv",
    root / "baseline_hybrid_boundary" / "grid_results.post_tie_rule_partial_20260929.csv",
]
for path in paths:
    if not path.exists():
        print(f"{path.parent.name:26s}   0/720 (not started)")
        continue
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = {
        (row["reference"], row["target"], row["depth"], row["replicate"])
        for row in rows
    }
    print(f"{path.parent.name:26s} {len(jobs):3d}/720 ({100 * len(jobs) / 720:5.1f}%)")
PY

echo
echo "Latest log lines"
for log in scripts/results/post_tie_rule_logs/*.log; do
    [[ -e "$log" ]] || continue
    echo "-- $log"
    tail -n 2 "$log"
done
