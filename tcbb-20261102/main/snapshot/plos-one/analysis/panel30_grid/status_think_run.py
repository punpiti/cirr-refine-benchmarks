"""Read-only progress report; no AI or benchmark imports needed."""
import argparse
import collections
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path.home() / "plos-reruns/cirr-20261030")
parser.add_argument("--json", action="store_true")
args = parser.parse_args()
state_path = args.root / "pipeline_status.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "not started"}
progress = {}
for name, expected, filename in (("runtime", 108, "runtime_results.csv"), ("read_length", 189, "results.csv"),
                                  ("real_reads", 144, "results.csv"), ("panel30", 146160, "results.csv")):
    path = args.root / name / filename
    rows = list(csv.DictReader(path.open(newline=""))) if path.exists() else []
    progress[name] = {"done": len(rows), "total": expected,
                      "statuses": dict(collections.Counter(r.get("status") for r in rows)),
                      "methods": dict(collections.Counter(r.get("method") for r in rows))}
state["progress"] = progress
if args.json:
    print(json.dumps(state, indent=2))
else:
    print(f"{state.get('phase', 'pipeline')}: {state['status']}")
    for name, counts in progress.items():
        print(f"{name}: {counts['done']:,}/{counts['total']:,} {counts['statuses']}")
