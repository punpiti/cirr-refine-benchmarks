"""Detached P8 chain: baseline precomputation, staging gate, then full audit/analysis."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    lock = (args.root/'.dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = Path(__file__).parent
    out = args.root/'results'
    out.mkdir(exist_ok=True)
    def state(stage, **extra):
        value = dict(pid=os.getpid(), stage=stage, updated_unix=time.time(), **extra)
        (args.root/'dispatch_status.json').write_text(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    try:
        state('reference_baseline_precompute', workers=args.workers)
        subprocess.run([sys.executable, str(code/'reference_baseline.py'), '--source', str(args.root/'source'),
                        '--output', str(out/'reference_baseline.csv'), '--workers', str(args.workers)], check=True)
        state('waiting_for_grid_copy')
        while not (args.root/'source.ready').exists():
            if (args.root/'source.failed').exists():
                raise RuntimeError('grid transfer failed; inspect i9 staging.log')
            time.sleep(10)
        state('full_acceptance_and_analysis')
        subprocess.run([sys.executable, str(code/'run_postprocess.py'), '--root', str(args.root),
                        '--workers', str(args.workers)], check=True)
        state('complete')
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
