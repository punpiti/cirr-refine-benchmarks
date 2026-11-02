"""Shared safety helpers for the ext-tool grid runners (run_grid_ext_novoplasty.py,
run_grid_ext_novoplasty_retry_timeouts.py, and friends):

- ts()/log(): wall-clock-timestamped log lines, so a log file makes sense on
  its own hours later without cross-referencing "elapsed seconds".
- acquire_lock()/release_lock(): a lightweight hostname+pid lock file per
  results dir, so a second accidental run (for example from another machine
  that shares the same synced results folder) cannot append to the same
  grid_results.csv at the same time.
- snapshot_csv(): copies the current grid_results.csv into a timestamped
  backups/ subfolder. The live file is rewritten throughout a run; a
  timestamped snapshot is a stable copy to inspect and the recovery point if
  the live file is ever corrupted.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import sys
import time

STALE_LOCK_HOURS = 12


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def acquire_lock(results_dir: str, script_name: str) -> str:
    """Exit the process if another host/pid appears to be actively using this
    results dir; otherwise claim it and register auto-release at exit."""
    lock_path = os.path.join(results_dir, ".run.lock")
    host = socket.gethostname()
    pid = os.getpid()

    if os.path.exists(lock_path):
        try:
            with open(lock_path) as fh:
                info = json.load(fh)
        except (json.JSONDecodeError, OSError):
            info = {}
        age_hours = (time.time() - os.path.getmtime(lock_path)) / 3600
        same_host = info.get("host") == host
        if same_host and _pid_alive(info.get("pid")):
            log(f"REFUSING TO START: {info.get('script')} is already running here "
                f"as pid {info.get('pid')} (lock: {lock_path}). If that's wrong "
                f"(stale lock from a crash), delete the lock file and retry.")
            sys.exit(1)
        if not same_host and age_hours < STALE_LOCK_HOURS:
            log(f"REFUSING TO START: lock held by host={info.get('host')} "
                f"pid={info.get('pid')} script={info.get('script')} "
                f"since {info.get('started')} ({age_hours:.1f}h ago). Looks like "
                f"a run may still be in progress on another machine sharing this "
                f"synced folder. If that run is actually finished/dead, delete "
                f"{lock_path} and retry.")
            sys.exit(1)
        log(f"found a stale lock (host={info.get('host')}, {age_hours:.1f}h old) "
            f"-- taking over.")

    os.makedirs(results_dir, exist_ok=True)
    with open(lock_path, "w") as fh:
        json.dump({"host": host, "pid": pid, "script": script_name, "started": ts()}, fh)
    atexit.register(release_lock, lock_path)
    return lock_path


def release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def snapshot_csv(csv_path: str) -> str | None:
    if not os.path.exists(csv_path):
        return None
    backup_dir = os.path.join(os.path.dirname(csv_path), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(backup_dir, f"{os.path.basename(csv_path)}.{stamp}")
    shutil.copy2(csv_path, dest)
    return dest
