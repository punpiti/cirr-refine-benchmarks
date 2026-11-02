"""Durable CSV checkpoints and recovery of a final interrupted write."""
from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path


def sync_directory(path: str | Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checkpoint(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    sync_directory(Path(handle.name).parent)


def recover_csv(path: str | Path, fields: list[str]) -> None:
    """Preserve and remove only an incomplete last record; reject earlier damage."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        if path.exists():
            path.unlink()
        return
    original = path.read_bytes()
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.end != len(original):
            raise
        text = original[:exc.start].decode("utf-8")
    lines = text.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    good_lines = 0
    try:
        for index, row in enumerate(reader):
            if reader.line_num == len(lines) and not original.endswith(b"\n"):
                break
            if index == 0 and row != fields:
                raise ValueError(f"unexpected CSV header: {path}")
            if len(row) != len(fields):
                if reader.line_num < len(lines):
                    raise ValueError(f"corrupt CSV record before end: {path}")
                break
            good_lines = reader.line_num
    except csv.Error:
        if reader.line_num < len(lines):
            raise
    kept = "".join(lines[:good_lines]).encode("utf-8")
    if kept != original:
        backup = path.with_name(path.name + f".interrupted-{time.time_ns()}.bak")
        with backup.open("wb") as handle:
            handle.write(original)
            checkpoint(handle)
        with path.open("wb") as handle:
            handle.write(kept)
            checkpoint(handle)
        if not kept:
            path.unlink()
            sync_directory(path.parent)
        print(f"Recovered interrupted CSV tail; original preserved at {backup}", flush=True)
