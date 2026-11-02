#!/usr/bin/env bash
# Wait for the other PLOS runs to finish and the machine to go idle, then
# start the controlled runtime benchmark (sequential, one job at a time).
cd "$(dirname "$0")"
while pgrep -f "run_read_length_all_methods|run_real_reads_all_methods|rerun_real_wgs|run_real_wgs|run_mia_followup" >/dev/null; do sleep 60; done
until awk '{exit !($1 < 1.0)}' /proc/loadavg; do sleep 30; done
echo "start $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
exec ~/.local/share/mamba/envs/genome/bin/python run_runtime_benchmark.py --max-load 1.5
