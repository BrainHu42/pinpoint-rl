#!/bin/bash
# R4 sweep: data (b1 / b2) x cells x epochs; dev3k top-cell accuracy per run in artifacts/value/region_expert_sweep.log
cd "$(dirname "$0")/.."
run() { systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 env PYTHONPATH=src .venv/bin/python scripts/region_expert.py train "$@" 2>&1 | grep -v Warn; }
{
run --data b1 --cells 2048 --epochs 15
run --data b2 --cells 2048 --epochs 4
run --data b2 --cells 2048 --epochs 10
run --data b2 --cells 8192 --epochs 10
run --data b2 --cells 16384 --epochs 10
} > artifacts/value/region_expert_sweep.log 2>&1
