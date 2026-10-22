#!/usr/bin/env bash
# Idea 2, the real test: once scripts/second_model_train.sh logs "all done", the answer head with the new columns (second_model_gate.py head).
# Usage: setsid nohup bash scripts/second_model_head.sh > artifacts/value/second_model_head.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
until grep -q "all done" artifacts/value/second_model_train.log 2>/dev/null; do sleep 120; done
echo "$(date +%T) head"
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 env PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py head \
  --models ${MODELS:-pe_g dino_7b} 2>&1 | grep --line-buffered -v Warn
echo "$(date +%T) all done"
