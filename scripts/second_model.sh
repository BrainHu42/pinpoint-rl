#!/usr/bin/env bash
# Idea 2 (VALUE_PLAN.md): PE-Core embeddings of the dev3k queries and their areas' photos, then the gate (scripts/second_model_gate.py).
# Usage: TIMM_PATH=<dir with timm> setsid nohup bash scripts/second_model.sh > artifacts/value/second_model.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
mkdir -p artifacts/value
for m in ${MODELS:-pe_l pe_g}; do
  echo "$(date +%T) embed $m"
  PYTHONPATH=${TIMM_PATH}:src .venvs/sft/bin/python scripts/second_model_gate.py embed --model $m 2>&1 | grep --line-buffered -v Warning || exit 1
done
echo "$(date +%T) gate"
PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py gate --models ${MODELS:-pe_l pe_g} 2>&1
echo "$(date +%T) all done"
