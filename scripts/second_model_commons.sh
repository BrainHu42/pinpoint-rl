#!/usr/bin/env bash
# Lesson V5 on commons26 dev: PE-Core-G embeddings of its queries and areas' photos, SigLIP2 columns from the caches, then the trained answer head
# (none / PE-G / PE-G + SigLIP2) on dev3k and commons26 dev.
# Usage: TIMM_PATH=<dir with timm> setsid nohup bash scripts/second_model_commons.sh > artifacts/value/second_model_commons.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
tag=${TAG:-commons26dev}
echo "$(date +%T) embed pe_g $tag"
PYTHONPATH=${TIMM_PATH}:src .venvs/sft/bin/python scripts/second_model_gate.py embed --model pe_g --tag $tag 2>&1 | grep --line-buffered -v Warning || exit 1
echo "$(date +%T) siglip $tag"
PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py siglip --tag $tag 2>&1 || exit 1
echo "$(date +%T) head"
PYTHONPATH=src .venv/bin/python scripts/second_model_gate.py head --tags dev3k $tag --configs none pe_g pe_g+siglip_final 2>&1 | grep --line-buffered -v Warn
echo "$(date +%T) all done"
