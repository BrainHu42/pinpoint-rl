#!/usr/bin/env bash
# Idea 2, train side: after the PE-Core-G train embedding (pid $1, if given) exits, embed DINOv3 ViT-7B over the same train photos (the ones
# that move the answer head's loss, artifacts/value/train_informative_setft_own.npy). Both are resumable: rerun to continue.
# Usage: TIMM_PATH=<dir with timm> setsid nohup bash scripts/second_model_train.sh [pid] > artifacts/value/second_model_train.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
if [ -n "${1:-}" ]; then while kill -0 "$1" 2>/dev/null; do sleep 60; done; fi
for m in ${MODELS:-pe_g dino_7b}; do
  if [ -f artifacts/value/${m}_train_setft_own.progress ] && grep -q "unreadable" artifacts/value/${m}_train.log 2>/dev/null; then continue; fi
  echo "$(date +%T) embed $m (train)"
  PYTHONPATH=${TIMM_PATH}:src .venvs/sft/bin/python scripts/second_model_gate.py embed --model $m --tag train \
    --photos artifacts/value/train_informative_setft_own.npy 2>&1 | grep --line-buffered -v Warning >> artifacts/value/${m}_train.log || exit 1
done
echo "$(date +%T) all done"
