#!/usr/bin/env bash
# Continue setft_own's reranker on bucket 99's train split (in-distribution: the retriever never saw these photos; dev3k photographers dropped,
# artifacts/retrieval/train99_ids.txt) vs the same steps on its usual buckets 0-98 (control), then score both on dev3k (retrieval_recall top1).
# Usage: bash scripts/reranker_b99.sh [STEPS] [LR]
set -euo pipefail
STEPS=${1:-600}; LR=${2:-2e-5}
INIT=artifacts/pinpoint_train/setft_own/checkpoints/ckpt_final.pt
args="--experiment-dir artifacts/retrieval/pinpoint_train --batch-size 256 --steps $STEPS --val-every 100000 --val-batches 1 --log-every 50
      --num-workers 2 --lr $LR --warmup-steps 50 --init-checkpoint $INIT"
for run in ret_b99_s${STEPS} ret_b99ctrl_s${STEPS}; do
  ids=""; [[ $run == ret_b99_* ]] && ids="--setenv=PINPOINT_TRAIN_IDS=artifacts/retrieval/train99_ids.txt"
  echo "== $run"
  systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 --setenv=EXCL=0 $ids \
    bash scripts/pinpoint_train.sh --run-name $run $args > artifacts/retrieval/$run.log 2>&1
  grep -E "PINPOINT_TRAIN_IDS|complete" artifacts/retrieval/$run.log || true
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --setenv=PYTHONPATH=src \
    --setenv=PINPOINT_RERANKER_CKPT=artifacts/retrieval/pinpoint_train/$run/checkpoints/ckpt_final.pt \
    .venv/bin/python -m geo_search_env.experiment.retrieval_recall top1 --tag ${run}_32_8_8 --targets 32 8 8 2>&1 | grep "top-1"
done
