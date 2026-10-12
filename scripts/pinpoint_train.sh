#!/bin/bash
# Fine-tune Pinpoint's reranker (PINPOINT_SET_PLAN.md step 4, lesson 70). Run A (+3.1 / +3.3):
#   bash scripts/pinpoint_train.sh --run-name setft_a --batch-size 256 --steps 6000 --val-every 1500 --val-batches 40 --log-every 50 --num-workers 4 --lr 5e-5 --warmup-steps 300
# Drop --set-head below for the control (+2.1 / +2.3). Run inside a memory-capped scope (MemoryMax=13G); needs ~27 GB of GPU memory.
cd /home/brianhu/pinpoint-rl/.claude/worktrees/geolocation-clues
IDX=data/pinpoint/pinpoint-submission/exp/contrastive_retrieval/checkpoints/attention_retrieval_index/cabeb0568af3b2f1
export PYTHONPATH=pinpoint_train/src:src PYTHONUNBUFFERED=1
data/pinpoint-main-src/pinpoint-main/.venv/bin/python pinpoint_train/src/pinpoint/attention_reranker/train.py \
  --contrastive-checkpoint-path data/pinpoint/pinpoint-submission/exp/contrastive_retrieval/checkpoints/ckpt_best.pt \
  --retrieval-index-dir $IDX --osv-support-cache-root $IDX/osv_support_cache --use-osv-support-token \
  --mp16-cache-root data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384 --val-pct 1 \
  --num-neighbors-image 32 --num-neighbors-raw-image 8 --num-neighbors-gps 8 --modality-pool-multiplier 1 --original-targets 8 2 2 \
  --reranker-dropout 0.2 --reranker-sequence-chunk-size 16384 --retrieval-chunk-size 131072 \
  --set-head --init-checkpoint data/pinpoint/pinpoint-submission/exp/attention_reranker/checkpoints/ckpt_best.pt \
  --exclude-author-path artifacts/pinpoint_train/author_by_cache_row.npy --experiment-dir artifacts/pinpoint_train "$@"
