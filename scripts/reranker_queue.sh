#!/bin/bash
# Reranker training queue (2026-10-13): a full pass of run A's recipe, then the three ablations around it (6,000 steps each).
# Each run: train in a memory-capped scope, then extract its 32 / 8 / 8 pools for train + dev3k and score its own pick (pinpoint_set eval-head).
cd /home/brianhu/pinpoint-rl/.claude/worktrees/geolocation-clues
common="--batch-size 256 --val-every 1500 --val-batches 40 --log-every 50 --num-workers 4 --lr 5e-5 --warmup-steps 300"
train() {  # name steps env...
  local name=$1 steps=$2; shift 2
  [ -s artifacts/pinpoint_train/$name/checkpoints/ckpt_final.pt ] && return 0
  echo "$(date +%T) train $name ($steps steps, $*)"
  systemd-run --user --scope --quiet -p MemoryMax=13G -p MemorySwapMax=0 env "$@" bash scripts/pinpoint_train.sh --run-name $name --steps $steps $common \
    > artifacts/pinpoint_train/$name.log 2>&1 || { echo "$(date +%T) $name failed"; return 1; }
}
evaluate() {  # name [head]
  local name=$1 head=""
  [ "${2:-}" = head ] && head="--checkpoint artifacts/pinpoint_train/$name/checkpoints/ckpt_final.pt"
  echo "$(date +%T) extract + eval-head $name"
  for t in mp16; do
    systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 env PYTHONPATH=src \
      PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/$name/checkpoints/ckpt_final.pt \
      .venv/bin/python -m geo_search_env.experiment.pinpoint_set extract --name $t --targets 32 8 8 --ckpt-tag $name > artifacts/pinpoint_train/$name.extract.log 2>&1 || return 1
  done
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.pinpoint_set eval-head --data artifacts/pinpoint_set/mp16_32_8_8_$name.npz $head >artifacts/pinpoint_train/$name.eval.txt 2>&1
  tail -20 artifacts/pinpoint_train/$name.eval.txt
}
train setft_full 16000 POOL=wide EXCL=1 HEAD=1 && evaluate setft_full head
train abl_plain 6000 POOL=narrow EXCL=0 HEAD=0 && evaluate abl_plain
train abl_wide 6000 POOL=wide EXCL=0 HEAD=0 && evaluate abl_wide
train abl_excl 6000 POOL=narrow EXCL=1 HEAD=0 && evaluate abl_excl
echo "$(date +%T) queue done"
