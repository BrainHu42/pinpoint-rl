#!/bin/bash
# Cross-fitting (lesson 75): rerankers xfit_a / xfit_b fine-tuned on photographer halves 0 / 1 (scripts/xfit_halves.py; run A's recipe), then
# each scores the other half's trained-on query photos (wide 32 / 8 / 8 pools with CLS states). Resumable: finished steps are skipped.
cd /home/brianhu/pinpoint-rl/.claude/worktrees/geolocation-clues
common="--batch-size 256 --steps 6000 --val-every 1500 --val-batches 40 --log-every 50 --num-workers 4 --lr 5e-5 --warmup-steps 300"
for pair in "xfit_a 0 1" "xfit_b 1 0"; do
  set -- $pair; name=$1; train_half=$2; score_half=$3
  if [ ! -s artifacts/pinpoint_train/$name/checkpoints/ckpt_final.pt ]; then
    rm -rf artifacts/pinpoint_train/$name
    echo "$(date +%T) train $name on photographer half $train_half"
    systemd-run --user --scope --quiet -p MemoryMax=13G -p MemorySwapMax=0 env PINPOINT_TRAIN_ROW_MASK=artifacts/pinpoint_train/xfit_half${train_half}_rows.npy \
      bash scripts/pinpoint_train.sh --run-name $name $common > artifacts/pinpoint_train/$name.log 2>&1 || { echo "$(date +%T) $name failed"; exit 1; }
  fi
done
for pair in "xfit_a 1" "xfit_b 0"; do
  set -- $pair; name=$1; half=$2
  out=artifacts/pinpoint_set/xfit${half}_32_8_8_$name.npz
  [ -s $out ] && continue
  echo "$(date +%T) extract half $half with $name"
  systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 env PYTHONPATH=src \
    PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/$name/checkpoints/ckpt_final.pt \
    .venv/bin/python -m geo_search_env.experiment.pinpoint_set extract --name xfit$half --targets 32 8 8 --ckpt-tag $name \
    > artifacts/pinpoint_train/$name.extract_xfit$half.log 2>&1 || { echo "$(date +%T) extract $name failed"; exit 1; }
done
echo "$(date +%T) xfit queue done"
