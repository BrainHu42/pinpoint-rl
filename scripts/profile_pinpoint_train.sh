#!/bin/bash
# Profile reranker training with run A's arguments (scripts/pinpoint_train.sh) for a few steps; extra args are appended (later wins).
cd /home/brianhu/pinpoint-rl/.claude/worktrees/geolocation-clues
sed -e 's#pinpoint_train/src/pinpoint/attention_reranker/train.py#scripts/profile_pinpoint_train.py#' scripts/pinpoint_train.sh > artifacts/pinpoint_train/.profile_launcher.sh
bash artifacts/pinpoint_train/.profile_launcher.sh --run-name profile --batch-size 256 --steps ${STEPS:-60} --val-every ${STEPS:-60} --val-batches 1 \
  --log-every 10 --num-workers 4 --lr 5e-5 --warmup-steps 10 "$@"
