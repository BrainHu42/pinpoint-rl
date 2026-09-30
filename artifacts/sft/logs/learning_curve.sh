#!/usr/bin/env bash
# Learning curve: train and evaluate SFT on 15k and all 34,523 training examples (5k was the pilot).
set -eu
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
for spec in 15000:sft-15k 34523:sft-34k; do
  n=${spec%%:*}; run=${spec##*:}
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 30; done
  echo "$(date +%T) train $run ($n examples)"
  ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --examples $n --run $run > artifacts/sft/logs/train_$run.log 2>&1
  echo "$(date +%T) eval $run"
  bash artifacts/sft/logs/eval_run.sh $run > artifacts/sft/logs/eval_$run.log 2>&1
done
echo "$(date +%T) done"
