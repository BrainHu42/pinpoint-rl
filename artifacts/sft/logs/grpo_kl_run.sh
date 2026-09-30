#!/usr/bin/env bash
# GRPO retune after the pilot's collapse: lr 5e-6, KL 0.04 to the SFT policy, T=1.0, 5 hours (stops early on entropy
# collapse), from sft-34k-retrieval; then merge the final adapter and run the 300-subset/MP16-val and full eval-half evals.
set -eu
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
logs=artifacts/sft/logs
data=artifacts/sft/sft_retrieval.jsonl
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
echo "$(date +%T) train grpo-kl"
PATH=$HOME/.venvs/grpo/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train train \
  --init sft-34k-retrieval --data $data --run grpo-kl --hours 5 > $logs/train_grpo-kl.log 2>&1
~/.venvs/grpo/bin/python -m geo_search_env.experiment.grpo_train merge --init sft-34k-retrieval --run grpo-kl >> $logs/train_grpo-kl.log 2>&1
echo "$(date +%T) eval grpo-kl"
bash $logs/eval_run.sh grpo-kl 0.7 sft-retrieval $data > $logs/eval_grpo-kl.log 2>&1
bash $logs/full_eval.sh grpo-kl sft-retrieval > $logs/full_eval_grpo-kl.log 2>&1
echo "$(date +%T) done"
