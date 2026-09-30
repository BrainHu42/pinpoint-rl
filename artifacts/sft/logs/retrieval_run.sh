#!/usr/bin/env bash
# No reranker: all pooled candidates in retrieval order with evidence; train on all 34,523 examples and evaluate.
set -eu
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
run=sft-34k-retrieval
echo "$(date +%T) train $run"
~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --examples 34523 --run $run --data artifacts/sft/sft_retrieval.jsonl > artifacts/sft/logs/train_$run.log 2>&1
echo "$(date +%T) eval $run"
bash artifacts/sft/logs/eval_run.sh $run 0.7 sft-retrieval artifacts/sft/sft_retrieval.jsonl > artifacts/sft/logs/eval_$run.log 2>&1
echo "$(date +%T) done"
