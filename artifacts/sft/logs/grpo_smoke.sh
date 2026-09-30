#!/usr/bin/env bash
# After the retrieval SFT run finishes: 3 GRPO steps from it, to check memory, speed and rendering before the pilot.
set -eu
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
until grep -q "done" artifacts/sft/logs/retrieval_run.log 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/.venvs/sft/bin/python -m geo_search_env.experiment.grpo_train train --init sft-34k-retrieval --data artifacts/sft/sft_retrieval.jsonl \
  --run grpo-smoke --max-steps 3
