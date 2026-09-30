#!/usr/bin/env bash
# Serve a merged run with vLLM and evaluate it on every im2gps3k + yfcc4k eval-half query (~3.7k).
# Usage: bash artifacts/sft/logs/full_eval.sh <run> <prompt> [temperature, default 0.7]   (needs an idle GPU)
set -eu
run=$1
prompt=$2
temp=${3:-0.7}
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 5; done
PATH=/home/brian/.venvs/vllm/bin:$PATH vllm serve /data/pinpoint/sft/$run/merged --served-model-name vlm $run --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization ${VLLM_MEM:-0.7} --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/sft/logs/vllm_full_$run.log 2>&1 &
server=$!
trap 'kill $server' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
.venv/bin/python -m geo_search_env.experiment.pivot_diagnostics vlm_sampling --model $run --prompt $prompt --temperature $temp --full-eval
