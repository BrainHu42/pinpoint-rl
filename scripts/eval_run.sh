#!/usr/bin/env bash
# Merge a LoRA run, serve it with vLLM, and evaluate: 300-query benchmark subset (SFT prompt) + 1,000 MP16 val photos.
# Usage: bash scripts/eval_run.sh <run> [temperature, default 0.7] [prompt, default sft] [val data, default artifacts/sft/sft.jsonl]
#   (needs an idle GPU; stops its own server by PID)
set -eu
run=$1
temp=${2:-0.7}
prompt=${3:-sft}
data=${4:-artifacts/sft/sft.jsonl}
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
mkdir -p artifacts/sft/logs
[ -d /data/pinpoint/sft/$run/merged ] || ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run $run
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 5; done
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve /data/pinpoint/sft/$run/merged --served-model-name vlm $run --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization ${VLLM_MEM:-0.7} --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/sft/logs/vllm_$run.log 2>&1 &
server=$!
trap 'kill $server' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
.venv/bin/python -m geo_search_env.experiment.pivot_diagnostics vlm_sampling --model $run --prompt $prompt --temperature $temp
.venv/bin/python -m geo_search_env.experiment.sft_train val_eval --model $run --temperature $temp --data $data
