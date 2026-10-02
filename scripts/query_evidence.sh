#!/usr/bin/env bash
# Serve base Qwen3.5-4B (multi-image) and run the query-evidence loop: select, generate, retrieve, answer, report.
# Usage: [TAG=dev] [N=1000] [SOURCE=mp16|bench] [STAGES="answer report"] [MODEL=<weights dir>] bash scripts/query_evidence.sh   (needs an idle GPU; stops its own server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
tag=${TAG:-dev}
model=${MODEL:-/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
run() { .venv/bin/python -m geo_search_env.experiment.query_evidence "$1" --tag $tag --n ${N:-1000} --source ${SOURCE:-mp16} 2>&1; }
mkdir -p artifacts/query_evidence/$tag
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 5; done
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.5 --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt '{"image":8}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/$tag/vllm.log 2>&1 &
server=$!
trap 'kill $server' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
for node in ${STAGES:-select generate retrieve answer report}; do
  echo "$(date +%T) $node"
  run $node
done
echo "$(date +%T) done"
