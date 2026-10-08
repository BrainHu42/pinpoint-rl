#!/usr/bin/env bash
# Stage-1 baseline: the base 4B names three places per photo (vLLM), then search + oracle accuracy (GPU, no server).
# Usage: [TAGS="dev val"] [VARIANTS="v2 v2b"] [MODEL=<weights dir>] bash scripts/stage1.sh   (needs an idle GPU; stops its own server by PID)
# The photo lists must exist: `query_evidence select --tag dev` and `select --source bench --tag val`.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
tags=${TAGS:-dev val}
model=${MODEL:-/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
run() { .venv/bin/python -m geo_search_env.experiment.stage1_eval "$@" 2>&1; }
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 5; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.5 --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_stage1.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
for v in ${VARIANTS:-v2}; do for tag in $tags; do echo "$(date +%T) places $v $tag"; run places --tag $tag --variant $v; done; done
kill $server; wait $server 2>/dev/null || true
for v in ${VARIANTS:-v2}; do for tag in $tags; do echo "$(date +%T) evaluate $v $tag"; run evaluate --tag $tag --variant $v; done; done
echo "$(date +%T) done"
