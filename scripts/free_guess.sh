#!/usr/bin/env bash
# Candidate-free guesses from the base Qwen3.5-4B on the dev and val photos, then the per-threshold combiner report.
# Usage: bash scripts/free_guess.sh   (needs an idle GPU; stops its own server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
model=$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1)
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_free.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
for tag in dev val; do echo "$(date +%T) $tag"; .venv/bin/python -m geo_search_env.experiment.free_guess --tag $tag 2>&1; done
kill $server; wait $server 2>/dev/null || true
echo "$(date +%T) report"
.venv/bin/python -m geo_search_env.experiment.threshold_headroom 2>&1
echo "$(date +%T) done"
