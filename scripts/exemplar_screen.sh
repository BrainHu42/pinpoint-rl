#!/usr/bin/env bash
# Exemplar screen: the Qwen3.5-9B judges "same place?" for the query photo against an exemplar of the right and of the wrong top-1 candidate.
# Usage: bash scripts/exemplar_screen.sh   (needs `exemplar_judge pairs` done and an idle GPU; stops its own server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
model=$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/* | head -1)
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":2}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_exemplar.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
echo "$(date +%T) judge"
.venv/bin/python -m geo_search_env.experiment.exemplar_judge judge 2>&1
kill $server; wait $server 2>/dev/null || true
echo "$(date +%T) report"
.venv/bin/python -m geo_search_env.experiment.exemplar_judge report 2>&1
echo "$(date +%T) done"
