#!/usr/bin/env bash
# Serve the Qwen3.5-9B and have it describe photos with coarse attributes. Usage: [TAGS="dev val train"] [LIMIT=60] bash scripts/photo_attributes.sh
#   (needs an idle GPU; stops its own server by PID; LIMIT writes a small pilot file instead of the full one)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
model=$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/* | head -1)
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin/vllm:$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_photo_attributes.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
for tag in ${TAGS:-dev}; do
  echo "$(date +%T) photo attributes $tag"
  .venv/bin/python -m geo_search_env.experiment.photo_attributes run --tag $tag ${LIMIT:+--limit $LIMIT} 2>&1
done
echo "$(date +%T) done"
