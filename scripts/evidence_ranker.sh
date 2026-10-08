#!/usr/bin/env bash
# Learned-chooser test of evidence: the base 4B names places for the MP16 train photos (vLLM), then evidence retrieval and the
# reranker fits (GPU, no server). Needs `evidence_ranker select` done and `dev` / `val` results cached (stage1.sh).
# Usage: [MODEL=<weights dir>] bash scripts/evidence_ranker.sh   (needs an idle GPU; stops its own server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
model=${MODEL:-/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
if [ ! -f artifacts/query_evidence/train/places.json ]; then
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
    --dtype bfloat16 --gpu-memory-utilization 0.5 --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image":1}' \
    --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_train.log 2>&1 &
  server=$!
  trap 'kill $server 2>/dev/null || true' EXIT
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
  echo "$(date +%T) places train"
  .venv/bin/python -m geo_search_env.experiment.stage1_eval places --tag train
  kill $server; wait $server 2>/dev/null || true
fi
echo "$(date +%T) retrieve"
.venv/bin/python -m geo_search_env.experiment.evidence_ranker retrieve
echo "$(date +%T) fit"
.venv/bin/python -m geo_search_env.experiment.evidence_ranker fit
echo "$(date +%T) done"
