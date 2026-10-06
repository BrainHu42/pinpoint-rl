#!/usr/bin/env bash
# Score the top 50 raw-neighbour clusters of every dev / val photo with a fine-tuned comparator, then report (see deep_rerank.py).
# Usage: RUN=comparator-a bash scripts/deep_rerank.sh   (needs `deep_rerank build` and the merged adapter; stops its server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:-comparator-a}
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve /data/pinpoint/sft/$run/merged --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 96 --limit-mm-per-prompt '{"image":2}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_deep_$run.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
echo "$(date +%T) judge"
.venv/bin/python -m geo_search_env.experiment.deep_rerank judge 2>&1
kill $server; wait $server 2>/dev/null || true
echo "$(date +%T) report"
.venv/bin/python -m geo_search_env.experiment.deep_rerank report 2>&1
echo "$(date +%T) done"
