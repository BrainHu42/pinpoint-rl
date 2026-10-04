#!/usr/bin/env bash
# Score up to 4 exemplars per top-8 candidate of every im2gps3k / yfcc4k eval-half photo with a merged comparator, then report with everything fitted on MP16 dev.
# Usage: RUN=comparator-b [TAG=wikimedia_balanced] bash scripts/multi_exemplar_full.sh   (needs `multi_exemplar full-pairs` and the dev / val scores of the same run; stops its server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:?set RUN}
tag=${TAG:-full}  # full | wikimedia | wikimedia_balanced (needs `multi_exemplar full-pairs --tag $tag`)
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve /data/pinpoint/sft/$run/merged --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 64 --logprobs-mode processed_logprobs --limit-mm-per-prompt '{"image":2}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_multi_full_$run.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
echo "$(date +%T) full-judge $run"
.venv/bin/python -m geo_search_env.experiment.multi_exemplar full-judge --name $run --tag $tag 2>&1
kill $server; wait $server 2>/dev/null || true
echo "$(date +%T) full-report"
.venv/bin/python -m geo_search_env.experiment.multi_exemplar full-report --name $run --tag $tag 2>&1
echo "$(date +%T) done"
