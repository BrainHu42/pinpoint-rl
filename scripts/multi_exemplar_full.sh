#!/usr/bin/env bash
# Score up to 4 exemplars per top-8 candidate of every photo of a set with a merged comparator, then report with everything fitted on MP16 dev.
# Usage: RUN=comparator-b [TAG=wikimedia_balanced] [NO_REPORT=1] bash scripts/multi_exemplar_full.sh   (needs `multi_exemplar full-pairs --tag $TAG` and the dev / val scores of the same run; stops its server by PID)
# The GPU is shared: wait for it to be idle, and if another job grabs memory while our server loads (the server then dies at startup), wait and try again.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:?set RUN}
tag=${TAG:-full}  # full | wikimedia | wikimedia_balanced (needs `multi_exemplar full-pairs --tag $tag`)
mkdir -p artifacts/query_evidence/logs
server=""
trap '[ -z "$server" ] || kill $server 2>/dev/null || true' EXIT
for attempt in $(seq 1 40); do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve /data/pinpoint/sft/$run/merged --served-model-name vlm --port 8765 --host 127.0.0.1 \
    --dtype bfloat16 --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 64 --logprobs-mode processed_logprobs --limit-mm-per-prompt '{"image":2}' \
    --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_multi_full_$run.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null || ! kill -0 $server 2>/dev/null; do sleep 3; done
  if curl -sf 127.0.0.1:8765/health >/dev/null; then break; fi
  echo "$(date +%T) server failed to start (attempt $attempt), waiting for the GPU"
  wait $server 2>/dev/null || true
  server=""
  sleep 30
done
[ -n "$server" ] || { echo "giving up: no server"; exit 1; }
for t in ${TAGS:-$tag}; do  # TAGS: several photo sets scored in one server session
  echo "$(date +%T) full-judge $run $t"
  .venv/bin/python -m geo_search_env.experiment.multi_exemplar full-judge --name $run --tag $t 2>&1
done
kill $server; wait $server 2>/dev/null || true
server=""
[ -n "${NO_REPORT:-}" ] || echo "$(date +%T) full-report"
[ -n "${NO_REPORT:-}" ] || .venv/bin/python -m geo_search_env.experiment.multi_exemplar full-report --name $run --tag $tag 2>&1
echo "$(date +%T) done"
