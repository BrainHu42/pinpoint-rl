#!/usr/bin/env bash
# Score candidate place names with a VLM (name_score.py score), then report. RUN=base serves the base Qwen3.5-4B; any other RUN serves /data/pinpoint/sft/$RUN/merged.
# Usage: RUN=base TAG=dev setsid nohup bash scripts/name_scores.sh > artifacts/query_evidence/logs/name_scores_base_dev.log 2>&1 &
# The GPU is shared: wait for it to be idle, and retry the server start if another job grabs memory while it loads. Stops its server by PID.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:-base}
tag=${TAG:-dev}
name=${NAME:-$run}  # scores are saved under this name (e.g. base448 for the base model at MAX_PIXELS=200704)
pixels=${MAX_PIXELS:-786432}  # the image size cap the model was trained with
model=/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
[ "$run" = base ] || model=/data/pinpoint/sft/$run/merged
server=""
trap '[ -z "$server" ] || kill $server 2>/dev/null || true' EXIT
for attempt in $(seq 1 40); do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 --gpu-memory-utilization ${GPU_UTIL:-0.6} \
    --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs "{\"max_pixels\": $pixels}" \
    > artifacts/query_evidence/logs/vllm_name_scores_$run.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null || ! kill -0 $server 2>/dev/null; do sleep 3; done
  if curl -sf 127.0.0.1:8765/health >/dev/null; then break; fi
  echo "$(date +%T) server failed to start (attempt $attempt), waiting for the GPU"
  wait $server 2>/dev/null || true
  server=""
  sleep 30
done
[ -n "$server" ] || { echo "giving up: no server"; exit 1; }
for t in ${TAGS:-$tag}; do
  echo "$(date +%T) score $run $t"
  .venv/bin/python -m geo_search_env.experiment.name_score score --tag $t --name $name 2>&1
done
kill $server; wait $server 2>/dev/null || true
server=""
for t in ${TAGS:-$tag}; do
  echo "$(date +%T) report $run $t"
  .venv/bin/python -m geo_search_env.experiment.name_score report --tag $t --name $name 2>&1
done
echo "$(date +%T) all done"
