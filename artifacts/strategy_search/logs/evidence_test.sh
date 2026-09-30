#!/usr/bin/env bash
# Serve base Qwen3.5-4B (multi-image) and run the evidence test: none, photos, places on the full eval halves.
set -eu
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
model=/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 5; done
PATH=/home/brian/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.7 --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt '{"image":11}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/strategy_search/logs/vllm_evidence_test.log 2>&1 &
server=$!
trap 'kill $server' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
for e in ${EVIDENCE:-none places photos}; do
  echo "$(date +%T) evidence=$e"
  .venv/bin/python -m geo_search_env.experiment.evidence_test run --evidence $e 2>&1 | grep -vE "^loading|^  [0-9]"
done
echo "$(date +%T) done"
