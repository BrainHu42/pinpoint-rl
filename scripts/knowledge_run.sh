#!/usr/bin/env bash
# Knowledge SFT check: wait for the MP16 place labels, select PHOTOS photos (knowledge_data select), filter burned-in GPS with the base VLM, write the data,
# LoRA-train the 4B on photo -> "country > region > city > neighbourhood" at 448x448, merge, then score dev / val candidate names with the trained model
# and with the base model at the same image size (name_score.py; the go / no-go numbers).
# Usage: setsid nohup bash scripts/knowledge_run.sh > artifacts/query_evidence/logs/knowledge_run.log 2>&1 &   (~7.5 h: ~6.4 h training at ~11 photos/s, needs ~27 GB)
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:-knowledge-a}
photos=${PHOTOS:-250000}
pixels=200704  # 448 x 448, also used to serve the trained model
base=/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
mkdir -p artifacts/query_evidence/logs
until grep -q "mp16 4122118/4122118" artifacts/place_labels/label_all.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) select"
.venv/bin/python -m geo_search_env.experiment.knowledge_data select --photos $photos 2>&1 || exit 1

echo "$(date +%T) overlay filter (base VLM)"
server=""
trap '[ -z "$server" ] || kill $server 2>/dev/null || true' EXIT
for attempt in $(seq 1 40); do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $base --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels": 786432}' \
    > artifacts/query_evidence/logs/vllm_knowledge_overlay.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null || ! kill -0 $server 2>/dev/null; do sleep 3; done
  if curl -sf 127.0.0.1:8765/health >/dev/null; then break; fi
  echo "$(date +%T) server failed to start (attempt $attempt), waiting for the GPU"
  wait $server 2>/dev/null || true
  server=""
  sleep 30
done
[ -n "$server" ] || { echo "giving up: no server"; exit 1; }
.venv/bin/python -m geo_search_env.experiment.knowledge_data overlay 2>&1
kill $server; wait $server 2>/dev/null || true
server=""
.venv/bin/python -m geo_search_env.experiment.knowledge_data dataset 2>&1 || exit 1

free_mib() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }
for attempt in 1 2 3 4 5 6; do
  until [ "$(free_mib)" -gt 28000 ]; do sleep 30; done
  echo "$(date +%T) train attempt $attempt"
  if ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --run $run --data artifacts/knowledge/knowledge.jsonl --examples 1000000 \
      --batch 8 --accumulation 2 --no-checkpointing --max-pixels $pixels 2>&1; then
    break
  fi
  echo "$(date +%T) training failed, retrying after a wait"
  sleep 120
done
[ -d /data/pinpoint/sft/$run/adapter ] || { echo "no finished training"; exit 1; }
echo "$(date +%T) merge"
~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run $run 2>&1
echo "$(date +%T) score trained model"
GPU_UTIL=0.85 RUN=$run NAME=$run MAX_PIXELS=$pixels TAGS="dev val" bash scripts/name_scores.sh 2>&1
echo "$(date +%T) score base model at the same image size (the control, on the same candidate names)"
GPU_UTIL=0.85 RUN=base NAME=base448 MAX_PIXELS=$pixels TAGS="dev val" bash scripts/name_scores.sh 2>&1
echo "$(date +%T) all done"
