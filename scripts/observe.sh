#!/usr/bin/env bash
# AGENT_DIRECTION.md observe step: serve a VLM (default the base Qwen3.5-4B at full resolution) and extract structured clues for every photo of
# the given sets (observe.py extract), then the reports for the MP16 sets.
# Usage: [NODE=extract|crops] [BASE=9b] [MODEL=<dir>] [NAME=base] [TAGS="dev val full"] [MAX_PIXELS=786432] setsid nohup bash scripts/observe.sh > artifacts/agent/logs/observe.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampler JIT-compiles and this machine has no CUDA toolkit
model=${MODEL:-data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
name=${NAME:-base}
mkdir -p artifacts/agent/logs

PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "$model" --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS:-786432}}" > artifacts/agent/logs/vllm_observe_$name.log 2>&1 &
server=$!
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died"; exit 1; }; sleep 5; done

for tag in ${TAGS:-dev val full}; do
  echo "$(date +%T) extract $tag"
  .venv/bin/python -m geo_search_env.experiment.observe ${NODE:-extract} --tag $tag --name $name --base ${BASE:-9b} 2>&1
done
kill $server; wait $server 2>/dev/null; server=""
for tag in ${TAGS:-dev val full}; do
  .venv/bin/python -m geo_search_env.experiment.observe report --tag $tag --name $name 2>&1
done
echo "$(date +%T) all done"
