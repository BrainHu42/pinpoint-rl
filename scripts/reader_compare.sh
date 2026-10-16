#!/usr/bin/env bash
# Lesson 88 reader comparison: serve a VLM and run observe.py's three reader prompts (extract, queries, landmarks) on dev3k under one name.
# Usage: NAME=38_27b MODEL=data/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/<rev> setsid nohup bash scripts/reader_compare.sh \
#          > artifacts/agent/logs/reader_compare.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampler JIT-compiles and this machine has no CUDA toolkit
name=${NAME:?NAME}
mkdir -p artifacts/agent/logs

# FP8 checkpoints keep their own dtype; no MTP (speculative decoding off)
PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "${MODEL:?MODEL}" --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --gpu-memory-utilization 0.92 --max-model-len 8192 --max-num-seqs 48 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS:-786432}}" > artifacts/agent/logs/vllm_reader_$name.log 2>&1 &
server=$!
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died"; exit 1; }; sleep 5; done

for node in ${NODES:-extract queries landmarks}; do
  echo "$(date +%T) $node"
  .venv/bin/python -m geo_search_env.experiment.observe $node --tag dev3k --name $name 2>&1
done
kill $server; wait $server 2>/dev/null; server=""
echo "$(date +%T) all done"
