#!/usr/bin/env bash
# Idea 1 (VALUE_PLAN.md, lesson V4): serve the base Qwen3.5-4B and read the visible text of every candidate area's index photos
# (scripts/gallery_text_check.py read), then the report.
# Usage: [TAG=dev3k_setft_own] setsid nohup bash scripts/gallery_text.sh > artifacts/value/gallery_text.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src:${EXTRA_PYTHONPATH:-}
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampler JIT-compiles and this machine has no CUDA toolkit
model=data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
tag=${TAG:-dev3k_setft_own}
mkdir -p artifacts/value

PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "$model" --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 \
  --gpu-memory-utilization ${GPU_UTIL:-0.6} --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/value/vllm_gallery_text.log 2>&1 &
server=$!
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died"; exit 1; }; sleep 5; done

echo "$(date +%T) read $tag"
.venv/bin/python scripts/gallery_text_check.py read --tag $tag 2>&1
kill $server; wait $server 2>/dev/null; server=""
.venv/bin/python scripts/gallery_text_check.py report --tag $tag 2>&1
echo "$(date +%T) all done"
