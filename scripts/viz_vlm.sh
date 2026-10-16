#!/usr/bin/env bash
# Serve Qwen3.5-4B (thinking off) for the pipeline viewer's live clue reading (viz.live reads $VIZ_VLM, default 127.0.0.1:8766; port 8765 is
# left to the experiment scripts). Small GPU share so the viewer's own models fit beside it.
# Usage: setsid nohup bash scripts/viz_vlm.sh > artifacts/viz/vlm.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0
model=${MODEL:-data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
PATH=$PWD/.venvs/vllm/bin:$PATH exec vllm serve "$model" --served-model-name vlm --port ${PORT:-8766} --host 127.0.0.1 --dtype bfloat16 \
  --gpu-memory-utilization ${GPU_UTIL:-0.3} --max-model-len 8192 --max-num-seqs 8 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}'
