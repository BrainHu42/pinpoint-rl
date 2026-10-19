#!/usr/bin/env bash
# Serve a VLM and judge every specific text-search clue of a photo set (clue_judge.py judge), then the report. RUN="<module> <args>" runs
# that module command against the server instead (no report).
# Usage: [MODEL=<dir>] [PIXELS=786432] [LIMIT_MM=1 (images per prompt)] [NAME=4b] [TAG=dev3k] [CLUES=9b] [VS=1] [RUN=...] setsid nohup bash scripts/clue_judge.sh > artifacts/agent/logs/clue_judge.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampler JIT-compiles and this machine has no CUDA toolkit
model=${MODEL:-data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
name=${NAME:-4b}
tag=${TAG:-dev3k}
clues=${CLUES:-9b}
mkdir -p artifacts/agent/logs
mm_kwargs=${MM_KWARGS:-}  # image processor options; Qwen's max_pixels by default, MM_KWARGS='{}' for models without it (Gemma 3)
[ -z "$mm_kwargs" ] && mm_kwargs="{\"max_pixels\": ${PIXELS:-786432}}"
PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "$model" --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 \
  --gpu-memory-utilization ${GPU_UTIL:-0.9} --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt "{\"image\":${LIMIT_MM:-1}}" \
  --mm-processor-kwargs "$mm_kwargs" > artifacts/agent/logs/vllm_clue_judge_$name.log 2>&1 &
server=$!
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died"; exit 1; }; sleep 5; done
run=${RUN:-clue_judge judge}
echo "$(date +%T) $run $tag ($name)"
if [ -z "${RUN:-}" ]; then
  .venv/bin/python -m geo_search_env.experiment.$run --tag $tag --name $name --clues $clues ${VS:+--vs} 2>&1
else
  .venv/bin/python -m geo_search_env.experiment.$run --tag $tag --name $name 2>&1
fi
kill $server; wait $server 2>/dev/null; server=""
[ -z "${RUN:-}" ] && .venv/bin/python -m geo_search_env.experiment.clue_judge report --tag $tag --name $name --clues $clues 2>&1
echo "$(date +%T) done"
