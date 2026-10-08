#!/usr/bin/env bash
# Serve a VLM and judge every specific text-search clue of a photo set (clue_judge.py judge), then the report. RUN="vlm_country sample" runs
# that module command against the server instead (no report).
# Usage: [MODEL=<dir>] [NAME=4b] [TAG=dev3k] [RUN=...] setsid nohup bash scripts/clue_judge.sh > artifacts/agent/logs/clue_judge.log 2>&1 &
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
mkdir -p artifacts/agent/logs
PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "$model" --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/agent/logs/vllm_clue_judge_$name.log 2>&1 &
server=$!
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died"; exit 1; }; sleep 5; done
run=${RUN:-clue_judge judge}
echo "$(date +%T) $run $tag ($name)"
.venv/bin/python -m geo_search_env.experiment.$run --tag $tag --name $name 2>&1
kill $server; wait $server 2>/dev/null; server=""
[ -z "${RUN:-}" ] && .venv/bin/python -m geo_search_env.experiment.clue_judge report --tag $tag --name $name 2>&1
echo "$(date +%T) done"
