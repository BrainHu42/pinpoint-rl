#!/usr/bin/env bash
# KNOWLEDGE_PLAN.md step-1 check: base 4B (with and without a format example) and the knowledge-SFT model name the place for every
# full-eval-half photo; then the report. Both models see 448x448 images (the knowledge model's training size). ~30 min.
# Usage: setsid nohup bash scripts/knowledge_check.sh > artifacts/knowledge/logs/knowledge_check.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampler JIT-compiles and this machine has no CUDA toolkit
run=${RUN:-knowledge-a}
base=data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
mkdir -p artifacts/knowledge/logs

serve() {  # $1 model dir, $2 log name
  PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "$1" --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
    --mm-processor-kwargs '{"max_pixels": 200704}' > artifacts/knowledge/logs/vllm_check_$2.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died ($2)"; exit 1; }; sleep 5; done
}
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT

echo "$(date +%T) base"
serve $base base
.venv/bin/python -m geo_search_env.experiment.knowledge_check answer --name base 2>&1
.venv/bin/python -m geo_search_env.experiment.knowledge_check answer --name base --hint 2>&1
kill $server; wait $server 2>/dev/null; server=""

echo "$(date +%T) $run"
serve data/pinpoint/sft/$run/merged $run
.venv/bin/python -m geo_search_env.experiment.knowledge_check answer --name $run 2>&1
kill $server; wait $server 2>/dev/null; server=""

echo "$(date +%T) report"
.venv/bin/python -m geo_search_env.experiment.knowledge_check report 2>&1
echo "$(date +%T) all done"
