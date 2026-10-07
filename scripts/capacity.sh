#!/usr/bin/env bash
# Capacity question (AGENT_DIRECTION.md): do larger open VLMs name the right region on the MP16 dev / val fixable photos, and read clues better?
# Per model: knowledge_check answers ("country > region > city") on dev and val, and observe.py v2 clues on dev and val; same image size for all.
# Usage: [MAX_PIXELS=786432] [SKIP_OBSERVE=1] MODELS="base4b=<dir> 9b=<dir>" setsid nohup bash scripts/capacity.sh > artifacts/agent/logs/capacity.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12  # Triton needs Python.h
export VLLM_USE_FLASHINFER_SAMPLER=0  # its sampler JIT-compiles and this machine has no CUDA toolkit
mkdir -p artifacts/agent/logs
server=""
trap '[ -n "$server" ] && kill $server 2>/dev/null' EXIT

for pair in $MODELS; do
  name=${pair%%=*}; model=${pair#*=}
  echo "$(date +%T) $name"
  PATH=$PWD/.venvs/vllm/bin:$PATH vllm serve "$model" --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
    --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS:-786432}}" > artifacts/agent/logs/vllm_capacity_$name.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died ($name)"; exit 1; }; sleep 5; done
  for tag in ${TAGS:-dev val}; do
    .venv/bin/python -m geo_search_env.experiment.knowledge_check answer --name $name --tag $tag 2>&1
    [ -n "${SKIP_OBSERVE:-}" ] || .venv/bin/python -m geo_search_env.experiment.observe extract --tag $tag --name $name 2>&1
  done
  kill $server; wait $server 2>/dev/null; server=""
done
echo "$(date +%T) all done"
