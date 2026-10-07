#!/usr/bin/env bash
# Can the knowledge model's place knowledge pick better among the retrieval candidates than the reranker? (go/no-go for KNOWLEDGE_PLAN.md step 2)
# For the knowledge-SFT model and the base 4B (control), both at 448 px: greedy answers on MP16 dev (for knowledge_check rule) and candidate-name scores
# with the knowledge prompt on dev and the full eval halves (name_score --levels 3); then the reports: rule and combiner chosen / fitted on dev, applied
# once to the full eval halves. ~45 min.
# Usage: setsid nohup bash scripts/knowledge_ceiling.sh > artifacts/knowledge/logs/knowledge_ceiling.log 2>&1 &
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
    --mm-processor-kwargs '{"max_pixels": 200704}' > artifacts/knowledge/logs/vllm_ceiling_$2.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "vLLM died ($2)"; exit 1; }; sleep 5; done
}
trap '[ -n "${server:-}" ] && kill $server 2>/dev/null' EXIT

for model in "$run" base; do
  echo "$(date +%T) $model"
  if [ "$model" = base ]; then serve $base base; else serve data/pinpoint/sft/$model/merged $model; fi
  .venv/bin/python -m geo_search_env.experiment.knowledge_check answer --name $model --tag dev 2>&1
  for tag in dev full; do
    .venv/bin/python -m geo_search_env.experiment.name_score score --tag $tag --name ${model}_l3 --levels 3 2>&1
  done
  kill $server; wait $server 2>/dev/null; server=""
done

for model in "$run" base; do
  echo "$(date +%T) report $model"
  .venv/bin/python -m geo_search_env.experiment.knowledge_check rule --name $model 2>&1
  .venv/bin/python -m geo_search_env.experiment.name_score report --tag dev --name ${model}_l3 --levels 3 2>&1
  .venv/bin/python -m geo_search_env.experiment.name_score transfer --fit-tag dev --tag full --name ${model}_l3 --levels 3 2>&1
done
echo "$(date +%T) all done"
