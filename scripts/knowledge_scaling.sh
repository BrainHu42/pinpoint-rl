#!/usr/bin/env bash
# Zero-shot chooser by model size: Qwen3.5-4B and 9B with vLLM, Qwen3.6-27B (Q4_K_M) with llama.cpp, on the dev and val photos with the reranker's
# top-10 as text. Usage: bash scripts/knowledge_scaling.sh   (needs an idle GPU; stops its own servers by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run() { .venv/bin/python -m geo_search_env.experiment.knowledge_scaling "$@" 2>&1; }
serve_vllm() {  # $1 model dir
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $1 --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_scaling.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
}
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
for pair in qwen3.5-4b:$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1) qwen3.5-9b:$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/* | head -1); do
  name=${pair%%:*}; model=${pair#*:}
  serve_vllm $model
  for tag in dev val; do echo "$(date +%T) $name $tag"; run run --name $name --tag $tag; done
  kill $server; wait $server 2>/dev/null || true
done
echo "$(date +%T) qwen3.6-27b (llama.cpp)"
bash /home/brian/.claude/jobs/1d1d34f6/tmp/serve27b.sh > artifacts/query_evidence/logs/llama_scaling.log 2>&1 &
server=$!
until curl -sf 127.0.0.1:8766/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
for tag in dev val; do echo "$(date +%T) qwen3.6-27b $tag"; run run --name qwen3.6-27b --tag $tag --server http://127.0.0.1:8766 --max-tokens 900; done
kill $server; wait $server 2>/dev/null || true
echo "$(date +%T) report"
run report --names qwen3.5-4b qwen3.5-9b qwen3.6-27b
echo "$(date +%T) done"
