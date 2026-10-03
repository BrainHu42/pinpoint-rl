#!/usr/bin/env bash
# The scaling diagnostic with each candidate's nearby Wikipedia articles in the prompt (4B and 9B with vLLM; optionally the 27B with llama.cpp: WITH_27B=1).
# Usage: bash scripts/knowledge_wiki.sh   (needs wiki_nearby.py done, the no-evidence runs of knowledge_scaling.sh, and an idle GPU; stops its servers by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run() { .venv/bin/python -m geo_search_env.experiment.knowledge_scaling "$@" 2>&1; }
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
for pair in qwen3.5-4b:$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1) qwen3.5-9b:$(ls -d /data/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/* | head -1); do
  name=${pair%%:*}; model=${pair#*:}
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 6144 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_wiki.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
  for tag in dev val; do echo "$(date +%T) $name wiki $tag"; run run --name $name-wiki --tag $tag --wiki; done
  kill $server; wait $server 2>/dev/null || true
done
names="qwen3.5-4b qwen3.5-4b-wiki qwen3.5-9b qwen3.5-9b-wiki"
if [ "${WITH_27B:-0}" = 1 ]; then
  bash /home/brian/.claude/jobs/1d1d34f6/tmp/serve27b.sh > artifacts/query_evidence/logs/llama_wiki.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8766/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
  for tag in dev val; do echo "$(date +%T) qwen3.6-27b wiki $tag"; run run --name qwen3.6-27b-wiki --tag $tag --server http://127.0.0.1:8766 --max-tokens 900 --wiki; done
  kill $server; wait $server 2>/dev/null || true
  names="$names qwen3.6-27b qwen3.6-27b-wiki"
fi
echo "$(date +%T) report"
run report --names $names
echo "$(date +%T) done"
