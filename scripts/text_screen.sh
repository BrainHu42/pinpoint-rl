#!/usr/bin/env bash
# Text screen: a VLM transcribes legible text for the dev and val photos (vLLM), the strings are searched (SigLIP2 text, Wikipedia),
# then the evidence screen. Usage: [SCREENS="text-9b:Qwen3.5-9B text-4b:Qwen3.5-4B"] bash scripts/text_screen.sh
#   (needs an idle GPU; each "variant:model" pair serves that model; stops its own server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
mkdir -p artifacts/query_evidence/logs
for pair in ${SCREENS:-text-9b:Qwen3.5-9B}; do
  variant=${pair%%:*}; name=${pair##*:}
  model=$(ls -d /data/hf/hub/models--Qwen--$name/snapshots/* | head -1)
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
  PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve $model --served-model-name vlm --port 8765 --host 127.0.0.1 \
    --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}' \
    --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_$variant.log 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
  for tag in dev val; do
    echo "$(date +%T) read text $variant $tag"
    .venv/bin/python -m geo_search_env.experiment.stage1_eval places --tag $tag --variant $variant 2>&1
  done
  kill $server; wait $server 2>/dev/null || true
  for tag in dev val; do
    echo "$(date +%T) evaluate $variant $tag"
    .venv/bin/python -m geo_search_env.experiment.stage1_eval evaluate --tag $tag --variant $variant > artifacts/query_evidence/logs/eval_${variant}_${tag}.log 2>&1
  done
  echo "$(date +%T) screen $variant"
  .venv/bin/python -m geo_search_env.experiment.evidence_screen --variant $variant 2>&1
done
echo "$(date +%T) done"
