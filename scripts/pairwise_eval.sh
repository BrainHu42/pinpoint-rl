#!/usr/bin/env bash
# Evaluate a pairwise comparator (query + two exemplars -> which matches) on the dev and val top-4 candidates: merge the LoRA, serve it, score
# every ordered pair, report. Usage: RUN=pairwise-a bash scripts/pairwise_eval.sh   (stops its own server by PID)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:?set RUN}
[ -d /data/pinpoint/sft/$run/merged ] || ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run $run
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
mkdir -p artifacts/query_evidence/logs
PATH=$HOME/.venvs/vllm/bin:$PATH vllm serve /data/pinpoint/sft/$run/merged --served-model-name vlm --port 8765 --host 127.0.0.1 \
  --dtype bfloat16 --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image":3}' \
  --mm-processor-kwargs '{"max_pixels": 786432}' > artifacts/query_evidence/logs/vllm_$run.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf 127.0.0.1:8765/health >/dev/null; do kill -0 $server || exit 1; sleep 3; done
echo "$(date +%T) pair-judge $run"
.venv/bin/python -m geo_search_env.experiment.exemplar_judge pair-judge --name $run 2>&1
kill $server; wait $server 2>/dev/null || true
echo "$(date +%T) pair-report"
.venv/bin/python -m geo_search_env.experiment.exemplar_judge pair-report --name $run 2>&1
echo "$(date +%T) done"
