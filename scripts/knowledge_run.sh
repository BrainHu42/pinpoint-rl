#!/usr/bin/env bash
# archive/KNOWLEDGE_PLAN.md step 1: knowledge dataset (no burned-in-GPS filter: rare at this scale, user 2026-10-06), LoRA SFT on photo -> "country > region > city"
# at 448x448, merge. Reuses artifacts/knowledge/selected.json (250k photos). ~3.5 h on the L40S (~20 photos/s training).
# Usage: setsid nohup bash scripts/knowledge_run.sh > artifacts/knowledge/logs/knowledge_run.log 2>&1 &
set -uo pipefail
# Run inside a memory-capped cgroup: if the job outgrows 24 GB it is killed, not the machine (30 GB RAM, no swap).
if [ -z "${IN_SCOPE:-}" ]; then
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:-knowledge-a}
pixels=200704  # 448 x 448, also used to serve the trained model
mkdir -p artifacts/knowledge/logs

.venv/bin/python -m geo_search_env.experiment.knowledge_data dataset --no-overlay 2>&1 || exit 1

echo "$(date +%T) train"
.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --run $run --data artifacts/knowledge/knowledge.jsonl --examples 1000000 \
  --batch 8 --accumulation 2 --no-checkpointing --max-pixels $pixels --dropout 0 --compile --pad-to 240 2>&1 || exit 1
echo "$(date +%T) merge"
.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run $run 2>&1 || exit 1
echo "$(date +%T) all done"
