#!/usr/bin/env bash
# Landmark / object identification with a specialised descriptor (lesson 58 follow-up): top-100 SigLIP2 crop search over MP16 without keypoint
# verification, then DINOv2-large re-scoring of those hits, then the dev report. ~30-45 min on the L40S.
# Usage: [TAG=dev3k] setsid nohup bash scripts/landmark_rescore.sh > artifacts/agent/logs/landmark_rescore.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src HF_HOME=$PWD/data/hf
tag=${TAG:-dev3k}
echo "$(date +%T) search"
.venvs/sft/bin/python -m geo_search_env.experiment.landmark_id search --tag $tag --top-k 100 --no-verify 2>&1 || exit 1
echo "$(date +%T) rescore"
.venvs/sft/bin/python -m geo_search_env.experiment.landmark_id rescore --tag $tag --top-k 100 2>&1 || exit 1
echo "$(date +%T) report"
.venv/bin/python -m geo_search_env.experiment.landmark_id report-rescore --tag $tag --top-k 100 2>&1
echo "$(date +%T) all done"
