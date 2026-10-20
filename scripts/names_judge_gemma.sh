#!/usr/bin/env bash
# Gemma 3 27B (unsloth FP8-Dynamic) on lesson 106's 202 named A/B dev3k pairs: plain first-token A/B, lesson 106's evidence-first format,
# and step-by-step reasoning first (Gemma 3 has no thinking mode). Then the three reports.
# Usage: setsid nohup bash scripts/names_judge_gemma.sh > artifacts/agent/logs/names_judge_gemma.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export MODEL=$(ls -d data/hf/hub/models--unsloth--gemma-3-27b-it-FP8-Dynamic/snapshots/*/ | head -1)
export MM_KWARGS='{}' TAG=dev3k RUN="names_judge ask"
for name in gemma27b gemma27b_reason gemma27b_think; do
  NAME=$name bash scripts/clue_judge.sh || exit 1
done
for name in gemma27b gemma27b_reason gemma27b_think 27b 27b_reason; do
  echo "=== $name"
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.names_judge report --name $name 2>&1
done
