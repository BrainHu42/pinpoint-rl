#!/usr/bin/env bash
# Per-area name likelihood instead of a choice (names_judge pmi) on lesson 106's 202 named A/B dev3k pairs: Qwen3.8-27B-FP8, Gemma 3 27B, the 4B.
# Usage: setsid nohup bash scripts/names_judge_pmi.sh > artifacts/agent/logs/names_judge_pmi.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export TAG=dev3k RUN="names_judge ask"
q27=$(ls -d data/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
g27=$(ls -d data/hf/hub/models--unsloth--gemma-3-27b-it-FP8-Dynamic/snapshots/*/ | head -1)
MODEL=$q27 NAME=27b_pmi bash scripts/clue_judge.sh
PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.names_judge report --name 27b_pmi 2>&1
MODEL=$g27 MM_KWARGS='{}' NAME=gemma27b_pmi bash scripts/clue_judge.sh
PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.names_judge report --name gemma27b_pmi 2>&1
NAME=4b_pmi bash scripts/clue_judge.sh
PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.names_judge report --name 4b_pmi 2>&1
