#!/usr/bin/env bash
# comparator-b: continue comparator-a on the multi-exemplar pairs (comparator_data.py multi), merge, then score 4 exemplars per dev / val top-8 candidate.
# Usage: bash scripts/comparator_b.sh   (~5 h on the shared GPU; logs in artifacts/query_evidence/logs/)
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
mkdir -p artifacts/query_evidence/logs
echo "$(date +%T) train"
~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run comparator-b --mode multi --samples 0 --batch 16 --accumulation 1 \
  --init-adapter /data/pinpoint/sft/comparator-a/adapter 2>&1
echo "$(date +%T) merge"
~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run comparator-b 2>&1
echo "$(date +%T) evaluate"
RUN=comparator-b bash scripts/multi_exemplar.sh 2>&1
echo "$(date +%T) all done"
