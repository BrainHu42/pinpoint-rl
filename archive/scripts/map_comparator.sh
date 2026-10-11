#!/usr/bin/env bash
# Score the map search's top-100 gallery photos per dev photo (map_search.py comparator-pairs, tag maptop100_dev) with a merged comparator, then report it as a
# verifier (geo_match.py report --verifier). Waits for scripts/near_multi_exemplar.sh to finish first (one vLLM server at a time).
# Usage: RUN=comparator-d setsid nohup bash scripts/map_comparator.sh > artifacts/query_evidence/logs/map_comparator.log 2>&1 &   (~10 min of GPU)
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:-comparator-d}
until grep -qE "all done|giving up|Traceback" artifacts/query_evidence/logs/near_multi_exemplar.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) score"
RUN=$run TAGS=maptop100_dev TAG=maptop100_dev NO_REPORT=1 bash scripts/multi_exemplar_full.sh 2>&1 || exit 1
echo "$(date +%T) report"
~/.venvs/match/bin/python -m geo_search_env.experiment.geo_match report --verifier $run 2>&1 | grep -v -i warn
echo "$(date +%T) all done"
