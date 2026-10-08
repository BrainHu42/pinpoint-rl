#!/usr/bin/env bash
# Score up to 4 exemplars per local candidate (near_miss.py multi-exemplar-pairs) of the held-out train photos, dev and the benchmark photos with a merged
# comparator, then report the combiners (near_miss.py combiners).
# Usage: RUN=comparator-d setsid nohup bash scripts/near_multi_exemplar.sh > artifacts/query_evidence/logs/near_multi_exemplar.log 2>&1 &
# (pairs ~15 min CPU; scoring waits for an idle GPU, ~1.5-2 h)
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN:-comparator-d}
echo "$(date +%T) pairs"
[ -f artifacts/query_evidence/multi_exemplar_nearmiss4_full_pairs.json ] || .venv/bin/python -m geo_search_env.experiment.near_miss multi-exemplar-pairs 2>&1 || exit 1
echo "$(date +%T) score"
RUN=$run TAGS="nearmiss4_trainhold nearmiss4_dev nearmiss4_full" TAG=nearmiss4_dev NO_REPORT=1 bash scripts/multi_exemplar_full.sh 2>&1 || exit 1
echo "$(date +%T) combiners"
.venv/bin/python -m geo_search_env.experiment.near_miss combiners --name $run 2>&1
echo "$(date +%T) all done"
