#!/usr/bin/env bash
# comparator-d (queued behind comparator-c): continue comparator-c on the large near-band set (near_miss.py near-pairs-full: every local candidate of every
# non-held-out train photo as a row, ~219k rows, one pass, ~7 h at 8.7 pairs/s), merge, score the same three candidate sets and report as for comparator-c.
# Usage: setsid nohup bash scripts/comparator_d.sh > artifacts/query_evidence/logs/comparator_d.log 2>&1 &   (waits for scripts/comparator_c.sh to finish; ~7.6 h in all)
# It starts if comparator-c's training finished; it does not look at comparator-c's results. To cancel: kill this script's PID before it starts training.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=comparator-d
until grep -qE "all done|no finished training" artifacts/query_evidence/logs/comparator_c.log 2>/dev/null; do sleep 60; done
[ -f /data/pinpoint/sft/comparator-c/held_out.json ] || { echo "$(date +%T) comparator-c did not finish training: not starting"; exit 1; }
free_mib() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }
for attempt in 1 2 3 4 5 6; do
  until [ "$(free_mib)" -gt 16000 ]; do sleep 30; done
  echo "$(date +%T) train attempt $attempt"
  if ~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run $run --mode near-full --samples 219000 --batch 16 --accumulation 1 \
      --init-adapter /data/pinpoint/sft/comparator-c/adapter 2>&1; then
    break
  fi
  echo "$(date +%T) training failed, retrying after a wait"
  sleep 120
done
[ -f /data/pinpoint/sft/$run/held_out.json ] || { echo "no finished training"; exit 1; }
echo "$(date +%T) merge"
~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run $run 2>&1
echo "$(date +%T) score"
RUN=$run TAGS="nearmiss_dev nearmiss_trainhold nearmiss_full" TAG=nearmiss_dev NO_REPORT=1 bash scripts/multi_exemplar_full.sh 2>&1
echo "$(date +%T) final report"
.venv/bin/python -m geo_search_env.experiment.near_miss final-report --name $run 2>&1
echo "$(date +%T) all done"
