#!/usr/bin/env bash
# comparator-c: continue comparator-b on near-band pairs (positives < 1 km from the truth, negatives 1-5 / 5-25 / >= 25 km; near_miss.py near-pairs), merge, score the
# local candidates of the held-out train photos, dev and the benchmark photos, then report the combiner results (near_miss.py final-report).
# Usage: setsid nohup bash scripts/comparator_c.sh > artifacts/query_evidence/logs/comparator_c.log 2>&1 &   (~2 h training + ~35 min scoring on the shared GPU)
# The GPU is shared: wait for ~14 GB to be free before training, and retry the training from the start if another job's memory use kills it.
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=comparator-c
free_mib() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }
for attempt in 1 2 3 4 5 6; do
  until [ "$(free_mib)" -gt 16000 ]; do sleep 30; done
  echo "$(date +%T) train attempt $attempt"
  if ~/.venvs/sft/bin/python -m geo_search_env.experiment.comparator_train train --run $run --mode near --samples 0 --batch 16 --accumulation 1 \
      --init-adapter /data/pinpoint/sft/comparator-b/adapter 2>&1; then
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
