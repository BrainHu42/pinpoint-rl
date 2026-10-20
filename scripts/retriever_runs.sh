#!/usr/bin/env bash
# Retriever fine-tunes (retriever_train), their indexes and dev3k recall (retrieval_recall), one run after another.
# Usage: bash scripts/retriever_runs.sh RUN "TRAIN FLAGS" [RUN "TRAIN FLAGS" ...]
#   e.g. bash scripts/retriever_runs.sh ret_ctrl "--steps 5000" ret_ii02 "--steps 5000 --ii-weight 0.2"
# Needs the NVMe copies under /opt/dlami/nvme/retrieval (mp16, osv5m + prepare-osv).
set -euo pipefail
PP=data/pinpoint-main-src/pinpoint-main/.venv/bin/python
cap() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 "$@"; }
while [ $# -ge 2 ]; do
  run=$1; flags=$2; shift 2
  echo "== $run: train $flags"
  cap --setenv=PYTHONPATH=pinpoint_train/src:src $PP -m geo_search_env.experiment.retriever_train train --run "$run" $flags 2>&1 | grep -v Warn | grep -E "step [0-9]*000 |done|GPU"
  echo "== $run: index"
  cap --setenv=PYTHONPATH=pinpoint_train/src:src $PP -m geo_search_env.experiment.retriever_train index --run "$run" 2>&1 | grep -E "^->"
  echo "== $run: recall"
  cap --setenv=PYTHONPATH=src --setenv=PINPOINT_CONTRASTIVE_CKPT=artifacts/retrieval/ckpt/$run/ckpt_final.pt .venv/bin/python -m geo_search_env.experiment.retrieval_recall \
    ranks --photo-set dev3k --tag "$run" --index-dir artifacts/retrieval/index/$run 2>&1 | grep -E "^->"
  PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.retrieval_recall report --tag "$run" > /dev/null
done
