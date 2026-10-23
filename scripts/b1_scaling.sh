#!/usr/bin/env bash
# PE_PLAN step 7 follow-ups on the B1 gallery (378 px), before committing to B2: gallery density + both-models input, then training-data scaling.
# Usage: bash scripts/b1_scaling.sh   (logs in artifacts/retrieval/b1/)
set -uo pipefail
cd "$(dirname "$0")/.."
run() {  # memory-capped cgroup
  PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/pe_retrieval.py trained "$@" 2>&1 |
    grep --line-buffered -v "Warning\|interaction_encoder"
}
run --backbones pe_g,siglip2,concat --eval-fracs 0.25,0.5,1 --tag _density > artifacts/retrieval/b1/trained_density.log
for f in 0.25 0.5; do
  run --backbones pe_g --train-frac "$f" --tag "_train$f" > "artifacts/retrieval/b1/trained_train$f.log"
done
echo "b1 scaling done"
