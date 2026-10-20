#!/usr/bin/env bash
# commons26 dev base MegaLoc columns (recomputed: the earlier queries were blank images), then the street-search inputs for commons26 dev and train.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "$(date +%T) commons26dev base MegaLoc"
[ -s artifacts/agent/megaloc_commons26dev_setft_own.npz ] || systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env PYTHONPATH=src \
  .venv/bin/python -m geo_search_env.experiment.megaloc_gate features --tag commons26dev --ckpt-tag setft_own || exit 1
BASES="commons26dev train" bash scripts/osv_street_pipeline.sh
