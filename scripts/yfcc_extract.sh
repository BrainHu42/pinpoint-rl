#!/usr/bin/env bash
# YFCC100M geotagged titles / tags: 16 photoid-range parts, 8 at a time (one sequential SQLite scan of the 1 KB-page file is latency-bound at ~12 MB/s),
# each in its own 3 GB memory scope; then merge into data/yfcc100m/geo.npz.
# Usage: setsid nohup bash scripts/yfcc_extract.sh > artifacts/agent/logs/yfcc_extract.log 2>&1 &
cd "$(dirname "$0")/.."  # repo root
export PYTHONPATH=src
seq 0 15 | xargs -P 8 -I{} systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 \
  .venv/bin/python -m geo_search_env.experiment.yfcc_names extract --part {} --parts 16
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python -m geo_search_env.experiment.yfcc_names merge --parts 16
echo "$(date +%T) all done"
