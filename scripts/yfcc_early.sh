#!/usr/bin/env bash
# Early read of the YFCC100M name lookup: wait for extract parts 0-7 (half of the photoid range), merge them as geo_first8, match the dev3k phrases,
# report (all matches, and only city-level-or-finer GPS). The full extraction keeps running; rerun match / report with --geo geo afterwards.
# Usage: setsid nohup bash scripts/yfcc_early.sh > artifacts/agent/logs/yfcc_early.log 2>&1 &
cd "$(dirname "$0")/.."  # repo root
export PYTHONPATH=src
for part in 00 01 02 03 04 05 06 07; do
  until [ -s data/yfcc100m/geo_part$part.npz ] && ! pgrep -f "extract --part ${part#0} --parts" > /dev/null; do sleep 30; done
done
echo "$(date +%T) parts 0-7 done"
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python -m geo_search_env.experiment.yfcc_names merge --only 0 1 2 3 4 5 6 7 --geo geo_first8 || exit 1
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 .venvs/overture/bin/python -m geo_search_env.experiment.yfcc_names match --tag dev3k --geo geo_first8 || exit 1
for acc in 0 11; do
  systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python -m geo_search_env.experiment.yfcc_names report --tag dev3k --geo geo_first8 --min-accuracy $acc
done
echo "$(date +%T) all done"
