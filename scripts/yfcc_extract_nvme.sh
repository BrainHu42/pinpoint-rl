#!/usr/bin/env bash
# Finish the YFCC100M extraction from the copy on the local NVMe instance store (/opt/dlami/nvme; the EBS copy's 1 KB-page reads are IOPS-bound at
# ~12 MB/s). Parts already written (data/yfcc100m/geo_partNN.npz) are kept; the rest run 4 at a time (4 CPUs), parts 0-7 first. Then the early read
# on parts 0-7 (geo_first8) and the full merge (geo).
# Usage: setsid nohup bash scripts/yfcc_extract_nvme.sh > artifacts/agent/logs/yfcc_extract_nvme.log 2>&1 &   (after the NVMe copy finishes)
cd "$(dirname "$0")/.."  # repo root
export PYTHONPATH=src
db=/opt/dlami/nvme/yfcc100m/yfcc100m_dataset.sql
todo=""
for part in $(seq 0 15); do
  [ -s data/yfcc100m/geo_part$(printf %02d $part).npz ] || todo="$todo $part"
done
echo "$(date +%T) parts to extract:$todo"
echo $todo | tr ' ' '\n' | xargs -P 4 -I{} systemd-run --user --scope --quiet -p MemoryMax=5G -p MemorySwapMax=0 \
  .venv/bin/python -m geo_search_env.experiment.yfcc_names extract --part {} --parts 16 --db $db &
extract=$!
for part in 00 01 02 03 04 05 06 07; do  # early read as soon as the first half exists
  until [ -s data/yfcc100m/geo_part$part.npz ]; do sleep 20; done
done
sleep 30  # let the last of those files finish writing
echo "$(date +%T) parts 0-7 done: early read"
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/python -m geo_search_env.experiment.yfcc_names merge --only 0 1 2 3 4 5 6 7 --geo geo_first8
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 .venvs/overture/bin/python -m geo_search_env.experiment.yfcc_names match --tag dev3k --geo geo_first8
for acc in 0 11; do
  systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/python -m geo_search_env.experiment.yfcc_names report --tag dev3k --geo geo_first8 --min-accuracy $acc
done
wait $extract
echo "$(date +%T) all parts done: full merge"
systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 .venv/bin/python -m geo_search_env.experiment.yfcc_names merge --parts 16 --geo geo
echo "$(date +%T) all done"
