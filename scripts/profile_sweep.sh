#!/bin/bash
# Throughput sweep for reranker training knobs (steady-state ms per step; scripts/profile_pinpoint_train.py, PROFILE_MODE=throughput).
cd /home/brianhu/pinpoint-rl/.claude/worktrees/geolocation-clues
run() {  # label args...
  local label=$1; shift
  rm -rf artifacts/pinpoint_train/profile
  systemd-run --user --scope --quiet -p MemoryMax=13G -p MemorySwapMax=0 env PROFILE_MODE=throughput STEPS=${STEPS:-100} \
    bash scripts/profile_pinpoint_train.sh "$@" > artifacts/pinpoint_train/profile_$label.log 2>&1
  echo "$label ($*): $(grep 'steady state' artifacts/pinpoint_train/profile_$label.log) $(grep -o 'cuda.*out of memory' artifacts/pinpoint_train/profile_$label.log | head -1)"
}
# SPECS: "label args...; label args...". Results (2026-10-13, 4 CPU cores, batch 256, network-disk index): base 399 ms, retrieval chunks
# 1M / 4M 387 / 393, batch 384 / 512 532 / 690 (~110 ms fixed per step + 1.14 ms per photo), NVMe index 366, 2 loader workers 368.
IFS=';' read -ra specs <<< "${SPECS:-base; chunk1m --retrieval-chunk-size 1048576; chunk4m --retrieval-chunk-size 4194304; bs384 --batch-size 384; bs512 --batch-size 512}"
for spec in "${specs[@]}"; do
  run $spec
done
