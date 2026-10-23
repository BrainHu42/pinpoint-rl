#!/usr/bin/env bash
# PE_PLAN.md step B1: PE-G embedding of the 10% gallery, resumable (rerun after the daily restart; it continues from
# artifacts/retrieval/b1/embed.progress). Outputs and timm live on EBS, so nothing needed here is on the wiped NVMe drive.
# Usage: setsid nohup bash scripts/b1_embed.sh >> artifacts/retrieval/b1/embed.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # worktree root (artifacts/ and data/ are links into the main checkout)
echo "$(date '+%F %T %Z') b1 embed start"
PYTHONPATH=/home/brianhu/pinpoint-rl/.venvs/pylib_pe:src \
  /home/brianhu/pinpoint-rl/.venvs/sft/bin/python scripts/pe_retrieval.py embed 2>&1 | grep --line-buffered -v Warning
echo "$(date '+%F %T %Z') b1 embed exit"
