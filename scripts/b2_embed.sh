#!/usr/bin/env bash
# PE_PLAN.md step B2: PE-G embedding of the full index (4.08M photos; reuses B1 rows), resumable (rerun after the daily restart; it continues from
# artifacts/retrieval/b2/done.npy). Outputs and timm live on EBS, so nothing needed here is on the wiped NVMe drive.
# Usage: setsid nohup bash scripts/b2_embed.sh >> artifacts/retrieval/b2/embed.log 2>&1 &
# Order: the dev3k pools' index rows, the train pools', then the rest. Extra args go to embed (later --first wins; --only-first stops there).
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # worktree root (artifacts/ and data/ are links into the main checkout)
echo "$(date '+%F %T %Z') b2 embed start"
PYTHONPATH=/home/brianhu/pinpoint-rl/.venvs/pylib_pe:src \
  /home/brianhu/pinpoint-rl/.venvs/sft/bin/python scripts/pe_retrieval.py embed --out artifacts/retrieval/b2 --reuse artifacts/retrieval/b1 \
    --first artifacts/retrieval/b2/first_dev3k_pool.npy,artifacts/retrieval/b2/first_train_pool.npy "$@" 2>&1 | grep --line-buffered -v Warning
echo "$(date '+%F %T %Z') b2 embed exit"
