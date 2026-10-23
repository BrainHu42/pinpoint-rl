#!/usr/bin/env bash
# PE-G embedding of any gallery directory at any resolution (pe_retrieval.py embed), resumable, in a memory-capped cgroup.
# Usage: setsid nohup bash scripts/pe_embed.sh <size> <dir> [<dir> ...] >> <log> 2>&1 &
#   e.g. the 224 px check: bash scripts/pe_embed.sh 224 artifacts/retrieval/q224 artifacts/retrieval/b1_224
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # worktree root (artifacts/ and data/ are links into the main checkout)
size=$1
shift
for dir in "$@"; do
  echo "$(date '+%F %T %Z') embed $dir at $size px start"
  PYTHONPATH=/home/brianhu/pinpoint-rl/.venvs/pylib_pe:src \
    /home/brianhu/pinpoint-rl/.venvs/sft/bin/python scripts/pe_retrieval.py embed --out "$dir" --size "$size" 2>&1 | grep --line-buffered -v Warning
  echo "$(date '+%F %T %Z') embed $dir exit"
done
