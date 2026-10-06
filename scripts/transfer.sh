#!/usr/bin/env bash
# Copy the data, model weights, checkpoints and experiment caches this repo needs to another machine, at the same absolute paths (SETUP.md §0).
# The code itself moves with git (branch worktree-pivot-query-evidence); the Python environments are rebuilt from envs/*.txt.
# Usage: bash scripts/transfer.sh user@host [--dry-run]    (rsync over ssh; resumable, re-run to finish an interrupted copy)
#        OPTIONAL=1 bash scripts/transfer.sh user@host     also copies the Qwen3.5-9B / Qwen3.6-27B weights (knowledge_scaling only, 35 GB)
#        MERGED=1 ...                                      also copies merged SFT models (else re-create them with `sft_train merge --run <run>`)
# Remote repo path: DEST_REPO (default: the main checkout path here). Needs write access to /data and DEST_REPO on the target; ~510 GB in total.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
host=${1:?usage: transfer.sh user@host [--dry-run]}
shift
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")  # the main checkout (this may be a worktree)
dest_repo=${DEST_REPO:-$main}
pinpoint=${PINPOINT_ROOT:-/home/brian/workspace/pinpoint-submission/submission}
hub=/data/hf/hub
rs() { rsync -aR --mkpath --partial --info=progress2 "$@"; }

paths=(
  /data/hf/datasets/MP16-Pro                         # 371 GB: images (tar shards) + metadata; training photos and exemplars
  $hub/models--Qwen--Qwen3.5-4B                      # 9 GB: base model
  $hub/models--google--siglip2-giant-opt-patch16-384 # 7 GB: embeddings for new photos
  /data/pinpoint/mp16-embed                          # 37 GB: MP16 SigLIP2 gallery
  /data/pinpoint/osv5m-embed                         # 22 GB: OSV-5M SigLIP2 gallery
  /data/pinpoint/im2gps3k /data/pinpoint/yfcc4k /data/pinpoint/wikimedia  # benchmarks (images + embeddings)
  /data/pinpoint/geonames                            # 2 GB
  /data/pinpoint/overture                            # 24 GB: place index (places.sqlite) and division polygons (place labels)
  /data/pinpoint/wikipedia                           # 15 GB: offline geotagged Wikipedia (BM25 + dense)
  /data/pinpoint/geo                                 # 1 GB: WorldClim + ETOPO rasters
  $pinpoint/exp/contrastive_retrieval/checkpoints/ckpt_best.pt     # Pinpoint retriever (candidate pools)
  $pinpoint/exp/contrastive_retrieval/checkpoints/retrieval_index
)
[ "${OPTIONAL:-0}" = 1 ] && paths+=($hub/models--Qwen--Qwen3.5-9B $hub/models--unsloth--Qwen3.6-27B-GGUF)

# SFT runs: adapters and held-out lists; merged models (8.5 GB each) only with MERGED=1.
sft_excludes=(--exclude 'merged/')
[ "${MERGED:-0}" = 1 ] && sft_excludes=()

rs "$@" "${paths[@]}" "$host:/"
rs "$@" "${sft_excludes[@]}" /data/pinpoint/sft "$host:/"
# Experiment caches and results (~11 GB; artifacts/ is a symlink in worktrees, so copy its target). .env holds API keys.
rs "$@" --no-R "$(readlink -f artifacts)/" "$host:$dest_repo/artifacts/"
env_file=${ENV_FILE:-$main/.env}  # worktrees have no .env
[ ! -f "$env_file" ] || rs "$@" --no-R "$env_file" "$host:$dest_repo/.env"
echo "done; on $host: clone the repo to $dest_repo, check out worktree-pivot-query-evidence, then follow SETUP.md §0"
