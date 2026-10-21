#!/usr/bin/env bash
# Step B of the combined-head plan: the whole setft_own pipeline on the commons_train pool (commons_bench with COMMONS_OUT, after select +
# download), so the answer head and text adapter can train on Commons photos. Every step skips when its output exists (resumable).
# Steps: release files, SigLIP2 cache, released-Pinpoint pools, photo set, 4B clues + queries, text search, clue judge, setft_own candidates and
# photo set, areas / clue places / features / zooms, MegaLoc columns, combined_head text searches + features.
# Usage: setsid nohup bash scripts/commons_train_pipeline.sh > artifacts/agent/logs/commons_train_pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src COMMONS_OUT=data/pinpoint/commons_train
ckpt=artifacts/pinpoint_train/setft_own/checkpoints/ckpt_final.pt
py() { systemd-run --user --scope --quiet -p MemoryMax=${MEM:-20G} -p MemorySwapMax=0 env PYTHONPATH=src COMMONS_OUT=$COMMONS_OUT \
  PINPOINT_RERANKER_CKPT=${CKPT_ENV:-} .venv/bin/python -m "$@"; }
need() { [ -s "$1" ] || { echo "missing $1"; exit 1; }; }
tag=commons_train
a=artifacts/agent

echo "$(date +%T) release files + SigLIP2"
[ -s $COMMONS_OUT/release/benchmark.csv ] || py geo_search_env.experiment.commons_bench trainset || exit 1
[ -s $COMMONS_OUT/release/image_embeddings/google_siglip2-giant-opt-patch16-384/embeddings.f16.bin ] || \
  py geo_search_env.experiment.embed_cache benchmark $tag || exit 1
echo "$(date +%T) released-Pinpoint pools + photo set"
[ -s artifacts/pinpoint_reference/$tag.npz ] || py geo_search_env.experiment.pinpoint_reference $tag || exit 1
[ -s artifacts/photo_sets/$tag/photos.json ] || py geo_search_env.experiment.photo_sets select --source $tag --tag $tag || exit 1
echo "$(date +%T) clues"
[ -s $a/observe_${tag}_4b.json ] || NODE=extract NAME=4b TAGS=$tag bash scripts/observe.sh || exit 1
need $a/observe_${tag}_4b.json
[ -s $a/queries_${tag}_4b.json ] || NODE=queries NAME=4b TAGS=$tag bash scripts/observe.sh || exit 1
need $a/queries_${tag}_4b.json
echo "$(date +%T) text search"
py geo_search_env.experiment.text_search stage || exit 1
[ -s $a/region_evidence_${tag}_4b.json ] || py geo_search_env.experiment.region_evidence search --tag $tag --clues 4b || exit 1
echo "$(date +%T) clue judge"
[ -s $a/clue_judge_${tag}_4b_4b.json ] || NAME=4b TAG=$tag CLUES=4b bash scripts/clue_judge.sh || exit 1
need $a/clue_judge_${tag}_4b_4b.json
echo "$(date +%T) setft_own candidates + photo set"
export CKPT_ENV=$ckpt
[ -s artifacts/pinpoint_set/${tag}_32_8_8_setft_own.npz ] || py geo_search_env.experiment.pinpoint_set extract --name $tag --ckpt-tag setft_own || exit 1
[ -s artifacts/photo_sets/${tag}_setft_own/photos.json ] || py geo_search_env.experiment.pinpoint_set photo-set --tags $tag --ckpt-tag setft_own || exit 1
echo "$(date +%T) areas, clue places, features, zooms"
BASES=$tag CKPT_TAG=setft_own CKPT=$ckpt bash scripts/retrained_pipeline.sh || exit 1
echo "$(date +%T) MegaLoc + text searches"
[ -s $a/megaloc_${tag}_setft_own.npz ] || py geo_search_env.experiment.megaloc_gate features --tag $tag --ckpt-tag setft_own &
megaloc=$!
MEM=12G py geo_search_env.experiment.combined_head gather --tag $tag || exit 1
wait $megaloc || exit 1
need $a/megaloc_${tag}_setft_own.npz
py geo_search_env.experiment.combined_head features --tag $tag || exit 1
echo "$(date +%T) all done"
