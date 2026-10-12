#!/usr/bin/env bash
# The hypothesis pipeline (areas + clue places, zooms, area model inputs) on a retrained Pinpoint reranker (lesson 70; merged-pipeline step 1),
# for train and dev3k. Needs pinpoint_set extract --ckpt-tag $CKPT_TAG, then pinpoint_set photo-set --ckpt-tag $CKPT_TAG.
# Usage: CKPT_TAG=setft_a CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt bash scripts/retrained_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
ckpt_tag=${CKPT_TAG:-setft_a}
export PINPOINT_RERANKER_CKPT=${CKPT:-artifacts/pinpoint_train/$ckpt_tag/checkpoints/ckpt_final.pt}
py() { systemd-run --user --scope --quiet -p MemoryMax=${MEM:-20G} -p MemorySwapMax=0 env PYTHONPATH=src PINPOINT_RERANKER_CKPT=$PINPOINT_RERANKER_CKPT .venv/bin/python -m "$@"; }
for base in dev3k train; do
  tag=${base}_${ckpt_tag}
  echo "$(date +%T) $tag: Pinpoint areas + restricted searches"
  [ -s artifacts/agent/hypotheses_${tag}.npz ] || py geo_search_env.experiment.hypotheses run --tag $tag --k 5 || exit 1
  echo "$(date +%T) $tag: clue places"
  [ -s artifacts/agent/hypotheses_${tag}_cluehyps.npz ] || py geo_search_env.experiment.hypotheses clue-hyps --tag $tag --judge 4b || exit 1
  echo "$(date +%T) $tag: features"
  py geo_search_env.experiment.hypothesis_compare features --tag $tag --e4 --set-dump artifacts/pinpoint_set/set_probs_${tag}.npz || exit 1
  echo "$(date +%T) $tag: zoom states"
  [ -s artifacts/pinpoint_set/zoom_${tag}.npz ] || py geo_search_env.experiment.pinpoint_set restricted --tag $tag --extra-hyps --model none || exit 1
done
echo "$(date +%T) all done"
