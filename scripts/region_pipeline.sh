#!/usr/bin/env bash
# The merged pipeline plus region hypotheses (lesson 85; region_hyps.py): region candidates and their restricted searches, hypothesis features
# and zoom states with --extra-hyps regionhyps, for dev3k and train on the retrained reranker. Needs scripts/retrained_pipeline.sh's outputs.
# Then: HYP_VARIANT=regionhyps python -m geo_search_env.experiment.hypothesis_agent --ckpt-tag setft_a (REGION_K limits the regions kept).
# Usage: [BASES="dev3k train"] [REGION_K=16] bash scripts/region_pipeline.sh > artifacts/agent/logs/region_pipeline.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
ckpt_tag=${CKPT_TAG:-setft_a}
export PINPOINT_RERANKER_CKPT=${CKPT:-artifacts/pinpoint_train/$ckpt_tag/checkpoints/ckpt_final.pt} REGION_K=${REGION_K:-16}
py() { systemd-run --user --scope --quiet -p MemoryMax=${MEM:-24G} -p MemorySwapMax=0 env PYTHONPATH=src PINPOINT_RERANKER_CKPT=$PINPOINT_RERANKER_CKPT REGION_K=$REGION_K .venv/bin/python -m "$@"; }
for base in ${BASES:-dev3k train}; do
  tag=${base}_${ckpt_tag}
  echo "$(date +%T) $tag: region hypotheses"
  [ -s artifacts/agent/hypotheses_${tag}_regionhyps.npz ] || py geo_search_env.experiment.region_hyps build --tag $tag || exit 1
  echo "$(date +%T) $tag: features"
  [ -s artifacts/agent/hypothesis_features_${tag}_e4_regionhyps_set.npz ] || py geo_search_env.experiment.hypothesis_compare features --tag $tag --e4 \
    --extra-hyps regionhyps --set-dump artifacts/pinpoint_set/set_probs_${tag}.npz || exit 1
  echo "$(date +%T) $tag: zoom states"
  [ -s artifacts/pinpoint_set/zoom_${tag}_regionhyps.npz ] || py geo_search_env.experiment.pinpoint_set restricted --tag $tag --extra-hyps regionhyps --model none || exit 1
done
echo "$(date +%T) all done"
