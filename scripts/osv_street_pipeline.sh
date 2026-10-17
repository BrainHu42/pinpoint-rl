#!/usr/bin/env bash
# The street-search action's inputs (OSV-5M areas, variant moreosv) on the setft_own pipeline: OSV top 50 per photo, OSV clusters outside
# the existing areas -> restricted Pinpoint searches, features, zoom states, MegaLoc columns (reusing the base run's). Then
#   HYP_VARIANT=moreosv action_agent --ckpt-tag setft_own --one-km 0.5 --apply commons26dev:setft_own
# Usage: BASES="dev3k commons26dev train" bash scripts/osv_street_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
ckpt_tag=setft_own
export PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/$ckpt_tag/checkpoints/ckpt_final.pt
py() { systemd-run --user --scope --quiet -p MemoryMax=${MEM:-24G} -p MemorySwapMax=0 env PYTHONPATH=src PINPOINT_KEEP_OWN=1 PINPOINT_RERANKER_CKPT=$PINPOINT_RERANKER_CKPT .venv/bin/python -m "$@"; }
for base in ${BASES:-dev3k commons26dev train}; do
  tag=${base}_${ckpt_tag}
  echo "$(date +%T) $tag: OSV search"
  [ -s artifacts/osv_propose/osv_${tag}.npz ] || py geo_search_env.experiment.osv_propose osv-search --tag $tag || exit 1
  echo "$(date +%T) $tag: OSV areas + restricted searches"
  [ -s artifacts/agent/hypotheses_${tag}_moreosv.npz ] || py geo_search_env.experiment.osv_propose hyps --tag $tag || exit 1
  echo "$(date +%T) $tag: features"
  [ -s artifacts/agent/hypothesis_features_${tag}_e4_moreosv_set.npz ] || py geo_search_env.experiment.hypothesis_compare features --tag $tag --e4 \
    --set-dump artifacts/pinpoint_set/set_probs_${tag}.npz --extra-hyps moreosv || exit 1
  echo "$(date +%T) $tag: zoom states"
  [ -s artifacts/pinpoint_set/zoom_${tag}_moreosv.npz ] || py geo_search_env.experiment.pinpoint_set restricted --tag $tag --extra-hyps moreosv --model none || exit 1
  echo "$(date +%T) $tag: MegaLoc"
  [ -e artifacts/photo_sets/${tag}_moreosv ] || ln -s $tag artifacts/photo_sets/${tag}_moreosv
  [ -s artifacts/agent/megaloc_${base}_${ckpt_tag}_moreosv.npz ] || py geo_search_env.experiment.megaloc_gate features --tag $base \
    --ckpt-tag ${ckpt_tag}_moreosv --base $ckpt_tag || exit 1
done
echo "$(date +%T) all done"
