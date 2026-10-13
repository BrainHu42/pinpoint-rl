#!/bin/bash
# Cross-fitting (lesson 75), after scripts/xfit_queue.sh: the merged pipeline's inputs for each half's trained-on query photos, built with the
# reranker that never trained on that half (photo set, Pinpoint areas + restricted searches, features, zoom states; no clue places).
cd /home/brianhu/pinpoint-rl/.claude/worktrees/geolocation-clues
py() { systemd-run --user --scope --quiet -p MemoryMax=${MEM:-20G} -p MemorySwapMax=0 env PYTHONPATH=src PINPOINT_RERANKER_CKPT=$CKPT .venv/bin/python -m "$@"; }
for pair in "1 xfit_a" "0 xfit_b"; do
  set -- $pair; half=$1; ckpt_tag=$2; tag=xfit${half}_$ckpt_tag
  CKPT=artifacts/pinpoint_train/$ckpt_tag/checkpoints/ckpt_final.pt
  echo "$(date +%T) $tag: photo set"
  [ -s artifacts/photo_sets/$tag/photos.json ] || py geo_search_env.experiment.pinpoint_set photo-set --ckpt-tag $ckpt_tag --tags xfit$half || exit 1
  echo "$(date +%T) $tag: Pinpoint areas + restricted searches"
  [ -s artifacts/agent/hypotheses_$tag.npz ] || py geo_search_env.experiment.hypotheses run --tag $tag --k 5 || exit 1
  echo "$(date +%T) $tag: features"
  [ -s artifacts/agent/hypothesis_features_${tag}_e4_set.npz ] || py geo_search_env.experiment.hypothesis_compare features --tag $tag --e4 --set-dump artifacts/pinpoint_set/set_probs_$tag.npz || exit 1
  echo "$(date +%T) $tag: zoom states"
  [ -s artifacts/pinpoint_set/zoom_$tag.npz ] || py geo_search_env.experiment.pinpoint_set restricted --tag $tag --extra-hyps --model none || exit 1
done
echo "$(date +%T) xfit pipeline done"
