#!/usr/bin/env bash
# The current best pipeline (HYPOTHESIS_PLAN.md, docs/pipeline_overview.png) on a photo set that has none of its inputs yet, e.g. the benchmark
# eval halves (TAG=full; run only when the user asks for a test-set result). Steps: 4B clues (observe extract + queries), text search of the clues,
# zero-shot 4B clue judge, 4B place proposals, Pinpoint areas + restricted searches, clue / VLM hypotheses, comparator features with the set layer
# (needs pinpoint_set extract + apply first: artifacts/pinpoint_set/set_probs_$TAG.npz).
# Usage: TAG=full setsid nohup bash scripts/full_pipeline.sh > artifacts/agent/logs/full_pipeline_full.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."  # repo root
tag=${TAG:-full}
export HF_HUB_OFFLINE=1 PYTHONPATH=src
need() { [ -s "$1" ] || { echo "missing $1"; exit 1; }; }  # the VLM scripts exit 0 even when their python step fails
py() { systemd-run --user --scope --quiet -p MemoryMax=${MEM:-20G} -p MemorySwapMax=0 env PYTHONPATH=src .venv/bin/python -m "$@"; }

echo "$(date +%T) clues"
[ -s artifacts/agent/observe_${tag}_4b.json ] || NODE=extract NAME=4b TAGS=$tag bash scripts/observe.sh || exit 1
need artifacts/agent/observe_${tag}_4b.json
[ -s artifacts/agent/queries_${tag}_4b.json ] || NODE=queries NAME=4b TAGS=$tag bash scripts/observe.sh || exit 1
need artifacts/agent/queries_${tag}_4b.json
echo "$(date +%T) text search"
py geo_search_env.experiment.text_search stage || exit 1
[ -s artifacts/agent/region_evidence_${tag}_4b.json ] || py geo_search_env.experiment.region_evidence search --tag $tag --clues 4b || exit 1
echo "$(date +%T) clue judge"
[ -s artifacts/agent/clue_judge_${tag}_4b_4b.json ] || NAME=4b TAG=$tag CLUES=4b bash scripts/clue_judge.sh || exit 1
need artifacts/agent/clue_judge_${tag}_4b_4b.json
echo "$(date +%T) 4B place proposals"
[ -s artifacts/agent/vlm_places_${tag}_4b.json ] || RUN="vlm_hypotheses propose" NAME=4b TAG=$tag bash scripts/clue_judge.sh || exit 1
need artifacts/agent/vlm_places_${tag}_4b.json
echo "$(date +%T) hypotheses"
[ -s artifacts/agent/hypotheses_${tag}.npz ] || py geo_search_env.experiment.hypotheses run --tag $tag --k 5 || exit 1
[ -s artifacts/agent/hypotheses_${tag}_cluehyps.npz ] || py geo_search_env.experiment.hypotheses clue-hyps --tag $tag --judge 4b || exit 1
[ -s artifacts/agent/hypotheses_${tag}_vlmhyps.npz ] || py geo_search_env.experiment.vlm_hypotheses search --tag $tag || exit 1
echo "$(date +%T) features"
py geo_search_env.experiment.hypothesis_compare features --tag $tag --e4 --extra-hyps vlmhyps --set-dump artifacts/pinpoint_set/set_probs_${tag}.npz || exit 1
echo "$(date +%T) all done"
