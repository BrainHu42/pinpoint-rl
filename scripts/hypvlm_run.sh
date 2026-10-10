#!/usr/bin/env bash
# Fine-tuned hypothesis plausibility (HYPOTHESIS_PLAN.md): LoRA SFT of the 4B on "Could this photo have been taken in or near {place}?" over
# train_extra's E4 hypotheses (hypothesis_vlm sft-data; GPS labels, photos never evaluated), merge, then judge every hypothesis of train and
# dev3k with it at the training resolution (448 x 448). Then: hypothesis_compare features --vlm-name $run, hypothesis_learn.
# Usage: [RUN_NAME=hypvlm-4b] [DATA=artifacts/agent/hypvlm_sft.jsonl] setsid nohup bash scripts/hypvlm_run.sh > artifacts/agent/logs/hypvlm_run.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then  # memory-capped cgroup (30 GB RAM, no swap)
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."  # repo root
export HF_HUB_OFFLINE=1 PYTHONPATH=src
run=${RUN_NAME:-hypvlm-4b}
pixels=200704
echo "$(date +%T) train"
.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --run $run --data ${DATA:-artifacts/agent/hypvlm_sft.jsonl} --examples 1000000 \
  --batch 8 --accumulation 2 --no-checkpointing --max-pixels $pixels --dropout 0 --compile --pad-to 280 2>&1 || exit 1
echo "$(date +%T) merge"
.venvs/sft/bin/python -m geo_search_env.experiment.sft_train merge --run $run 2>&1 || exit 1
for tag in dev3k train; do
  echo "$(date +%T) judge $tag"
  IN_SCOPE=1 MODEL=data/pinpoint/sft/$run/merged PIXELS=$pixels NAME=$run TAG=$tag RUN="hypothesis_vlm judge" bash scripts/clue_judge.sh 2>&1 || exit 1
done
echo "$(date +%T) all done"
