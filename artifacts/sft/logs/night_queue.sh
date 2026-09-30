#!/usr/bin/env bash
# Overnight queue (after grpo_smoke.sh passes): full eval-half evals of both SFT models, a gradient-checkpointing speed
# test, then a 2-hour GRPO pilot from whichever SFT model has the higher best-of-8, with the same evals.
set -u
cd /home/brian/workspace/pinpoint-rl
export HF_HUB_OFFLINE=1
logs=artifacts/sft/logs
idle() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done; }

echo "$(date +%T) full eval sft-34k-retrieval"
bash $logs/full_eval.sh sft-34k-retrieval sft-retrieval > $logs/full_eval_sft-34k-retrieval.log 2>&1
echo "$(date +%T) full eval sft-34k"
bash $logs/full_eval.sh sft-34k sft > $logs/full_eval_sft-34k.log 2>&1

for spec in "8 4 on:" "4 8 off:--no-checkpointing"; do
  read -r b a mode <<< "${spec%%:*}"; flag=${spec#*:}
  idle
  echo "$(date +%T) speed test batch $b x $a, checkpointing $mode"
  ~/.venvs/sft/bin/python -m geo_search_env.experiment.sft_train train --examples 34523 --run speedtest --data artifacts/sft/sft_retrieval.jsonl \
    --batch $b --accumulation $a --max-steps 8 $flag > $logs/speed_b${b}_ckpt-$mode.log 2>&1 || echo "  failed (see speed_b${b}_ckpt-$mode.log)"
done
rm -rf /data/pinpoint/sft/speedtest

init=$(.venv/bin/python - <<'PY'
import glob, json
score = {}
for run in ("sft-34k-retrieval", "sft-34k"):
    [path] = glob.glob(f"artifacts/strategy_search/pivot_vlm_sampling_{run}_sft*-prompt*_full-eval.json")
    best = json.load(open(path))["both"]["VLM best of 8 (oracle)"]
    score[run] = best["Under_1_km"] + best["Under_25_km"]
print(max(score, key=score.get))
PY
) || { echo "no full-eval results; stopping"; exit 1; }
if [ "$init" = sft-34k-retrieval ]; then data=artifacts/sft/sft_retrieval.jsonl; prompt=sft-retrieval; else data=artifacts/sft/sft.jsonl; prompt=sft; fi
idle
echo "$(date +%T) GRPO pilot from $init"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/.venvs/sft/bin/python -m geo_search_env.experiment.grpo_train train --init $init --data $data --run grpo-pilot --hours 2 > $logs/train_grpo-pilot.log 2>&1 \
  || { echo "GRPO failed"; exit 1; }
~/.venvs/sft/bin/python -m geo_search_env.experiment.grpo_train merge --init $init --run grpo-pilot >> $logs/train_grpo-pilot.log 2>&1
echo "$(date +%T) eval grpo-pilot"
bash $logs/eval_run.sh grpo-pilot 0.7 $prompt $data > $logs/eval_grpo-pilot.log 2>&1
bash $logs/full_eval.sh grpo-pilot $prompt > $logs/full_eval_grpo-pilot.log 2>&1
echo "$(date +%T) done"
