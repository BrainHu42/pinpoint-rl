#!/usr/bin/env bash
# GRPO for the named-candidate judge (names_grpo.py train), vLLM colocated, in a memory-capped cgroup; then the LoRA merge.
# Usage: RUN_NAME=names-grpo-a HOURS=4 [EXTRA="--lr 1e-5"] setsid nohup bash scripts/names_grpo.sh > artifacts/agent/logs/names_grpo_train.log 2>&1 &
set -uo pipefail
if [ -z "${IN_SCOPE:-}" ]; then
  exec systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 env IN_SCOPE=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH=src VLLM_USE_FLASHINFER_SAMPLER=0
export C_INCLUDE_PATH=$HOME/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12
export PATH=$PWD/.venvs/vllm/bin:$PATH
run=${RUN_NAME:-names-grpo-a}
echo "$(date +%T) train $run"
.venvs/grpo/bin/python -m geo_search_env.experiment.names_grpo train --run $run --hours ${HOURS:-4} ${EXTRA:-} || exit 1
echo "$(date +%T) merge $run"
.venvs/grpo/bin/python -m geo_search_env.experiment.names_grpo merge --run $run || exit 1
echo "$(date +%T) done"
