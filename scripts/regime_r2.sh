#!/bin/bash
# VALUE_PLAN regime idea R2: one shared multi-scale value network vs one network per radius (dev3k only), two runs at a time.
cd "$(dirname "$0")/.."
run() {  # $1: log suffix, rest: extra flags
    local tag=$1; shift
    systemd-run --user --scope --quiet -p MemoryMax=11G -p MemorySwapMax=0 env PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.value_agent \
        --gain --mc --listwise 2 --multi --targets --lambdas 0 0.005 "$@" > "artifacts/value/value_agent_r2_${tag}.log" 2>&1
}
run shared & run r25 --only-r 25 & wait
run r1 --only-r 1 & run r750 --only-r 750 & wait
