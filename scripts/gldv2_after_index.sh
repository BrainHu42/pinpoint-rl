#!/usr/bin/env bash
# After the GLDv2 index is embedded (lesson R10): search commons26 dev / dev3k / train, the 1 km refinement eval, the answer-head columns.
# Usage: WAIT_PID=<embed pid> setsid nohup bash scripts/gldv2_after_index.sh > artifacts/landmarks/gldv2/after_index.log 2>&1 &
set -euo pipefail
export PYTHONPATH=src
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
run() { systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 --setenv=PYTHONPATH=src "$@"; }
for s in commons26dev dev3k train; do
  run .venv/bin/python scripts/gldv2_index.py search --set "$s"
done
run .venv/bin/python scripts/gldv2_eval.py 0.7 | tee artifacts/landmarks/gldv2/eval_full_cut0.7.txt
run .venv/bin/python scripts/gldv2_head.py --seeds 2 --cut 0.7
echo "after_index done"
