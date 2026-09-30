#!/usr/bin/env bash
# Equal-memory vLLM rollout benchmark: each model runs alone with the same GPU memory budget (needs an idle GPU).
# For each (photos, samples, tokens) workload, reports the median rollout time per model (and the ratio when both
# models are present). Models whose weights are not downloaded are skipped; Qwen3.5-4B is the kept setup.
# Env: UTIL (0.85), ROUNDS (4), WORKLOADS ("4x8x128 8x8x128 4x8x512 8x8x512"), Q35_ARGS / Q3_ARGS.
set -u
cd /home/brian/workspace/pinpoint-rl
export PATH=/home/brian/.venvs/vllm/bin:$PATH  # vLLM JIT-compiles kernels with ninja
UTIL=${UTIL:-0.85}; ROUNDS=${ROUNDS:-4}
WORKLOADS=${WORKLOADS:-"4x8x128 8x8x128 4x8x512 8x8x512"}
Q3=/data/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
Q35=/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
mkdir -p /tmp/claude-1001
RESULTS=/tmp/claude-1001/vllm_solo_results.txt
: > "$RESULTS"
for spec in "qwen3-vl-4b|$Q3|${Q3_ARGS:-}" "qwen3.5-4b|$Q35|${Q35_ARGS:-}"; do
  IFS='|' read -r name model extra <<< "$spec"
  [ -d "$model" ] || { echo "skipping $name: weights not present at $model"; continue; }
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done
  # shellcheck disable=SC2086
  HF_HUB_OFFLINE=1 vllm serve "$model" --served-model-name "$name" --port 8778 --host 127.0.0.1 --dtype bfloat16 \
    --gpu-memory-utilization "$UTIL" --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image":1}' $extra \
    > "/tmp/claude-1001/vllm_solo_$name.log" 2>&1 &
  server=$!
  until curl -sf 127.0.0.1:8778/health >/dev/null; do kill -0 $server 2>/dev/null || { echo "$name server exited"; exit 1; }; sleep 2; done
  echo "$name: $(grep -hoE 'GPU KV cache size: [0-9,]+ tokens' /tmp/claude-1001/vllm_solo_$name.log | head -1)"
  .venv/bin/python - "$name" "$ROUNDS" $WORKLOADS >> "$RESULTS" <<'EOF'
import base64, json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from geo_search_env.experiment.pivot_diagnostics import PROMPT
name, rounds, workloads = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
images = sorted(Path("/data/pinpoint/im2gps3k/images").glob("*.jpg"))
prompt = PROMPT.format(options="\n".join(f"{i}. Candidate city {i}, Region, Country ({10 * i:.3f}, {5 * i:.3f})" for i in range(1, 11)))

def call(image, seed, samples, gen):
    body = {"model": name, "temperature": 1.0, "seed": seed, "max_tokens": gen, "n": samples, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}}, {"type": "text", "text": prompt}]}]}
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8778/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})) as r:
        return json.loads(r.read())["usage"]["completion_tokens"]

def rollout(photos, samples, gen, offset):
    batch = [base64.b64encode(p.read_bytes()).decode() for p in images[offset : offset + photos]]
    start = time.time()
    with ThreadPoolExecutor(photos) as pool:
        tokens = sum(pool.map(lambda image: call(image, offset, samples, gen), batch))
    assert tokens == photos * samples * gen, tokens
    return time.time() - start

for w in workloads:
    photos, samples, gen = (int(x) for x in w.split("x"))
    rollout(photos, samples, gen, 900)  # warm-up for this shape
    walls = [rollout(photos, samples, gen, 100 * r) for r in range(rounds)]
    print(f"{w} {name} {statistics.median(walls):.3f}")
EOF
  kill $server; wait $server 2>/dev/null
done
.venv/bin/python - "$RESULTS" <<'EOF'
import sys
rows = [line.split() for line in open(sys.argv[1]) if line.strip()]
table = {}
for w, name, t in rows:
    table.setdefault(w, {})[name] = float(t)
for w, v in table.items():
    ratio = f" | ratio Qwen3.5 / Qwen3-VL {v['qwen3.5-4b'] / v['qwen3-vl-4b']:.2f}" if len(v) == 2 else ""
    print(f"{w:38s} " + " | ".join(f"{name} {t:.2f}s" for name, t in v.items()) + ratio)
EOF
