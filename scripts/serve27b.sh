#!/usr/bin/env bash
# Serve Qwen3.6-27B Q4_K_M (+ vision projector) with llama.cpp on :8766, thinking off, 8 slots (used by knowledge_scaling.sh, knowledge_wiki.sh).
# Needs a CUDA build of llama.cpp (LLAMA_CPP, default ~/llama.cpp/build/bin) and the unsloth/Qwen3.6-27B-GGUF snapshot.
set -euo pipefail
B=${LLAMA_CPP:-$HOME/llama.cpp/build/bin}
S=/data/hf/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf
export LD_LIBRARY_PATH=$B
exec $B/llama-server -m $S/Qwen3.6-27B-Q4_K_M.gguf --mmproj $S/mmproj-BF16.gguf --alias vlm \
  --host 127.0.0.1 --port 8766 -ngl 99 -c 16384 -np 8 --reasoning off \
  --chat-template-kwargs '{"enable_thinking": false}' --image-max-tokens 1024
