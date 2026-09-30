# pinpoint-rl

Research code for training a vision-language model (Qwen3.5-4B, LoRA, thinking off) to geolocate photos with
supervised fine-tuning and reinforcement learning (GRPO), on top of a retrieval baseline (Pinpoint: SigLIP2 embeddings
of MP16-Pro and OSV-5M). Evaluation is on the im2gps3k and yfcc4k eval halves with same-photographer gallery images
excluded; wikimedia is the planned third test set.

- **`LEARNINGS.md`**: results so far and what they rule out (read first).
- **`CLAUDE.md`**: current research plan, project rules, environment notes and code map.
- **`SETUP.md`**: complete setup guide (hardware, environments, models, datasets, cache build order, commands).

## Repository layout

- `src/geo_search_env/experiment/`: experiment modules, each run as
  `.venv/bin/python -m geo_search_env.experiment.<module> <node>` (usage lines at the top of each file; code map in
  `CLAUDE.md`).
- `src/geo_search_env/data/benchmarks.py`: benchmark loader and metrics.
- `src/geo_search_env/models/pinpoint.py`: frozen Pinpoint retrieval baseline.
- `scripts/`: end-to-end evaluation scripts (serve a model with vLLM, evaluate, stop the server).
- `tests/`: pytest suite; `fixtures/`: synthetic test fixtures.
- `archive/`: superseded plans.
- `artifacts/` (git-ignored): caches, datasets, logs and result JSONs, all rebuilt by the modules (see `SETUP.md`).

## Conventions

- The GPU may be shared with other jobs: check `nvidia-smi` before launching, and stop only your own processes (by
  PID).
- Never train on the benchmarks, always exclude same-photographer gallery rows, and train only on MP16 bucket 99.
- No distillation from Gemini: it is a benchmark and a source of evaluation-only labels.
