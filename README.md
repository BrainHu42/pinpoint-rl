# pinpoint-rl

Research code for training a vision-language model (Qwen3.5-4B, LoRA, thinking off) to geolocate photos with
supervised fine-tuning and reinforcement learning (GRPO), on top of a retrieval baseline (Pinpoint: SigLIP2 embeddings
of MP16-Pro and OSV-5M). Evaluation is on the im2gps3k and yfcc4k eval halves with same-photographer gallery images
excluded.

- **`LEARNINGS.md`**: main points of the results so far (read first), linked to the archived lessons 1-125 in `docs/archive/`; new lessons in `LESSONS.md`.
- **`EXPERT_PLAN.md`**: the current plan. Earlier plans are at git tag `plans-2026-10-15`.
- **`CLAUDE.md`**: research plan summary, project rules, environment notes and code map.
- **`SETUP.md`**: complete setup guide (hardware, environments, models, datasets, cache build order, commands).

## Repository layout

- `src/geo_search_env/experiment/`: experiment modules, each run as
  `.venv/bin/python -m geo_search_env.experiment.<module> <node>` (usage lines at the top of each file; code map in
  `CLAUDE.md`).
- `src/geo_search_env/data/benchmarks.py`: benchmark loader and metrics.
- `src/geo_search_env/models/pinpoint.py`: frozen Pinpoint retrieval baseline.
- `scripts/`: end-to-end evaluation scripts (serve a model with vLLM, evaluate, stop the server).
- `tests/`: pytest suite.
- Code of superseded directions (comparator, near-miss, map search, stage-1 evidence, the Mapillary environment) was removed;
  it is at git tag `pre-cleanup-2026-10-07`.
- `data/` (git-ignored): datasets, model weights, gallery embeddings and Hugging Face caches.
- `.venvs/` (git-ignored): SFT, GRPO and vLLM environments.
- `artifacts/` (git-ignored): caches, datasets, logs and result JSONs, all rebuilt by the modules (see `SETUP.md`).

## Conventions

- Stop only your own processes, by PID.
- Never train on the benchmarks, always exclude same-photographer gallery rows, and train only on MP16 bucket 99 (knowledge SFT exception in `CLAUDE.md`).
- No distillation from Gemini: it is a benchmark and a source of evaluation-only labels.
