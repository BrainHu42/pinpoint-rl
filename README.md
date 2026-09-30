# pinpoint-rl

Research code for training a vision-language model (Qwen3.5-4B, LoRA) to geolocate photos with reinforcement learning,
on top of a retrieval baseline (Pinpoint: SigLIP2 embeddings of MP16-Pro and OSV-5M). Evaluation is on im2gps3k and
yfcc4k with same-photographer gallery images excluded; wikimedia is the planned third test set.

- **`LEARNINGS.md`**: results so far and what they rule out (start here).
- **`CLAUDE.md`**: current research plan, project rules, environment and code map.

## Layout
- `src/geo_search_env/experiment/`: experiment modules, each runnable as
  `python -m geo_search_env.experiment.<module> <node>` (usage lines at the top of each file).
  - `strategy_search.py`, `verifiers.py`: retrieval caches, candidate pools, reranker.
  - `sft_data.py`, `sft_train.py`, `grpo_train.py`: MP16 training data, LoRA SFT, TRL GRPO.
  - `pivot_diagnostics.py`: VLM evaluation (`vlm_sampling`) against any OpenAI-compatible server.
  - `llm_advantage.py`, `query_headroom.py`, `evidence_test.py`: headroom and go/no-go analyses.
- `tests/`: `pytest` suite.
- `artifacts/`: result JSONs and run scripts (`*/logs/*.sh`). Caches, datasets and logs are git-ignored and rebuilt
  by the modules above.
- `archive/`: superseded plans.

## Setup
Python ≥ 3.11 with [uv](https://github.com/astral-sh/uv): `uv sync --extra retrieval --extra feasibility`.
Training and serving use separate environments (torch + transformers 5.17, TRL 1.14, peft, flash-linear-attention;
vLLM 0.30); see `CLAUDE.md`. Paths to datasets, embeddings and model weights are hard-coded for our machine
(`/data/...`) and need adapting elsewhere. API keys go in `.env` (git-ignored).
