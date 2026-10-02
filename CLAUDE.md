# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read it before proposing experiments).**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (pivoted 2026-10-01; user's decision)
**Question: can Qwen3.5-4B turn visual clues and geographic hypotheses into natural-language search queries that
acquire useful evidence beyond whole-image retrieval and improve its final geolocation prediction?**
- Long-term goal: an RL agent that learns what to search for, how to interpret new evidence, and when to stop. The
  final answer may be an initial candidate, a retrieved image's location, or any lat/lon.
- First experiment: a fixed search loop with an existing model, no new SFT or RL. Full design, arms, gate and
  prerequisites: `QUERY_EVIDENCE_PLAN.md`.
- Supersedes the earlier plan (per-candidate evidence SFT: exemplar photos + GeoNames landmarks), which was never run.
  That plan and the "learning what to search for" no-go are recorded in LEARNINGS.md (lessons 7-9). Known risks to
  watch: SigLIP2's text tower knows concepts, not place names (lesson 7); off-list guesses are almost always wrong
  (lesson 4); the SFT checkpoint was trained on one image + a candidate list, so multi-image evidence prompts are out
  of distribution; the 200-photo dev set is too noisy for a 2-pt gate (use a larger one).
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (e.g. new evidence vs
  the shown-candidate oracle and the extra-whole-image-retrieval arm), not against current greedy.

## Rules
- Final test set: **im2gps3k, yfcc4k and wikimedia** (`/data/pinpoint/wikimedia`). Wikimedia isn't in
  `data/benchmarks.py` or the retrieval caches yet; add it (same-photographer exclusion, no training on it) before
  final numbers. Results so far cover only im2gps3k and yfcc4k.
- Exclude same-photographer gallery images (yfcc4k shares photographers with MP16). Never train on the benchmarks.
- Train only on Pinpoint's held-out MP16 bucket 99 (its retriever trained on the rest).
- Other projects (InnerSight, gems) also use the GPU; never touch their processes.
- Filter images with GPS coordinates burned into the frame.
- The GPU (RTX 5090, 32 GB) is shared with InnerSight jobs: check `nvidia-smi` first and never kill other
  processes. Stop our own processes by PID (`pkill -f` patterns have matched our own shell).
- Skip the Gemini contamination test.
- Confirm plans with the user before long runs; report results concisely with numbers. Use the full eval halves
  (`vlm_sampling --full-eval`) for decisions; the 300 subset is too noisy for gaps under ~4 pts.

## Environment
- Repo env: `.venv` (uv; extras `retrieval`, `feasibility`, `real`). Run code as
  `.venv/bin/python -m geo_search_env.experiment.<module>`.
- Training envs: `~/.venvs/sft` (torch 2.14, transformers 5.17, peft, trl 1.14, fla) for SFT; `~/.venvs/grpo` = copy
  of the vLLM env + trl/peft/accelerate/datasets/fla (TRL colocated vLLM rollouts; put its `bin` on PATH).
- Rollouts/eval: vLLM 0.30 at `/home/brian/.venvs/vllm` (put its `bin` on PATH for ninja). Weights:
  `/data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Serve bf16,
  `enable_thinking: False`, `n=8` per request, `max_pixels` 786432, **no MTP**. Scripts: `scripts/`
  (`eval_run.sh`, `full_eval.sh`, `evidence_test.sh`).
- Data:
  - MP16 embeddings: `/data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384`.
  - MP16 metadata: `/data/hf/datasets/MP16-Pro/metadata/MP16_Pro_filtered.csv`.
  - MP16 images: tar shards read through `tar_index.pkl` (`sft_data.MP16Images`).
  - OSV-5M train embeddings: `/data/pinpoint/osv5m-embed/`.
  - Benchmarks: `/data/pinpoint/{im2gps3k,yfcc4k}`. GeoNames: `/data/pinpoint/geonames/allCountries.txt`.
- OpenRouter key: `OPENROUTER_API_KEY` in `.env` (credits ran out on 2026-09-29).
- Pinpoint baseline code: `/home/brian/workspace/pinpoint-submission/submission`.

## Code map (`src/geo_search_env/`)
- `data/benchmarks.py`: benchmark loader and metrics.
- `experiment/strategy_search.py`: world loading, neighbour caches, region head, candidate pools, reranker features.
- `experiment/verifiers.py`: reranker ranking, study subset.
- `experiment/pivot_diagnostics.py`: `region_search`, `vlm_sampling` (prompts `default|sft|sft-evidence|sft-retrieval`,
  `--full-eval`), `candidate_evidence`.
- `experiment/sft_data.py`: MP16 query pool, candidates, leak check, overlay filter, SFT datasets (`--variant`).
- `experiment/sft_train.py`: LoRA SFT, merge, MP16-val decode eval. `experiment/grpo_train.py`: TRL GRPO + merge.
- `experiment/llm_advantage.py`: Gemini photo labels, per-slice model vs reranker, GeoNames coverage.
- `experiment/embed_cache.py`: builds the SigLIP2 gallery and benchmark embedding caches (`--compare` checks one).
- `experiment/query_headroom.py`: go/no-go for crop and text retrieval queries (SigLIP2 giant).
- `experiment/evidence_test.py`: zero-shot test of per-candidate evidence (exemplar photos, GeoNames landmarks);
  `landmarks` builds `/data/pinpoint/geonames/landmarks.npz` (~1 min).
- Caches: `artifacts/strategy_search/` (benchmarks) and `artifacts/sft/` (MP16 pool); `artifacts/` is not tracked.
- Full setup from scratch: `SETUP.md`.
