# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read it before proposing experiments).**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (paused 2026-09-30)
**Current hypothesis: choosing among retrieved candidates is capped by the evidence the model has about each one.**
- Where we are (details in LEARNINGS.md):
  - Retrieval already returns the answer: the ~17 shown candidates contain it for 33.4 / 59.0% (within 1 / 25 km),
    but every chooser we trained (reranker, SFT, GRPO) tops out at ~16.9 / 38 (LEARNINGS 1-3).
  - Finding *new* candidates doesn't help: text and crop queries add < 2 pts to the candidate oracle (LEARNINGS 7).
    The query-rewriting plan ("learning what to search for", locked 2026-09-29) failed its go/no-go and is dropped.
  - Gemini, given the same kind of candidate list, closes about two thirds of the gap (27.7 / 55.7 on the 300 subset),
    so the cap is not fundamental; for a 4B model the missing piece is knowledge, and per-candidate evidence is the
    substitute (LEARNINGS 9).
  - Untrained, per-candidate evidence helps a little: + nearby GeoNames landmarks or + one exemplar photo per
    candidate each add ~+1.3 pts within 25 km to the base 4B (LEARNINGS 8).
- Proposed next step (not started): one SFT run with both kinds of evidence (exemplar photo + nearby place names per
  candidate), same data/recipe as sft-34k-retrieval, full eval halves.
  - Go (evidence gathering becomes the core of the agent) if it clearly beats the reranker (> ~2 pts within 25 km
    over 38.2). Same as sft-34k-retrieval (16.3 / 37.2) → the 4B chooser is knowledge-capped; rethink (larger model,
    or a different research question).
  - Cost: dataset build ~30 min (`experiment/evidence_test.py` has the evidence builders), SFT ~5-6 h.
- If go, the agent: tools that fetch evidence *about candidates* (exemplar photos, nearby places via a fuzzy
  geocoder / OpenStreetMap, basic geography), then `answer`; SFT warm start on trajectories built from ground truth
  (no Gemini distillation); multi-turn GRPO with hard-photo filtering, entropy control, step-0 and best-of-8 evals
  (LEARNINGS 3). Novelty to argue: candidate-conditioned evidence gathering over a geotagged image memory, vs
  one-shot retrieve-then-pick (Img2Loc, G3, GeoRanker) and map-only agents without image memory (Thinking with Map).
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (e.g. new candidates
  vs the shown-candidate oracle), not against current greedy, which the existing selection gap would pass trivially.

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
