# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read it before proposing experiments).**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (pivoted 2026-10-01, staged 2026-10-02; user's decisions)
**Stage 1 (now): acquire new evidence.** Can Qwen3.5-4B write search queries that retrieve evidence beyond what we
already have (whole-image retrieval, the reranker's candidates) that contains the answer? **Metric: oracle accuracy**
= % of photos where at least one location stage 2 would see (shown candidates + retrieved evidence coordinates) is
within 1 / 25 km of the truth. It is the ceiling for any stage-2 chooser, with no chooser in the loop. Report it next
to the shown-candidates-only oracle (the gain is the stage-1 result), the reranker top-1 and the extra-whole-image-
retrieval control, all at a fixed retrieval budget (queries per photo, results per query), since the oracle only grows
with more results. Backends: SigLIP2 photo search, offline geotagged Wikipedia (`wiki_backend.py`), later live APIs.
**Stage 2 (later): consume the evidence and decide between candidates.** Out of scope until stage 1 works. The first
attempt (LEARNINGS 11: 4B, query photo + six evidence photos in one prompt) failed and is not a fair test: too many
images for a 4B model (see the small-VLM-prompts memory).
- Protocol (user's decision, 2026-10-02): **develop on the MP16 dev set** (1,000 held-out MP16 val photos; tag `dev`,
  `query_evidence select`) and **validate on 1,000 photos from the im2gps3k / yfcc4k eval halves** (500 each, fixed
  seed; tag `val`, `query_evidence select --source bench --tag val`), not the full 3,795, to keep runs manageable.
  Validation reference (reranker top-1 / shown top-10 oracle, % <1 km / <25 km): 18.4 / 42.8 and 34.1 / 59.0
  (im2gps3k 20.4 / 50.8 and 38.8 / 64.2; yfcc4k 16.4 / 34.8 and 29.4 / 53.8). The 50/50 mix is not comparable to the
  pooled 3,795 numbers (yfcc4k is 61% of those). Final test: wikimedia once its loader exists (still to add).
- Long-term goal: an RL agent that learns what to search for, how to interpret new evidence, and when to stop. The
  final answer may be an initial candidate, a retrieved image's location, or any lat/lon. Stage 1 reward = the oracle
  accuracy gain from the retrieved coordinates, computed from ground truth.
- Where stage 1 stands (LEARNINGS 11-12, always against the reranker top-1): SigLIP2 text search finds almost nothing
  new; Wikipedia search works with good names (Gemini's: +12 pts oracle over reranker on 645 landmark photos), but the
  4B's names add only +2.5 greedy / +4.2 best of 8, the 27B +4.7 / +6.2. New coverage beyond the shown candidates is
  ~1-2 pts for the 4B. The query design is in `QUERY_EVIDENCE_PLAN.md` (its stage-2 revision loop is superseded).
- Supersedes the earlier plan (per-candidate evidence SFT: exemplar photos + GeoNames landmarks), which was never run.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the reranker top-1
  and the shown-candidate oracle), not against current greedy.

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
- `experiment/query_evidence.py`: fixed query → SigLIP2 evidence → revise loop on MP16 dev photos
  (`scripts/query_evidence.sh`; outputs `artifacts/query_evidence/<tag>/`).
- `experiment/wiki_backend.py`: offline geotagged-Wikipedia search (`/data/pinpoint/wikipedia/enwiki_geo.sqlite` BM25 +
  `enwiki_geo_bge-base.f16.npy` dense; run with `~/.venvs/sft/bin/python`, `.venv` lacks pyarrow), `probe` go/no-go.
- Caches: `artifacts/strategy_search/` (benchmarks) and `artifacts/sft/` (MP16 pool); `artifacts/` is not tracked.
- Full setup from scratch: `SETUP.md`.
