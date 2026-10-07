# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read it before proposing experiments).** **Current state and the next step: `HANDOFF.md`.**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research state (2026-10-06; details in `HANDOFF.md`, numbers in `LEARNINGS.md`)
- **Baseline: Pinpoint's attention reranker top-1** (the submission's full model, same-photographer gallery rows excluded; LEARNINGS 51):
  17.9 / 38.1 / 55.1% within 1 / 25 / 200 km on the 3,713 benchmark eval-half photos, 8.5 / 24.7 / 57.5 on wikimedia; oracle over its 12
  candidates 30.6 / 55.9 / 75.8. Results before 2026-10-07 (LEARNINGS 1-50) were measured against our **one-step reranker** (17.0 / 38.9 /
  56.4; fitted on the benchmark tune halves), which is about equal; its ~17-candidate pool (oracle 34.2 / 60.2 / 80.3) still feeds the
  choosers and comparator data. Either way the headroom is in **choosing among candidates**.
- Closed (LEARNINGS 1-49): SFT / single-turn GRPO of the 4B as a chooser (reranker parity); new evidence from search queries,
  Wikipedia, place names, text, attributes ("stage 1", 11-21; old plan in `archive/stage1_plan.md`); zero-shot choosers up to 27B;
  comparator scaling and combiners (best confirmed **+0.9 [+0.3, +1.6] at 25 km**, fine-tuned 4B comparator, 24-29, 39-46);
  near-miss refinement, keypoint matching and map search (43-49).
- **Open: knowledge SFT** (LEARNINGS 50): train the 4B to name a photo's place (country > region > city > neighbourhood, Overture
  labels on MP16), score each candidate's name, combine with the rank. Labels and training photos are built; training has not run
  (`scripts/knowledge_run.sh`). Then, depending on the result: combine with the comparator and scale, or bring options to the user.
- Long-term goal unchanged: an RL agent over tools (what to check, how to read it, when to stop), once a tool has a strong signal.
- Protocol: develop on MP16 dev (tag `dev`, 1,000 photos), validate on the 1,000-photo benchmark mix (tag `val`), confirm on all
  benchmark eval-half photos (tag `full`) and wikimedia. Choose scorers and combiners on dev only. Go / no-go: what a change adds
  beyond the reranker top-1 (and the pool oracle for new evidence), not against current greedy; the usual bar is +2 pts.

## Rules
- Final test set: **im2gps3k, yfcc4k and wikimedia** (`/data/pinpoint/wikimedia`; loaded by `data/benchmarks.py`, candidates from
  `wikimedia_eval.py`). Wikimedia so far: baseline, oracle and the comparator (LEARNINGS 42); never train on it.
- Exclude same-photographer gallery images (yfcc4k shares photographers with MP16). Never train on the benchmarks.
- Train only on Pinpoint's held-out MP16 bucket 99 (its retriever trained on the rest). Exception (user's decision, 2026-10-05): the
  knowledge SFT (LEARNINGS 50) teaches place names from buckets 0-98; anything that learns to choose between candidates stays on bucket 99.
- Other projects (InnerSight, gems) also use the GPU; never touch their processes.
- Filter images with GPS coordinates burned into the frame.
- The GPU (RTX 5090, 32 GB) is shared with InnerSight jobs: check `nvidia-smi` first and never kill other
  processes. Stop our own processes by PID (`pkill -f` patterns have matched our own shell).
- Skip the Gemini contamination test.
- Confirm plans with the user before long runs; report results concisely with numbers. Use the full eval halves
  (`vlm_sampling --full-eval`) for decisions; the 300 subset is too noisy for gaps under ~4 pts.

## Environment
- Repo env: `.venv` (uv; extras `retrieval`, `feasibility`, `real`; no pyarrow). Run code as
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
  - Benchmarks: `/data/pinpoint/{im2gps3k,yfcc4k,wikimedia}`. GeoNames: `/data/pinpoint/geonames/allCountries.txt`.
  - Overture: `/data/pinpoint/overture` (places, division polygons). Wikipedia: `/data/pinpoint/wikipedia`.
  - `artifacts/` snapshot (2026-10-06): Hugging Face `kinghorton42/geo-benchmarks`, folder `artifacts/` (download command in `SETUP.md`).
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
- `experiment/stage1_eval.py`: stage-1 evaluation (oracle accuracy over the pool, matched whole-image control, evidence informativeness);
  `evidence_ranker.py`, `attribute_ranker.py`: learned-reranker tests with evidence / attribute features; `photo_attributes.py`,
  `candidate_attributes.py` (run with `~/.venvs/geo/bin/python`: rasterio, WorldClim + ETOPO under `/data/pinpoint/geo`),
  `attribute_check.py`; `overture_text.py` (80M-place name index, `/data/pinpoint/overture`), `evidence_screen.py`, `audit_sets.py`
  (leakage / near-duplicate / placeholder audit of train, dev, val; flags in `artifacts/query_evidence/<tag>/`).
- `experiment/exemplar_judge.py`: exemplar "same place?" judge: screens, top-8 scoring, combiner and full-eval-halves report;
  `comparator_data.py`, `comparator_train.py` (pointwise / pairwise LoRA comparator; merge with `sft_train merge`);
  `knowledge_scaling.py` (zero-shot choosers by model size, optional nearby Wikipedia text from `wiki_nearby.py`);
  `multi_exemplar.py` (several exemplars per candidate).
- `experiment/near_miss.py`: near-miss photos (top-1 1-25 km off): ceiling, local re-rankers, near-band comparator pairs, combiners;
  `map_search.py` (zoom search over the gallery around the top-1), `geo_match.py` (DISK + LightGlue inliers, `~/.venvs/match`).
- `experiment/place_labels.py`: Overture division labels for all MP16 photos and candidate names (`~/.venvs/overture`, DuckDB);
  `name_score.py` (log P(place name | photo) per candidate, combiner report); `knowledge_data.py` (knowledge SFT data:
  `select` with `~/.venvs/sft` for pyarrow, `overlay`, `dataset`); pipeline `scripts/knowledge_run.sh`, scoring `scripts/name_scores.sh`.
- `experiment/wikimedia_eval.py`: wikimedia candidates and baseline / oracle report.
- `models/pinpoint_reranker.py`: Pinpoint's attention reranker through the submission's own code, plus a same-photographer filter on its
  MP16 search; `experiment/pinpoint_reranker_eval.py` (`run` / `parity` / `report`; run with the submission's `.venv` python and
  `PYTHONPATH=src:<submission>/src`; one photo per call, batching changes bf16 results) -> `artifacts/pinpoint_reranker/`.
  Scripts: `comparator_eval.sh`, `comparator_full.sh`, `pairwise_eval.sh`, `knowledge_scaling.sh`, `stage1.sh`, `text_screen.sh`,
  `photo_attributes.sh`, `evidence_ranker.sh`. Photo sets: tags `dev` (MP16 val), `val` (1,000 benchmark eval-half), `full`
  (all 3,795 eval-half), `train` (MP16 train), `wikimedia` under `artifacts/query_evidence/`.
- Caches: `artifacts/strategy_search/` (benchmarks), `artifacts/sft/` (MP16 pool), `artifacts/query_evidence/` (this line of
  experiments); `artifacts/` is not tracked.
- Full setup from scratch: `SETUP.md` (exact env package lists in `envs/`). Handoff state and next steps: `HANDOFF.md`.
