# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read its summary before proposing
experiments); full numbered lessons in `LESSONS.md`.**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (2026-10-06)
- Where we are (LEARNINGS summary): the answer is usually in the ~17-candidate pool (oracle 34.2 / 60.2% within
  1 / 25 km) but every chooser, evidence source and comparator lands at -0.5 to +1.3 pts over the reranker top-1
  (17.1 / 38.9), mostly because it re-encodes what visual retrieval already knows.
- **Main question: can a model learn geographic knowledge that visual similarity doesn't already capture?**
- Next: **`KNOWLEDGE_PLAN.md`** (agreed 2026-10-06): knowledge SFT warm start (photo -> "country > region > city"),
  then GRPO on photo + retrieval candidates from the knowledge model vs from the base 4B; measured by fixes on the
  benchmark photos where the reranker is wrong. Real geolocation only; no artificial held-out tasks.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the reranker top-1
  and the pool oracle), not against current greedy. Choose scorers and combiners on MP16 dev, never on benchmarks.

## Rules
- Test sets: **im2gps3k and yfcc4k** eval halves. Don't use the wikimedia photos for validation (user, 2026-10-06);
  they aren't on this machine.
- Exclude same-photographer gallery images (yfcc4k shares photographers with MP16). Never train on the benchmarks.
- Train only on Pinpoint's held-out MP16 bucket 99 (its retriever trained on the rest). Exception (user's decision,
  2026-10-05): the knowledge SFT teaches place names from buckets 0-98; anything that learns to choose between
  candidates stays on bucket 99.
- Filter images with GPS coordinates burned into the frame.
- The GPU (L40S, 46 GB) is ours alone. Stop our own processes by PID (`pkill -f` patterns have matched our own shell).
- Skip the Gemini contamination test.
- Confirm plans with the user before long runs; report results concisely with numbers. Use the full eval halves for
  decisions; small subsets are for plumbing.

## Environment
- Everything lives under the repo: datasets and weights in `data/` (`src/geo_search_env/paths.py`), training envs in
  `.venvs/`; both git-ignored. Code from the old machine still hard-codes `/data/...` and `~/.venvs`; port it to
  `paths.py` before use.
- Repo env: `.venv` (uv; extras `retrieval`, `feasibility`, `real`; no pyarrow). Run code as
  `.venv/bin/python -m geo_search_env.experiment.<module>`.
- Training envs: `.venvs/sft` (torch 2.14, transformers 5.17, peft, trl 1.14, fla); `.venvs/grpo`; package lists in `envs/`.
- Serving: vLLM 0.30 at `.venvs/vllm` (put its `bin` on PATH for ninja). Weights:
  `data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Serve bf16,
  `enable_thinking: False`, `max_pixels` 786432, **no MTP**.
- Data: MP16 embeddings `data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384`; MP16 metadata and tar shards
  `data/hf/datasets/MP16-Pro` (`sft_data.MP16Images`); OSV-5M `data/pinpoint/osv5m-embed/`, `data/hf/datasets/osv5m`;
  benchmarks `data/pinpoint/{im2gps3k,yfcc4k}`; GeoNames `data/pinpoint/geonames`; Pinpoint `data/pinpoint/pinpoint-submission`
  (released index, version 6).
- OpenRouter key: `OPENROUTER_API_KEY` in `.env` (credits ran out on 2026-09-29).

## Code map (`src/geo_search_env/`)
- `data/benchmarks.py`: benchmark loader and metrics. `paths.py`: storage roots.
- `experiment/strategy_search.py`: world loading, neighbour caches, region head, candidate pools, reranker features.
- `experiment/verifiers.py`: reranker ranking. `experiment/sft_data.py`: MP16 query pool, candidates, overlay filter.
- `experiment/sft_train.py`: LoRA SFT, merge. `experiment/grpo_train.py`: TRL GRPO + merge.
- `experiment/embed_cache.py`: SigLIP2 gallery and benchmark embedding caches.
- Knowledge SFT: `place_labels.py` (Overture division labels), `knowledge_data.py` (training photos), `name_score.py`
  (candidate name scoring + combiner). Photo sets: `query_evidence.py select` (tags `dev`, `val`, `full`).
- Earlier lines (comparator, near-miss, stage-1 evidence, map search): modules named in their lessons (`LESSONS.md`).
- Caches: `artifacts/` (not tracked). Setup from scratch: `SETUP.md`.
