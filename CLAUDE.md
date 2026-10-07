# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read its summary before proposing
experiments); full numbered lessons in `LESSONS.md`.**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (2026-10-07)
- Where we are (LEARNINGS summary): the answer is usually in the ~17-candidate pool (oracle 33.4 / 59.0% within
  1 / 25 km) but every chooser, evidence source, comparator and the knowledge SFT lands at -0.5 to +1.3 pts over the
  reranker top-1 (16.7 / 38.2), because each re-encodes what visual retrieval already knows. Knowledge in the weights
  sits on the photos retrieval already solves (lessons 52-53).
- **Direction: `AGENT_DIRECTION.md`** (agreed 2026-10-07): train an agent that geolocates like a professional, using
  constraints, not similarity: read text, apply rule-like clues, query maps for features that co-occur near each
  candidate, geocode names, then eliminate and verify. Retrieval is the candidate-generator tool. Target: the photos
  where the reranker is wrong but knowing the region would fix it (413 of 3,795 on the eval halves; develop on the
  MP16 dev equivalent).
- Superseded plans (knowledge SFT + GRPO, earlier directions) are in `archive/`.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the reranker top-1
  and the pool oracle), not against current greedy. Choose scorers and combiners on MP16 dev, never on benchmarks.

## Data (simplified 2026-10-07, user)
One split, ids in `artifacts/agent/splits.json` (`agent_targets --splits`):
| Split | What | Size |
|---|---|---|
| **train** | MP16 bucket 99 (Pinpoint never trained on it) minus dev | 28,359 |
| **dev** | `dev10k` (`artifacts/query_evidence/dev10k`): choose and tune everything here | 10,000 |
| **test** | im2gps3k + yfcc4k eval halves (`artifacts/query_evidence/full`): measure once per finished method | 3,795 |
- **Leave the test set alone** (user, 2026-10-07): no training, tuning, model runs, feature extraction, diagnostics or measurements on
  it unless the user asks. Everything is chosen and measured on dev.
- No photographer-based splits. (The cached retrieval already drops gallery photos by the query's own photographer; nothing to do about it.)
- Retired: the `val` tag (1,000 test photos, never use), the 1,000-photo `dev` (inside dev10k), the knowledge-SFT bucket 0-98 exception.
- Burned-in GPS photos are already filtered out of train and dev. No wikimedia (not on this machine).

## Rules
- RAM is 30 GB with no swap: run long jobs in a memory-capped cgroup (`systemd-run --user --scope -p MemoryMax=24G
  -p MemorySwapMax=0`); an uncapped job crashed the machine once.
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
- Knowledge diagnostics (lessons 52-53): `knowledge_check.py` (answers, report, rule), `knowledge_oracle.py` (perfect-knowledge
  ceiling per level, fixable photos), `knowledge_coverage.py` (accuracy by training coverage).
- Earlier lines (comparator, near-miss, stage-1 evidence, map search, the Mapillary environment): removed 2026-10-07; their
  modules and scripts (named in `LESSONS.md`) are at git tag `pre-cleanup-2026-10-07`. Kept helpers they shared:
  `evidence_test`, `exemplar_judge`, `multi_exemplar`, `wiki_backend`, `category_evidence`, `coarse_filter`, `feasibility`.
- Caches: `artifacts/` (not tracked). Setup from scratch: `SETUP.md`.
