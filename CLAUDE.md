# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results: `LEARNINGS.md`, one page of main points (read it before proposing experiments);
new lessons in `LESSONS.md` (per track: R1.. retrieval, S1.. search agent); lessons 1-137 archived in `docs/archive/LESSONS_1-137.md`, linked
from the summary.**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (2026-10-15)
- **The reference to beat is Pinpoint's attention reranker** ("the reranker" everywhere; `models/pinpoint_reranker.py`, an
  exact port of the released model). **Primary benchmark (user, 2026-10-15): the original Pinpoint protocol, the query photographer's
  photos kept in the index** (`PINPOINT_KEEP_OWN=1`): a table per benchmark (im2gps3k, yfcc4k, both) x 1 / 25 / 200 / 750 / 2500 km vs the
  released Pinpoint under the same protocol (eval halves: both 28.9 / 46.4 / 60.7 / 75.5 / 86.7; im2gps3k 21.8 / 48.9 / 64.6 / 79.6 / 89.8;
  yfcc4k 33.5 / 44.8 / 58.2 / 73.0 / 84.8), with commons26 alongside. Photographer-excluded numbers are secondary.
- **Current best pipeline and where it stands: `LEARNINGS.md`** (`action_agent.py` on the `setft_own` reranker; test 51.2% < 25 km vs
  released 46.4). Run: `action_agent --ckpt-tag setft_own --apply full:setft_own` (test only when asked).
- **Two tracks, run in parallel sessions (user, 2026-10-20)**: `RETRIEVAL_PLAN.md` (retrain Pinpoint's retriever / index side, the largest
  loss) and `SEARCH_PLAN.md` (an RL agent whose sequential, constrained searches create candidates). Each brief has the coordination rules
  for sharing the machine. The previous plan (`EXPERT_PLAN.md`: contrast candidates with fetched references, closed by lessons 99-106, 128,
  131-137) is at `docs/archive/EXPERT_PLAN.md`; older plans at tag `plans-2026-10-15`.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the current pipeline and the pool
  oracle), not against a weaker baseline. Choose scorers and combiners on MP16 dev, never on benchmarks.

## Data (simplified 2026-10-07, user)
One split, ids in `artifacts/agent/splits.json` (`agent_targets --splits`):
| Split | What | Size |
|---|---|---|
| **train** | MP16 bucket 99 (Pinpoint never trained on it) minus dev | 35,359 |
| **dev** | `dev3k` (photo set `artifacts/photo_sets/dev3k`, Pinpoint's pool): choose and tune everything here | 3,000 |
| **test** | **all of im2gps3k + yfcc4k** (both halves; user, 2026-10-16). Results through lesson 111 used only the eval halves (photo set `full`, 3,795), Pinpoint's reference rows too | 7,533 |
- **Leave the test set alone** (user, 2026-10-07): no training, tuning, model runs, feature extraction, diagnostics or measurements on
  it unless the user asks. Everything is chosen and measured on dev.
- No photographer-based splits. (The cached retrieval already drops gallery photos by the query's own photographer; nothing to do about it.)
- Target: improvements of **>= 1 pt** at 25 km over the reranker top-1 (user, 2026-10-07); dev3k (+-0.26 pts SE) resolves that.
- dev3k is rebuilt from `artifacts/agent/dev3k_ids.json` (`agent_targets --make-dev dev3k`, then `place_labels pool-labels --tag dev3k`).
- Retired: the `val` photo set (1,000 test photos, never use), the photo sets `dev` / `devall` and the old `artifacts/query_evidence/*`
  sets (old one-step-reranker pools), dev10k, the knowledge-SFT bucket 0-98 exception.
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
- `data/benchmarks.py`: benchmark loader and metrics (im2gps3k, yfcc4k, commons26). `paths.py`: storage roots.
- `models/pinpoint_reranker.py`: Pinpoint's attention reranker (retrieval over its own index + scoring), batched, exact; loads a
  retrained checkpoint via `PINPOINT_RERANKER_CKPT`; `PINPOINT_KEEP_OWN=1` is the original protocol. `models/set_head.py`: the set head.
- `experiment/pinpoint_reference.py`: its pools and top-1 for the benchmarks and MP16 queries (`load_pools`), report.
- `experiment/photo_sets.py`: fixed photo sets with Pinpoint's pool (`load_photos`; in use: `dev3k`, `full`, `commons26`).
- `experiment/agent_targets.py`: dev set, splits, fixable targets.
- **The pipeline, in build order** (`scripts/full_pipeline.sh` for a new photo set's clue and hypothesis inputs,
  `scripts/retrained_pipeline.sh` for a retrained reranker; `docs/pipeline_overview.png`, `scripts/pipeline_figure.py`):
  - Retrained reranker: `pinpoint_train/` (pinpoint-main's training code with a set head, wide pools, author exclusion and a source alias
    for training against the shipped index; run with `data/pinpoint-main-src/pinpoint-main/.venv/bin/python` and
    `PYTHONPATH=pinpoint_train/src:src`; launcher `scripts/pinpoint_train.sh`, flags in lesson 70). Checkpoints
    `artifacts/pinpoint_train/{setft_own,setft_a,ft_nohead}`. `pinpoint_set.py`: set layer, wide extraction (`extract --ckpt-tag`),
    `photo-set`, zoom pools (`restricted`), `eval-head`.
  - Read chain (clue features): `observe.py` + `scripts/observe.sh` (4B clues, crops), `text_search.py` (SQLite FTS5 over GeoNames, Overture
    places / websites / phones, YFCC titles + tags in `data/text_search/`; `stage` copies them to the NVMe drive after a restart),
    `region_evidence.py`, `clue_judge.py` + `scripts/clue_judge.sh` (fine-tuned judge), `yfcc_names.py` (YFCC100M -> `data/yfcc100m/geo.npz`;
    `scripts/yfcc_extract*.sh`). Expert clue catalogue `GEOLOCATION_CLUES.md`; locatability `scripts/locatability_*.py` (lesson 67).
  - Areas and hypotheses: `pinpoint_tool.py`, `pool_agreement.py` (lesson 68), `hypotheses.py` (areas, restricted search, clue hypotheses),
    `vlm_hypotheses.py` (4B place proposals), `hypothesis_compare.py` (features), `hypothesis_learn.py` (comparator, lesson 69).
  - Heads and agents: `area_model.py`, `hypothesis_agent.py` (answer + action heads over area tokens, gated pick; `fit_answers`;
    `load(extra=("megaloc",))`), `zoom_agent.py`, `read_agent.py` (lesson 76), `megaloc_gate.py` (MegaLoc columns, lesson 93),
    `crop_look.py` (crop views; view 0 is the policy's free glance), `refine_headroom.py` (geometry helpers), `action_agent.py` (the RL action
    policy, `--apply`, `--one-km`).
- Diagnostics and screens (lessons 122-133): `scripts/pipeline_diagnostics.py` and `scripts/candidate_drill.py` (where photos are lost),
  `graph_diffusion.py` (MP16 kNN graph, `artifacts/diffusion/`), `names_judge.py` (named A/B and name PMI with a local VLM), `profile_judge.py`
  (Wikipedia / Wikivoyage area profiles), `cue_check.py` (photo cues vs GeoNames country tables), `scripts/subset_photo_set.py` (a split-only
  photo set with its caches, e.g. `commons26dev_setft_own`).
- Probes of lessons 134-137: `density_probe.py` (denser photo database), `clue_verify.py` (fetched YFCC photos / OSV-5M per text place,
  MegaLoc and keypoints), `osv_propose.py` / `osv_refine.py` / `osv_remote.py` (OSV-5M as proposer, street action, 1 km refiner),
  `names_grpo.py` (trained named A/B judge; GRPO and exact expected reward).
- Pipeline viewer (`viz/`): `agent.py` saves the lesson 109 answer head + RL policies (reproduces 51.2 / 29.1 on test), `traces.py` writes
  per-photo stage traces (`artifacts/viz/traces/{full,dev3k}`), `server.py` the web app (`--live`: uploads through `live.py`, the whole
  pipeline on one photo, ~1 min; clue reading needs `scripts/viz_vlm.sh`).
- Agent tools, `tools/` (lesson 81; contract in `tools/__init__.py`): `pinpoint.py` (`Pinpoint.search` with constraints), `maps.py` (OSM /
  GeoNames / Overture name search, `features_near`; `data/maps_search/osm.db`), `osm_extract.py` (`.venvs/osm`), `web.py` (Wikipedia /
  Wikivoyage via `experiment/web_search.py` + `text_search`).
- commons26: `commons_bench.py` (scan, selection, download, near-duplicates, tiers, release `data/pinpoint/commons26/release/`),
  `commons26_score.py` (standalone scorer), `scripts/commons26_score_km.py`.
- Shared infrastructure: `strategy_search.py` (world loading, neighbour caches, geometry; uses `coarse_filter`, `feasibility`,
  `candidate_locations`), `embed_cache.py` (SigLIP2 caches), `sft_data.py`, `sft_train.py` (LoRA SFT, merge), `grpo_train.py` (TRL GRPO),
  `pivot_diagnostics.py` (VLM harness, `vlm_sampling`), `place_labels.py` (`pool-labels`), `knowledge_data.py`, `knowledge_check.py`, `name_score.py`.
- Removed code, by tag: closed lines of lessons 77-121 `pre-merge-cleanup-2026-10-16`; agent-tool / landmark / text-search checks of
  lessons 55-69, 72 `pre-prune-2026-10-13` and `pre-tool-consolidation`; comparator / near-miss / stage-1 / map-search / Mapillary
  `pre-cleanup-2026-10-07`; the one-step reranker and modules on it commit 57918cb. Old plans `plans-2026-10-15`.
- Caches: `artifacts/` (not tracked). Setup from scratch: `SETUP.md`.
