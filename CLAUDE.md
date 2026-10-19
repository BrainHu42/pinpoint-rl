# pinpoint-rl

RL agent for image geolocation. Target: beat our retrieval baselines and Gemini, and later call live data APIs at
inference, so design for tool use. **Past results and lessons: `LEARNINGS.md` (read its summary before proposing
experiments); numbered lessons, condensed, in `LESSONS.md` (full original text at git tag `lessons-full-2026-10-15`).**

## Direction (decided with the user)
- Keep the RL framing. Not a plain supervised reranker (that just rebuilds Pinpoint).
- Train on ground-truth labels only. No distillation from Gemini (it would cap us at Gemini); Gemini is a benchmark.
- Base model: **Qwen3.5-4B, thinking off**, LoRA.
- Work toward one research idea with novelty, not small incremental experiments.

## Research plan (2026-10-15)
- **The reference to beat is Pinpoint's attention reranker** ("the reranker" everywhere; `models/pinpoint_reranker.py`, an
  exact port of the released model; lesson 54): top-1 17.5 / 37.2 % within 1 / 25 km on the eval halves, oracle over its pool
  (<= 12 candidates, the pool every experiment chooses from) 30.1 / 54.7; dev3k 17.0 / 33.4 (oracle 27.5 / 50.3).
  **Primary benchmark (user, 2026-10-15): the original Pinpoint protocol, the query photographer's photos kept in the index**
  (`PINPOINT_KEEP_OWN=1`, ckpt tag `setft_aown`; lesson 98): a table per benchmark (im2gps3k, yfcc4k, both) x 1 / 25 / 200 / 750 / 2500 km
  vs the released Pinpoint under the same protocol (both 28.9 / 46.4 / 60.7 / 75.5 / 86.7; im2gps3k 21.8 / 48.9 / 64.6 / 79.6 / 89.8;
  yfcc4k 33.5 / 44.8 / 58.2 / 73.0 / 84.8). Photographer-excluded numbers (our training rule so far; Pinpoint 17.5 / 37.2) are secondary.
  Our models were all trained with the photographer excluded and lose at 1 km on yfcc4k under the primary protocol; expect to retrain
  them (reranker, heads, policy, train / dev3k pools) without the exclusion.
- Where we are (LEARNINGS summary): every chooser, evidence source, comparator and the knowledge SFT landed at -0.5 to +1.3 pts
  over the reranker top-1 (then our one-step reranker, 16.7 / 38.2, and its ~17-candidate pool), because each re-encodes what
  visual retrieval already knows. Knowledge in the weights sits on the photos retrieval already solves (lessons 52-53).
- **Plan: `EXPERT_PLAN.md`** (2026-10-15): choose among retrieval's candidate areas the way an expert does: contrast the candidates,
  fetch reference imagery (`references`: YFCC100M offline, Flickr / Mapillary / Commons live) and map data of each, eliminate, verify,
  stop; an RL policy decides which check on which candidates, with real costs. Earlier plans (agent direction, hypothesis testing,
  look-zoom-rerank, text search, knowledge SFT) were deleted; they are at git tag `plans-2026-10-15`, which lessons citing them refer to.
- **Current best pipeline (lesson 109; lesson 111 adds `--one-km`)**: `action_agent.py` on the reranker fine-tuned without photographer
  exclusion (`setft_own`, lesson 108): candidate areas (agreement clusters + clue places) -> an answer head over area tokens, with costly
  actions (zoom = Pinpoint restricted to an area, read = 4B clues + text search + clue judge, recognise = MegaLoc same-place similarity)
  chosen by a REINFORCE policy. Test set (eval halves), original protocol, RL lambda 0: 29.1 / 51.2 / 62.9 / 75.5 / 86.9 vs released
  28.9 / 46.4 / 60.7 / 75.5 / 86.7 (+4.8 at 25 km); lambda 0.005 30.9 / 50.5 (+2.0 / +4.1). commons26 test (lesson 113): 24.2% < 25 km vs
  released 16.8, the gain mostly from reading. Run: `action_agent --ckpt-tag setft_own --apply full:setft_own` (test only when asked).
  What built it (LEARNINGS table): agreement (68), hypotheses (69), reranker fine-tune + area model / answer head (70-74), clue reading as
  a costly action (76), MegaLoc (93), the RL action policy (94).
- **Closed lines (lessons 77-121), none above the 1-pt bar beyond the pipeline**: map / name checks and names as candidates (77-79), query
  and clue RL (82, 87, 96), nested / region proposals (83-85), the 4B reading neighbour tags (86), bigger readers (88, 106, 114), crop views
  (89), round trips (90), next-bucket and MegaLoc refinement (91, 107, 112), word-location statistics (92), MegaLoc as proposer / over more
  photos (95, 97), contrast with fetched references (99-103), more areas (112), and the commons26 text-search line (115-121: span search,
  OSM names, a learned candidate ranker, a confidence override, +0.9 at best). Their code was removed on 2026-10-16 and is at git tag
  `pre-merge-cleanup-2026-10-16`. Paid-LLM readers (80, 102-105) were diagnostics only. Since then, also closed (122-133): clustering
  variants (122), clue-area recall (123), geotag accuracy / hubness (124), near-miss training (125), head context columns (126), graph
  diffusion over the gallery (127), and local judges of the head's top two areas: Gemma 3 27B in three formats, name PMI, Wikipedia /
  Wikivoyage profiles in the prompt, photo cues vs country tables (128, 131-133).
- **Where the pipeline loses (lessons 129-130, dev3k, 47.3 < 25 km)**: retrieval miss 26.5 pts (24.6 of them photos without the
  photographer's own photo in the pool), right candidate never an area 11.8 (noise-level reranker weight, out of reach), wrong area picked
  14.4 (wrong picks a median 381 km from the truth). The reachable ceiling with today's retrieval is the area oracle (61.7), not the pool
  oracle (73.1). commons26 dev: retrieval miss 49.1, concentrated where the gallery is sparse.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the reranker top-1
  and the pool oracle), not against current greedy. Choose scorers and combiners on MP16 dev, never on benchmarks.

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
