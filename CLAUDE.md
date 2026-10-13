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
- **The reference to beat is Pinpoint's attention reranker** ("the reranker" everywhere; `models/pinpoint_reranker.py`, an
  exact port of the released model; lesson 54): top-1 17.5 / 37.2 % within 1 / 25 km on the eval halves, oracle over its pool
  (<= 12 candidates, the pool every experiment chooses from) 30.1 / 54.7; dev3k 17.0 / 33.4 (oracle 27.5 / 50.3).
- Where we are (LEARNINGS summary): every chooser, evidence source, comparator and the knowledge SFT landed at -0.5 to +1.3 pts
  over the reranker top-1 (then our one-step reranker, 16.7 / 38.2, and its ~17-candidate pool), because each re-encodes what
  visual retrieval already knows. Knowledge in the weights sits on the photos retrieval already solves (lessons 52-53).
- **Direction: `AGENT_DIRECTION.md`** (agreed 2026-10-07): train an agent that geolocates like a professional, using
  constraints, not similarity: read text, apply rule-like clues, query maps for features that co-occur near each
  candidate, geocode names, then eliminate and verify. Retrieval is the candidate-generator tool. Target: the photos
  where the reranker is wrong but knowing the region would fix it (dev3k: 302 of 3,000, `agent_targets --tags dev3k`).
- **Current plan (2026-10-12): `PINPOINT_SET_PLAN.md`**: look, zoom, rerank. Results of the autonomous run (lesson 70):
  - retrained Pinpoint reranker +3.1 / +3.3 (train CV / dev3k) as one model;
  - area model over zoomed states +4.4 / +4.9;
  - RL zoom agent: 2 zooms = the full gain.
  - Next: run the area pipeline on the retrained reranker.

  Benchmark result (eval halves, lesson 73): merged pipeline (retrained reranker, 2 agent zooms, answer head) 42.1% < 25 km,
  +4.9 pts over Pinpoint; ties the previous pipeline (+4.6, `HYPOTHESIS_PLAN.md`).
  Lesson 74: gated final pick (default), longer lists and more reranker training don't help (fine-tune saturates at ~+3; photographer
  exclusion is its key ingredient). Training throughput notes in lesson 74.
  Lesson 75: cross-fitted labels don't help (photos Pinpoint's encoder trained on are easier than evaluation); only bucket 99 is
  in-distribution. Next: new evidence (clues, tools) with RL deciding when to use it.
  Lesson 76 (`read_agent.py`): clue reading as a costly action; a learned read policy keeps +0.5 / +0.8 of the clue gain with 20% of the reads.
- **Previous plan (2026-10-09): `HYPOTHESIS_PLAN.md`, geolocation as hypothesis testing.** Propose area hypotheses (Pinpoint's agreement
  areas, clue places), gather evidence inside each with Pinpoint restricted to it, compare with a learned set-level comparator; later an RL
  agent choosing which hypotheses to test. Development reference: Pinpoint + agreement (lesson 68). The text-search line and its results are
  in `archive/TEXT_SEARCH_PLAN.md`.
- Superseded plans (knowledge SFT + GRPO, earlier directions) are in `archive/`.
- Go/no-go rule learned the hard way: measure what a change adds *beyond what we already have* (the reranker top-1
  and the pool oracle), not against current greedy. Choose scorers and combiners on MP16 dev, never on benchmarks.

## Data (simplified 2026-10-07, user)
One split, ids in `artifacts/agent/splits.json` (`agent_targets --splits`):
| Split | What | Size |
|---|---|---|
| **train** | MP16 bucket 99 (Pinpoint never trained on it) minus dev | 35,359 |
| **dev** | `dev3k` (photo set `artifacts/photo_sets/dev3k`, Pinpoint's pool): choose and tune everything here | 3,000 |
| **test** | im2gps3k + yfcc4k eval halves (photo set `full`) | 3,795 |
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
- `data/benchmarks.py`: benchmark loader and metrics (`commons26` entry). `paths.py`: storage roots.
- `models/pinpoint_reranker.py`: Pinpoint's attention reranker (retrieval over its own index + scoring), batched, exact; loads a
  retrained checkpoint via `PINPOINT_RERANKER_CKPT`. `models/set_head.py`: the set head used in retraining.
- `experiment/pinpoint_reference.py`: its pools and top-1 for the benchmarks and MP16 queries (`load_pools`), report.
- `experiment/photo_sets.py`: fixed photo sets with Pinpoint's pool (`load_photos`; in use: `dev3k`, `full`).
- `experiment/agent_targets.py`: dev set, splits, fixable targets.
- **The pipeline (lessons 68-71, 73-74, 76; `docs/pipeline_overview.png`, `scripts/pipeline_figure.py`).** In order:
  - Clues: `observe.py` + `scripts/observe.sh` (VLM clues, crops); `yfcc_names.py` (YFCC100M extraction -> `data/yfcc100m/geo.npz`,
    README there; `scripts/yfcc_extract*.sh`; name -> place proposals); `text_search.py` (SQLite FTS5 over GeoNames, Overture places /
    websites / phones, YFCC titles + tags in `data/text_search/`; `stage` copies them to the NVMe drive after a restart);
    `region_evidence.py`; `clue_judge.py` + `scripts/clue_judge.sh` (fine-tuned judge of read-text clues).
  - Areas and hypotheses: `pinpoint_tool.py` (Pinpoint as agreement-weighted areas), `pool_agreement.py` (lesson 68), `hypotheses.py`
    (areas, restricted search, clue hypotheses), `vlm_hypotheses.py` (4B place proposals), `hypothesis_compare.py` (features),
    `hypothesis_learn.py` (set-level comparator, lesson 69; `scripts/full_pipeline.sh` runs this version end to end).
  - Look-zoom-rerank (`PINPOINT_SET_PLAN.md`, lesson 70): `pinpoint_set.py` (set layer, wide extraction `extract --ckpt-tag`, zoom
    pools, `eval-head`, `photo-set`), `area_model.py` (area-then-spot model over zoomed CLS states), `zoom_agent.py` (RL zoom policy).
  - Merged pipeline (lessons 71, 73-74): `hypothesis_agent.py` (answer + action heads over area tokens, gated final pick) on the
    retrained reranker (`scripts/retrained_pipeline.sh`). `read_agent.py`: clue reading as a costly action (lesson 76).
  - Retrained reranker: `pinpoint_train/` (pinpoint-main's training code with a set head, wide pools, author exclusion and a source alias
    for training against the shipped index; run with `data/pinpoint-main-src/pinpoint-main/.venv/bin/python` and
    `PYTHONPATH=pinpoint_train/src:src`; launcher `scripts/pinpoint_train.sh`, flags in lesson 70). Checkpoints:
    `artifacts/pinpoint_train/{setft_a,ft_nohead}`.
  - Locatability tiers (lesson 67): `scripts/locatability_score.py`, `fixtures/locatability_dev3k.json`.
- Offline search tools for the agent (stand-ins for Google web / Maps): `osm_extract.py` (named objects and landmark features from the
  OSM planet), `maps_search.py` (name search over OSM + GeoNames / Overture, `features_near`), `web_search.py` (multilingual Wikipedia +
  Wikivoyage, FTS5).
- `experiment/strategy_search.py`: world loading, neighbour caches, region head, shared geometry helpers (uses `coarse_filter`,
  `feasibility`, `candidate_locations`, `core/contracts`).
- `experiment/sft_data.py`: MP16 query pool, overlay filter, SFT datasets. `experiment/pivot_diagnostics.py`: chooser prompts and the
  VLM harness (`vlm_sampling`). `experiment/sft_train.py`: LoRA SFT, merge. `experiment/grpo_train.py`: TRL GRPO + merge.
- `experiment/embed_cache.py`: SigLIP2 gallery and benchmark embedding caches.
- `experiment/commons_bench.py`: the commons26 benchmark (lesson 79): Commons scan, selection, download, gallery near-duplicates,
  27B tiers, release (`data/pinpoint/commons26/release/`) and `score`; `commons26_score.py` is the standalone scorer.
- Shared helpers from the knowledge SFT line: `place_labels.py` (Overture division labels; `pool-labels` labels a photo set's pool),
  `knowledge_data.py`, `knowledge_check.py`, `name_score.py`.
- Removed 2026-10-13 (lines that didn't pan out; code at git tag `pre-prune-2026-10-13`): the agent-tool, landmark and text-search checks
  (`geo_tools`, `poi_query`, `poi_near`, `landmark_match`, `landmark_id`, `landmark_vote`, `agent_capacity`, `agent_vote`,
  `pinpoint_constrained`, `clue_points`, `web_geocode`, `recall_verify`, `clue_survey`, `chain_footprint`, `vlm_country`, `plates`,
  `area_rules`, `deep_votes`; lessons 55-68), the plausibility judge (`hypothesis_vlm`, lesson 69), the location distribution
  (`location_density`, lesson 72), cross-fitting, reranker ablation and profiling scripts (lessons 74-75), `knowledge_oracle`,
  `knowledge_coverage`, `wiki_backend`, the knowledge and VLM-chooser eval scripts. Plans and reports of those lines are in `archive/`.
  Earlier lines (comparator, near-miss, stage-1 evidence, map search, the Mapillary environment) are at tag `pre-cleanup-2026-10-07`;
  the one-step reranker and the modules built on it at commit 57918cb.
- Caches: `artifacts/` (not tracked). Photo sets and pools built before lesson 54 (`artifacts/query_evidence/*/dev.json`,
  `artifacts/sft/candidates.npz`, `artifacts/place_labels/candidates_*.json`) hold the old reranker's pools; current code doesn't read them.
  Setup from scratch: `SETUP.md`.
