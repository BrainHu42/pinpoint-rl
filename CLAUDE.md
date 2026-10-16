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
- Results of the look-zoom-rerank line (lesson 70):
  - retrained Pinpoint reranker +3.1 / +3.3 (train CV / dev3k) as one model;
  - area model over zoomed states +4.4 / +4.9;
  - RL zoom agent: 2 zooms = the full gain.
  - Next: run the area pipeline on the retrained reranker.

  Benchmark result (eval halves, lesson 73): merged pipeline (retrained reranker, 2 agent zooms, answer head) 42.1% < 25 km,
  +4.9 pts over Pinpoint; ties the previous pipeline (+4.6, hypothesis testing). With the gated pick (lesson 74 default): +4.8 at 25 km, +0.8 at 1 km.
  Lesson 74: gated final pick (default), longer lists and more reranker training don't help (fine-tune saturates at ~+3; photographer
  exclusion is its key ingredient). Training throughput notes in lesson 74.
  Lesson 75: cross-fitted labels don't help (photos Pinpoint's encoder trained on are easier than evaluation); only bucket 99 is
  in-distribution. Next: new evidence (clues, tools) with RL deciding when to use it.
  Lesson 76 (`read_agent.py`): clue reading as a costly action; a learned read policy keeps +0.5 / +0.8 of the clue gain with 20% of the reads.
  Lesson 77 (map checks): OSM checks of the features the 4B sees add nothing (retrieval's wrong areas look alike).
  Lesson 78 (name checks): OSM name search inside areas is redundant with the text-search clue evidence.
  Lesson 79 (name search): names as a candidate generator (Pinpoint restricted to a name's matches): oracle +1.3, learned switch +0.
  Lesson 80: a blind test with Claude as the reader beats the 4B on hard photos through recognition / regional knowledge (4 / 40 within 25 km vs 0).
  Lesson 81: the agent's tools consolidated into `tools/` (pinpoint, maps, web); the modules of lessons 55-63 and 77-79 archived at tag
  `pre-tool-consolidation`.
  Lesson 82 (`query_passk.py`): gate for query RL: novel pass@16 9.7% / pass@1 2% on fixable photos (~3 pts ceiling, ~0.4 at pass@1).
  Lesson 83 (`nested_passk.py`): nested country / region / place proposals: ceiling +6.1 (best of 16), single sample -19.6, simple selector +0.1;
  verification is the bottleneck.
  Lesson 84: regions from Pinpoint's 1,000 nearest neighbours: best of 16 +19.0 vs the 4B's +6.1; the 4B is confidently wrong where it disagrees.
  Lesson 85 (`region_hyps.py`, `region_checks.py`, `HYP_VARIANT=regionhyps`): region candidates + maps / text checks in the head: +0.3 at most;
  the head gets 38.7% where the best of its own areas is 51.7%.
  Lesson 86 (`tag_reader.py`): the 4B reading each area's neighbour YFCC titles / tags picks near chance (21.7% vs area A 36.2%).
  Lesson 87 (`clue_grpo.py`): GRPO on the 4B's clue / region output with a Pinpoint-relative label reward; pilot -0.37 (zero-shot -1.13): learns
  silence, not precision. Full-train run `clue-grpo-full` stopped at step 950 (reward flat; data/pinpoint/sft/).
  Lesson 88 (`reader_compare.py`): Qwen3.8-27B-FP8 as the reader finds fewer right place names (3.6 vs 5.1% of top-1 mistakes) and countries than the 4B.
  Lesson 89 (`crop_look.py`, no LLM): 10 crop views per photo compared with each area's restricted photos, in the answer head: +0.0 / +0.4.
  Lesson 90 (`round_trip.py`): reverse-rank (round-trip) verification of areas: AUC 0.66 vs 0.655 for forward similarity; gate failed.
  Lesson 91 (`refine_headroom.py`, `refine_gate.py`): refining to the next bucket: oracle +11-17 pts, local chooser +-0.1 (signal at chance).
  Lesson 92 (`text_geo.py`): word-location statistics from YFCC for all read text: +0.1 over the existing clue features.
  Lesson 93 (`megaloc_gate.py`): MegaLoc (visual place recognition) same-place similarity per area in the answer head (`--extra megaloc`):
  +0.8 to +1.0 train CV, +0.5 to +1.1 dev3k; the largest evidence gain since lesson 70. Weights in `data/hf/hub/models--gberton--MegaLoc`.
  Lesson 94 (`action_agent.py`): RL action policy over zoom / read / recognise with costs; at or above every fixed combination, best
  39.7% on dev3k (+6.3 over Pinpoint). The current best pipeline.
  Lesson 95 (`megaloc_deep.py`): MegaLoc as a proposer (deep same-place search) finds right new places on ~1% of photos; verifier only.
  Lesson 96 (`query_grpo.py`): GRPO on free-text search queries with the tool result as reward learns silence (precision 26 -> 64%),
  new right places 0.6 -> 0.3% of dev3k.
  Lesson 97 (`megaloc_area.py`): MegaLoc over 64 photos per area instead of 12: AUC 0.707 -> 0.737, area pick +0.1; inside an area
  it is worse than Pinpoint's restricted top-1 at 1 km (15.5 vs 17.0%).
  Lesson 98 (test set, `action_agent --apply`): 43.5% < 25 km (+6.2) / 19.1% < 1 km (+1.6) photographer excluded; original protocol
  (`PINPOINT_KEEP_OWN=1`, ckpt tag `setft_aown`) 48.2-48.7% (+1.8 to +2.3), 1 km 25.5-28.6% vs 28.9.
  Lesson 99 (`contrast_gate.py`, EXPERT_PLAN first experiment): the 4B contrasting the head's top two areas with fetched YFCC references is
  position-biased and below the head (67 vs 83% on decisive, order-consistent pairs); fetched references beat Pinpoint's photos (67 vs 57%).
  Lesson 100 (`contrast_model.py`): our own comparator over MegaLoc matches to fetched references adds nothing to the head (AUC 0.821 vs 0.819).
  Lesson 101 (`patch_matcher.py`): a learned patch-level matcher (frozen MegaLoc DINOv2 patches) over the same references: stacked weight ~0, end to end -1.6.
  Lesson 102: Sonnet (subagent) as the contrast reader, 40 pairs: right 12 / 20 where the head is wrong, 17 / 20 where right; confident answers only on landmarks the head gets.
  Lesson 103 (`pair_discriminant.py`): a discriminant between the two areas' photos: nothing beyond the head.
  Lesson 104: Sonnet told the candidates' town names (not photos): 6 fixes / 0 breaks at confidence >= 0.7 on 20 + 20 pairs (~ +1.8 pts if it holds).
  Lesson 105: the same on dev3k (202 decisive pairs): 8 fixes / 3 breaks at confidence >= 0.7, +0.94 pts at 25 km; the first gain beyond the head since lesson 94.
  Lesson 106 (`names_judge.py`): local readers on the same pairs (4B 55%, Qwen3.8-27B 60-71%) are uncalibrated; every threshold nets negative. No paid LLMs (user, 2026-10-15).
  Lesson 107 (`refine_gate megaloc`): MegaLoc in a learned 1 km local chooser around the top-1: +0.6 to +0.8 pts at 1 km on dev3k
  (SE ~0.3; without MegaLoc +0.1), 25 km unchanged; below the 1-pt bar.
  Lesson 108: dev3k under the original protocol (`PINPOINT_KEEP_OWN=1` drops only the query photo for MP16 queries): released 27.4 / 42.3,
  setft_a 25.1 / 41.3 (it skips the photographer's own photos); `setft_own` (no exclusion) 28.3 / 44.3, +0.9 / +2.0: the new base.
  Lesson 109 (test set, original protocol, `action_agent --ckpt-tag setft_own --apply full:setft_own`): RL lambda 0 29.1 / 51.2 / 62.9 / 75.5 / 86.9
  vs released 28.9 / 46.4 / 60.7 / 75.5 / 86.7 (+4.8 at 25 km); lambda 0.005 30.9 / 50.5 (+2.0 / +4.1). The current best pipeline.
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
- `data/benchmarks.py`: benchmark loader and metrics. `paths.py`: storage roots.
- `models/pinpoint_reranker.py`: Pinpoint's attention reranker (retrieval over its own index + scoring), batched, exact.
- `experiment/pinpoint_reference.py`: its pools and top-1 for the benchmarks and MP16 queries (`load_pools`), report.
- `experiment/photo_sets.py`: fixed photo sets with Pinpoint's pool (`load_photos`; in use: `dev3k`, `full`).
- **Agent tools, `tools/` (lesson 81; contract in `tools/__init__.py`: a text face the agent reads, structured batch calls for training):**
  - `tools/pinpoint.py`: `Pinpoint.search(embeddings, constraints, authors)`, constraint none / box / country / region / near points / rows
    (photographer excluded; wide 32 / 8 / 8 pools as the pipeline; `selftest`). `PinpointReranker.widen` / `rank_subset` underneath.
  - `tools/maps.py`: Maps name search with `near=`, phone / website -> the businesses listing it, `features_near`, batch `FeatureIndex`;
    `data/maps_search/osm.db` (`stage` to the NVMe drive after a restart). `tools/osm_extract.py`: OSM planet -> `data/osm/extract`.
  - `tools/web.py`: Wikipedia / Wikivoyage pages (`experiment/web_search.py`, owned by the osm-search session) + where the words occur in
    Flickr titles / tags and named places (`text_search`).
- The read chain behind the pipeline's clue features (lessons 55-65): `agent_targets.py` (dev set, splits, fixable
  targets), `observe.py` + `scripts/observe.sh` (VLM clues, crops), `text_search.py` (SQLite FTS5 over GeoNames, Overture places / websites /
  phones, YFCC titles + tags in `data/text_search/`; `stage` copies them to the NVMe drive after a restart), `region_evidence.py`,
  `clue_judge.py` + `scripts/clue_judge.sh`, `yfcc_names.py` (YFCC100M extraction -> `data/yfcc100m/geo.npz`), `pinpoint_tool.py` (Pinpoint's
  pool as areas). Expert clue catalogue: `GEOLOCATION_CLUES.md`; locatability ceiling `scripts/locatability_score.py` (lesson 67).
  Archived at tag `pre-tool-consolidation`: landmark matching (`landmark_id`, `landmark_match`, `landmark_vote`), `geo_tools`, `poi_query`,
  `poi_near`, `wiki_backend`, `web_geocode`, `clue_points`, `recall_verify`, `clue_survey`, `chain_footprint`, `vlm_country`, `plates`,
  `pinpoint_constrained`, `agent_capacity`, `agent_vote`, `map_checks`, `name_checks`, `name_search` and their scripts.
- **Agreement re-scoring of Pinpoint's pool (lesson 68, +1.0 pts on train): `pool_agreement.py`.**
- **Hypothesis testing (lesson 69, +2.9 / +3.5 train, +3.8 / +4.6 dev3k):** `hypotheses.py` (Pinpoint areas, restricted search, clue
  hypotheses), `vlm_hypotheses.py` (4B place proposals), `hypothesis_compare.py` (features), `hypothesis_learn.py` (comparator; best:
  `--e4 --variant vlmhyps --drop vlm --agree --ensemble 5`, saved `artifacts/agent/hyp_final_comparator.pt`), `hypothesis_vlm.py` +
  `scripts/hypvlm_run.sh` (plausibility judge, zero-shot / fine-tuned; no gain).
- **Look-zoom-rerank (lesson 70):**
  - `pinpoint_set.py`: set layer, wide extraction (`extract --ckpt-tag`), zoom pools, `eval-head`.
  - `area_model.py`: area-then-spot model over zoomed CLS states.
  - `hypothesis_agent.py`: merged pipeline (answer + action heads over area tokens; lesson 71). Retrained-reranker photo sets: `pinpoint_set photo-set`, `scripts/retrained_pipeline.sh`.
  - `zoom_agent.py`: RL zoom policy. `read_agent.py`: clue reading as a costly action with a learned read policy (lesson 76).
  - `hypothesis_agent.fit_answers`: many answer heads at once (vmap); `load(extra=...)` appends column groups (lessons 77-78).
  - `models/set_head.py`.
  - `pinpoint_train/`: pinpoint-main's training code with a set head, wide pools, author exclusion and a source alias for training against the shipped index. Run it with
    `data/pinpoint-main-src/pinpoint-main/.venv/bin/python` and `PYTHONPATH=pinpoint_train/src:src`. Launcher flags are in lesson 70.
  - Retrained checkpoints: `artifacts/pinpoint_train/{setft_a,ft_nohead}`. The port loads one via `PINPOINT_RERANKER_CKPT`.
- `experiment/strategy_search.py`: world loading, neighbour caches, region head, shared geometry helpers.
- `experiment/sft_data.py`: MP16 query pool, overlay filter, SFT datasets. `experiment/pivot_diagnostics.py`: chooser prompts
  and the benchmark harness for any VLM chooser (`vlm_sampling`).
- `experiment/sft_train.py`: LoRA SFT, merge. `experiment/grpo_train.py`: TRL GRPO + merge.
- `experiment/embed_cache.py`: SigLIP2 gallery and benchmark embedding caches.
- Knowledge SFT: `place_labels.py` (Overture division labels; `pool-labels` labels a photo set's pool), `knowledge_data.py`
  (training photos), `name_score.py` (candidate name scoring + combiner).
- Knowledge diagnostics (lessons 52-53): `knowledge_check.py` (answers, report, rule), `knowledge_oracle.py` (perfect-knowledge
  ceiling per level, fixable photos), `knowledge_coverage.py` (accuracy by training coverage).
- Earlier lines (comparator, near-miss, stage-1 evidence, map search, the Mapillary environment): removed 2026-10-07; their
  modules and scripts (named in `LESSONS.md`) are at git tag `pre-cleanup-2026-10-07`. The one-step reranker, our ~17-candidate
  pool builder and the modules built on them (`verifiers`, `evidence_ranker`, `exemplar_judge`, `multi_exemplar`,
  `category_evidence`, `stage1_eval`, `evidence_test`, `query_headroom`, `query_evidence`) were removed when Pinpoint's reranker
  became the reference (lesson 54); they are at commit 57918cb. Kept: `coarse_filter`, `feasibility`.
- Caches: `artifacts/` (not tracked). Photo sets and pools built before lesson 54 (`artifacts/query_evidence/*/dev.json`,
  `artifacts/sft/candidates.npz`, `artifacts/place_labels/candidates_*.json`) hold the old reranker's pools; current code doesn't read them.
  Setup from scratch: `SETUP.md`.
