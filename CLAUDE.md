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
  All benchmark numbers exclude the query photographer's own photos from the index; the original protocol (kept) gives 28.9 / 46.4
  (yfcc4k 33.5 / 44.8 vs 14.8 / 29.8; im2gps3k unchanged, no photographer overlap). Report both (user, 2026-10-13; lesson 54).
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
  +4.9 pts over Pinpoint; ties the previous pipeline (+4.6, `HYPOTHESIS_PLAN.md`). With the gated pick (lesson 74 default): +4.8 at 25 km, +0.8 at 1 km.
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
- **Previous plan (2026-10-09): `HYPOTHESIS_PLAN.md`, geolocation as hypothesis testing.** Propose area hypotheses (Pinpoint's agreement
  areas, clue places), gather evidence inside each with Pinpoint restricted to it, compare with a learned set-level comparator; later an RL
  agent choosing which hypotheses to test. Development reference: Pinpoint + agreement (lesson 68). The text-search line and its results are
  in `TEXT_SEARCH_PLAN.md`.
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
- The read chain behind the pipeline's clue features (lessons 55-65; `TEXT_SEARCH_PLAN.md`): `agent_targets.py` (dev set, splits, fixable
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
