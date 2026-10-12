# Pinpoint-Set: look, zoom, rerank (2026-10-12; autonomous run done, results at the end)

One set-aware model on top of Pinpoint replaces the hypothesis pipeline's hand-built parts. It is cleaner and keeps or improves
performance. Background: `HYPOTHESIS_PLAN.md` (current pipeline, all results), `docs/pipeline_overview.png`, lessons 68-69.

## Where we are
Current pipeline (`HYPOTHESIS_PLAN.md`):
1. Pinpoint pool.
2. Set layer (`pinpoint_set.py`).
3. About 6 area hypotheses: Pinpoint's top-5 agreement areas, clue places, 4B place proposals.
4. Pinpoint restricted to each area.
5. A comparator over 29-36 hand features picks an area.
6. The answer is the agreement or set-layer pick inside it.

| % within 25 km (change vs Pinpoint) | train CV | dev3k | Benchmarks (eval halves, one run) |
|---|---|---|---|
| Set layer alone | +2.0 | +2.1 | +1.5 |
| Pinpoint areas + set layer, no LLM / text search | +3.4 | +4.0 | |
| Full pipeline, agreement pick | +4.1 | +4.7 | **+4.6** (41.8 vs 37.2) |
| Full pipeline, set-layer pick (1 km in brackets) | +3.9 (+1.6) | +4.5 (+1.7) | +4.1 (**+1.8**) |

## What the results taught us (the design follows from these)
1. **Retrieval inside the right area is very strong.** Pinpoint restricted to the true region reaches 61% < 25 km on dev3k. The right
   area is among our 5 areas 46-51% of the time; we choose right 37-38%. Zooming is the core operation.
2. **Reasoning over the whole candidate set pays; scoring candidates one at a time loses.** Agreement gave +1.0, the set layer +2.0, the
   comparator +3.5-4.1. Pinpoint's loss pushes the single nearest candidate above all others, even ones 1 km from the truth, so it never
   learns to judge areas.
3. **Pinpoint's internal states carry the signal.** Set layer +2.0 with them, +0.4 without. Hand features have saturated: richer
   restricted features, more hypotheses, attention and more comparator data each landed within +-0.2.
4. **Train on the candidate lists you will see.** The set layer trained on global pools did not transfer to the restricted +-25 km
   pools.
5. **The loss decides.** Cross-entropy to all good candidates spread probability and picked ~5 pts *worse* than Pinpoint. The
   expected-hit loss (-log of the softmax mass on candidates < 25 km, double weight < 1 km) fixed it.
6. **Labels on top of Pinpoint are scarce.** Only ~38k bucket-99 photos are unseen by Pinpoint. Pinpoint overfits buckets 0-98
   (top-1 40% vs 32%), so extra photos from there help little beyond 35k. Pinpoint's own 4M training photos are the real data.
7. **The 4B's place knowledge is redundant with retrieval.** Plausibility judges: zero-shot AUC 0.63, fine-tuned 0.78, no gain; place
   proposals +0.1-0.2. Explicit text clues are the one LLM contribution: reading + search + judge, +0.6-0.7.
8. **Live geocoders are a modest add-on.** Nominatim / Wikipedia reach 38 more of Pinpoint's dev3k mistakes than offline search. Ceiling
   +3.8 -> +5.0 pts; realistic +0.4-0.6.
9. **About half the photos show nothing place-specific** (lesson 67). No method fixes those.

## The design
1. **Look.** Pinpoint searches its 4.08M-photo gallery globally (image / raw / GPS similarity): ~46 candidates.
2. **Zoom.** Its top-K areas are searched again with Pinpoint restricted to each: ~12 candidates per area.
3. **Rerank once.** One set-aware model scores all candidates together, global and zoomed (~100).
   - Inputs: Pinpoint's CLS states, scores, pairwise distances.
   - Output: P(area) x P(spot | area). Loss: expected hit.
   - Replaces: the area-grouping rule, the comparator, its hand features and the choice of pick rule.
4. **Optional clue tokens.** Geocoded text clues (offline search, later live geocoders) enter step 3 as extra tokens with their place and
   judge score. This is the only LLM use: reading text in the photo.

Removed:
- the 4B place proposals and the plausibility judges;
- the clue features and the hand-built comparator features;
- the separate comparator and the choice between pick rules.

## Path, each step with a go / no-go (decide on train 5-fold CV; check on dev3k; test set only when the user asks)
| Step | What | Cost | Go if (25 km, train CV) |
|---|---|---|---|
| 1 | Set model over global + zoomed candidates, on frozen Pinpoint states. First save CLS states of the restricted candidates for train / dev3k (`pinpoint_set restricted`, ~20 min). | 1-2 h | >= +3.4 (Pinpoint-only today), ideally near +4.1 |
| 2 | Area-then-spot output (area tokens attend to their members) | ~1 h | keeps step 1 at 25 km and gains at 1 km |
| 3 | Clue tokens (judge score + place) | ~2 h | recovers the clues' +0.6 |
| 4 | Retrain Pinpoint's reranker with the set objective and in-area hard negatives (runner-up area) on all 4M photos (`pinpoint-main.zip` training code), then recompute pools and zoom searches | setup + a few GPU hours | beats step 3 |
| 5 (later) | An LLM agent, trained with RL, choosing which areas to zoom and which clues to look up (live tools) | open | beats fixed top-K at equal cost, or reaches evidence retrieval lacks |

- If step 1 fails, keep the current pipeline (+4.6 on the benchmarks) and go to step 4.
- Step 4 is the most promising lever: it fixes the training objective where the data is.

## Why
- **Elegant.** One model, one objective; retrieval is the only tool in the core.
- **Strong.** The kept parts already carry +3.4 of +4.1 on train. Step 4 attacks the measured root cause: Pinpoint was never trained to
  judge areas.
- **Paper story.** Geolocation as retrieve, zoom, rerank with a set-aware model trained for area-level correctness. Ablation per step;
  the benchmark run above is the baseline.
- **The agent.** The LLM / RL agent becomes a well-defined extension: spending lookups and zooms where they help. It is no longer the
  core.

## Results of the autonomous run (2026-10-12; lesson 70)
All numbers: % within 25 km, change vs the released Pinpoint top-1 (train 32.4%, dev3k 33.4%). Train = 5-fold CV on the 35k train photos,
dev3k = trained on all of train. 1 km in brackets where it matters. The test set was not touched.

**Headline: step 4 is the lever.** Fine-tuning Pinpoint's own reranker on its training photos for 1.5M photos (6,000 steps of 256, about 1 h
on the L40S) gives **+3.1 train / +3.3 dev3k** (1 km +1.6 / +1.0) as one model with no LLM and no zoom. That matches the whole Pinpoint-only
hypothesis pipeline (+3.4 / +3.9).
The run changed three things at once (plus continued training):
- wide pools (32 / 8 / 8 hits);
- same-photographer exclusion in training, so training pools look like evaluation pools;
- a co-trained set head.

The control run below separates them.

| Step | What was built | Train CV | dev3k | Verdict |
|---|---|---|---|---|
| 1 | Flat set model over global + zoomed candidates (~92 per photo; `pinpoint_set zoom`) | +1.4 (+0.8) | +1.3 (+0.4) | **no-go**: same as the global wide set layer (+1.4 / +1.4); the zoomed tokens add nothing |
| 2 | Area model: one token per hypothesis, attention-pooled from Pinpoint's CLS states in its zoom + hand features (`area_model.py`) | agree pick +4.4; set pick +4.2 (+1.7) | +4.9; +4.7 (+1.6) | **go**: +0.3 over the comparator (+4.1 / +4.7) |
| 2' | Same, 16 Pinpoint-only features + states (no clue features) | +4.0; +3.7 (+1.7) | +4.5; +4.2 (+1.2) | matches the comparator with 16 instead of ~36 hand features |
| 3 | Clue tokens = clue places as hypotheses + their clue-search statistics (all features, areas + clues, no 4B proposals) | +4.2; +3.9 (+1.7) | **+5.2**; +5.0 (+1.8) | clues add +0.7 over Pinpoint areas only (+3.5 / +3.9); 4B place proposals add +0.2 train, -0.3 dev3k: drop them |
| 4 | Pinpoint reranker fine-tuned (pinpoint-main code in `pinpoint_train/`, `models/set_head.py`); its pick over the wide pool | **+3.1 (+1.6)** | **+3.3 (+1.0)** | **go** (one model; the set layer was +1.4 / +2.0 with extra photos) |
| 4 | ... its own 8 / 2 / 2 pool | +2.5 (+1.2) | +2.7 (+0.8) | |
| 4 | ... + its co-trained set head (trained on photos the reranker has seen) | +2.6 (+1.5) | +2.9 (+0.9) | set head no gain |
| 4 | ... + the offline set layer (bucket 99) on its states | +2.1 | +2.2 | no gain: the retrained reranker already does what the set layer did |
| 5 | RL zoom agent: which hypotheses to zoom under a budget (`zoom_agent.py`) | see below | | **go**: 1 zoom reaches ~90% of the full gain, 2 zooms all of it |

Step 5 table. The leak-free version: the chooser scores each zoomed hypothesis from its own evidence only.

| Budget (restricted searches) | Pinpoint's top-B areas | RL policy | All ~6.2 hypotheses zoomed |
|---|---|---|---|
| 1 | +1.1 / +2.1 | **+3.5 / +3.8** | +4.0 / +4.2 |
| 2 | +2.8 / +3.6 | **+4.0 / +4.2** | |
| 3 | +3.3 / +4.0 | +3.9 / +4.1 | |

(train CV / dev3k)
- Policy: a DeepSets MLP over pre-zoom evidence (source, area rank, pool support, gallery density, clue-search statistics, distance to
  Pinpoint's top-1, set-layer mass).
- Training: Gumbel top-B sampling and REINFORCE with the greedy picks as baseline. Reward: the zoomed answer is within 25 km.
- Fixed mixes of areas and clue / 4B places are all worse than top-B areas.

**Setup notes for retraining Pinpoint on this machine:**
1. **Leave-one-out needs a source alias with the shipped submission index.** That index names its source "source_0" (it was built
   with unnamed cache roots), while the data module tags samples "mp16". Training against it maps no query to its own row, so every
   training pool contains the query itself (our first attempt: train accuracy 100%, discarded). Fixed with an alias in
   `retrieval_index._build_source_lookup`.
   - This is a setup issue, not a bug in Pinpoint's training. `train.py` builds its own index with named roots (`{"mp16": ...}`),
     so leave-one-out works there. The released reranker was trained against a different index (`e8b495d78203a679` in its hparams).
     Its accuracy on unseen photos confirms it was not trained with the query in its pool.
2. **Same-photographer exclusion in training** (new; Pinpoint's training doesn't have it), matching our evaluation rule.
   Added `--exclude-author-path`, with authors per cache row in `artifacts/pinpoint_train/author_by_cache_row.npy`.

**Control run (no set head, otherwise identical, `artifacts/pinpoint_train/ft_nohead`):** wide-pool pick +2.1 / +2.3 (1 km +1.2 / +0.4),
own pool +1.7 / +2.0. So:
- continued training with wide pools and photographer exclusion gives about +2 (not separated further);
- co-training the set head adds +1.0 / +1.0 to the reranker's *own* scores, though the head's picks are no better. The set objective
  shapes the states.

**What this means for the design.**
- The zoom tokens and the set head trained on seen photos do not help.
- What helps:
  1. retrain the reranker so its training pools match evaluation;
  2. choose between areas from per-area evidence (area model);
  3. spend zooms where they help (RL agent).
- Next steps:
  1. Run the hypothesis pipeline (areas, zooms, area model) on top of the retrained reranker.
  2. Train longer (a full epoch is ~4 h) and with an expected-hit loss on the base scores.
  3. Re-run the agent on the new states.

## Merged pipeline, steps 1-2 (budgeted hypothesis-testing agent; lesson 71)
% within 25 km vs the released Pinpoint (train 32.4%, dev3k 33.4%), train CV / dev3k; 1 km in brackets. Hypotheses: Pinpoint areas + clue places.

**Step 1: the area pipeline on the retrained reranker** (`pinpoint_set photo-set --ckpt-tag setft_a`, `scripts/retrained_pipeline.sh`, `area_model --ckpt-tag setft_a`).
The retrained reranker's 46-candidate pool replaces Pinpoint's pool, its zooms replace Pinpoint's zooms, and its softmax replaces the offline set layer.
- More areas (4.6 vs 4.2 per photo); hypothesis oracle K = 5: 51.5% on dev3k, vs 48.5%.

| Area model (areas + clues) | Released Pinpoint as the tool | Retrained reranker as the tool |
|---|---|---|
| All features + states, agreement pick | +4.2 / +5.2 | **+4.9 / +5.4** |
| ... restricted top-1 | +3.6 / +4.5 | +4.6 (+1.9) / +4.7 (+0.8) |
| ... set pick | +3.9 (+1.7) / +5.0 (+1.8) | +4.7 (+1.8) / +4.9 (+1.1) |
| 16 Pinpoint-only features + states, agreement pick | | +4.6 / +4.9 |

**Go:** +0.7 on train over the same model on released Pinpoint; +0.2 on dev3k. The two gains partly overlap: retrained alone +3.1 / +3.3, area model alone +4.2 / +5.2.

**Step 2: one network, two heads** (`hypothesis_agent.py`).
- Area tokens: pre-zoom evidence, plus zoom states once zoomed.
- Answer head: trained on random zoom subsets. An unzoomed area answers with the best global-pool candidate inside it.
- Action head: sequential, REINFORCE against the frozen answer head.

| Zoom budget | Retrained: top-B areas | Retrained: RL agent | Released: top-B | Released: RL agent |
|---|---|---|---|---|
| 0 (no zoom) | +3.9 / +3.8 | | +3.3 / +4.1 | |
| 1 | +4.2 / +4.7 | +4.4 / +4.9 | +3.9 / +4.9 | +3.8 / +4.8 |
| 2 | +4.7 / +5.2 | **+4.7 / +5.5** | +3.9 / +5.1 | +4.0 / +4.9 |
| 3 | +4.7 / +5.4 | +4.8 / +5.4 | +4.0 / +5.0 | +4.0 / +5.0 |
| all (~4.6) | +4.9 / +5.5 | | +4.0 / +5.0 | |

Agreement pick throughout. The set pick keeps 1 km at about +1.7 train / +1.1 dev3k, about 0.3 lower at 25 km.

**Go:**
- 2 zooms by the agent are within 0.2 of zooming every area (train), and equal on dev3k.
- The answer head alone (zooming everything) matches the area model.

**But:**
- The learned policy beats the fixed rule (top-B Pinpoint areas) by only +0.0 to +0.3.
- Most of the gain needs no zoom at all: area-level choice over the global pool and clue places gives +3.9 / +3.8.
- So the action head has little to decide among areas alone. It needs costlier, more varied actions (clue reading on demand, live tools) to matter: steps 3 onward.

## Merged pipeline on the test set (2026-10-13, the user asked; lesson 73)

Frozen: answer head and budget-2 action head trained on all of train (retrained reranker setft_a), inputs normalised with train statistics.
The benchmark photos only went through the pipeline (`pinpoint_set extract --name bench --ckpt-tag setft_a`, `photo-set --tags full`,
`BASES=full scripts/retrained_pipeline.sh`, `hypothesis_agent --ckpt-tag setft_a --budgets 0 1 2 --apply-tag full`).
Pre-declared main config: 2 zooms by the agent, agreement pick. % within 1 / 25 / 200 / 750 / 2500 km (change vs Pinpoint at 1 / 25 km, 95% CI):

| | im2gps3k (1,480) | yfcc4k (2,315) | both (3,795) |
|---|---|---|---|
| Pinpoint top-1 | 21.8 / 48.9 / 64.6 / 79.6 / 89.8 | 14.8 / 29.8 / 47.0 / 66.1 / 81.3 | 17.5 / 37.2 / 53.9 / 71.4 / 84.6 |
| Retrained reranker top-1 | 22.4 / 50.5 / 65.8 / 79.3 / 90.0 | 16.4 / 33.3 / 49.5 / 67.9 / 81.6 | 18.8 / 40.0 / 55.9 / 72.3 / 84.9 (+1.3 [+0.6, +1.9], +2.8 [+2.0, +3.7]) |
| No zoom, agreement pick | 22.5 / 51.4 / 66.1 / 79.7 / 90.3 | 16.5 / 33.5 / 49.8 / 67.6 / 81.4 | 18.8 / 40.5 / 56.2 / 72.3 / 84.9 (+1.3, +3.3) |
| **2 zooms (agent), agreement pick** | 20.1 / 53.2 / 66.8 / 79.5 / 90.1 | 14.8 / 35.0 / 50.5 / 67.6 / 80.9 | 16.8 / **42.1** / 56.8 / 72.2 / 84.5 (-0.7 [-1.5, +0.2], **+4.9** [+4.0, +5.9]) |
| 2 zooms (agent), set pick | 22.6 / 52.1 / 66.4 / 79.5 / 90.1 | 16.6 / 34.5 / 50.4 / 67.6 / 80.9 | **19.0** / 41.3 / 56.7 / 72.2 / 84.5 (**+1.5** [+0.8, +2.2], +4.1 [+3.2, +5.1]) |
| All areas zoomed, agreement pick | 20.0 / 52.7 / 66.6 / 79.9 / 90.5 | 15.0 / 35.2 / 50.8 / 68.4 / 82.1 | 17.0 / 42.0 / 56.9 / 72.9 / 85.4 (-0.5, +4.8) |
| Previous pipeline, agreement pick | 20.3 / 52.5 / 65.5 / 79.7 / 89.9 | 14.7 / 35.0 / 50.9 / 67.9 / 81.1 | 16.9 / 41.8 / 56.6 / 72.5 / 84.6 (-0.6, +4.6) |

- **Merged = previous pipeline on test.** Paired, at 25 km: agreement pick +0.2 [-0.5, +1.0] over the previous agreement pick; set pick
  +0.0 at 25 km, -0.3 at 1 km vs the previous set-layer pick. On dev3k the merged pipeline looked +0.8 better (+5.5 vs +4.7).
- What it buys: the same accuracy with 2 zooms instead of a restricted search in every area, one network instead of a set layer + comparator,
  and a retrained reranker that alone gives +2.8 / +1.3 (25 / 1 km) on test.
- Matches train / dev3k in size (+4.7 / +5.5 there); yfcc4k gains more (+5.2) than im2gps3k (+4.3). Gains sit at 1-200 km; 750 / 2500 km don't move.
- The agreement pick costs 1 km (-0.7); the set pick keeps it (+1.5) for 0.8 less at 25 km. A soft kernel (lesson 72) is the obvious fix.

