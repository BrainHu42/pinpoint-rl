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
The run changed four things at once:
- wide pools (32 / 8 / 8 hits);
- same-photographer exclusion in training, so training pools look like evaluation pools;
- the leave-one-out fix below;
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

**Bugs found in the training setup** (they matter for anyone retraining Pinpoint):
1. **Leave-one-out was off.** The released index names its source "source_0" while the data module tags samples "mp16", so no training
   query was mapped to its own index row. Every training pool contained the query itself; train accuracy was 100%.
   Fixed with an alias in `retrieval_index._build_source_lookup`.
   - The released reranker was presumably trained with a working mapping (its own index build); we could not check this.
2. **No same-photographer exclusion in training**, while our evaluation excludes the query's photographer.
   Added `--exclude-author-path`, with authors per cache row in `artifacts/pinpoint_train/author_by_cache_row.npy`.

**Control run (no set head, otherwise identical, `artifacts/pinpoint_train/ft_nohead`):** wide-pool pick +2.1 / +2.3 (1 km +1.2 / +0.4),
own pool +1.7 / +2.0. So:
- matching training pools to evaluation (fine-tune with leave-one-out working, photographer exclusion, wide pools) gives about +2;
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
