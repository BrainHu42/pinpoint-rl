# Pinpoint-Set: look, zoom, rerank (2026-10-12; the user started an autonomous run on this plan)

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

## Open decision (user)
This moves the LLM / RL agent from the centre of the project (CLAUDE.md "Direction") to a later extension (step 5). If agreed:
1. Update CLAUDE.md's Direction and research plan.
2. Start step 1.
