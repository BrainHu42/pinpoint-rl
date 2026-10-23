# Paper notes (started 2026-10-23)

What we would put in a paper, and the findings behind it. One place for the claims, the numbers that support them, the negative results
worth reporting, and what is still missing. Sources: lesson numbers (`[N]` = `docs/archive/LESSONS_1-137.md#lN`; V / S lessons in
`LESSONS.md`; R lessons on branch `worktree-retrieval-track`; 140-141 on `worktree-error-viewer`). All numbers are dev (dev3k = MP16 photos
Pinpoint never trained on; commons26 dev = fresh 2026 Wikimedia Commons photos) unless marked test. Keep this file short; update it when a
result changes a claim.

## Working thesis
**Image geolocation as value-based decision making over tools and answers.** One calibrated value network prices every action in reward
units: answers ("commit to this place") and tool calls (zoom, read text, recognise). Because its answer values are calibrated probabilities
P(< r km), the scoring rule is an input at inference: the same network names the exact place when it is confident and hedges toward a region
when it is not. It sits on top of a strong retrieval system (Pinpoint) and improves it at every distance scale but 1 km.

## Candidate contributions and their evidence

| # | Claim | Evidence (dev unless marked) | Status |
|---|---|---|---|
| C1 | Structure over retrieval (agreement areas, restricted search, retrained reranker, an answer head over areas) beats the released Pinpoint at 25 km | **Test**, original protocol: 46.4 -> 51.2 % < 25 km [109]; commons26 test 16.8 -> 24.2 [113] | Done |
| C2 | One value network over tools and answers matches the two-model system (answer head + REINFORCE) | dev3k 47.40 / 30.56 vs 47.23 / 30.67, commons26 dev 25.08 / 7.68 vs 24.97 / 7.68 (% < 25 / 1 km); cost curve level within noise (V6) | Done |
| C3 | Calibrated multi-scale answers with hedges move the coarse columns, which no prior stage moved | Table weights: +3.6 / +4.0 at 750 km, +2.9 / +4.7 at 200 km, 25 km -0.08 / +0.41, 1 km -2.0 / -0.7 (dev3k / commons26 dev; V8) | Dev only |
| C4 | Method: with tool costs ~1% of the reward, fitted Q-iteration's overestimation swamps the costs; paired Monte Carlo targets fix it | Policy spend at the highest cost 6.7 units (plain FQI) vs 0.6 REINFORCE; with paired MC targets spend falls with lambda like REINFORCE's (V6) | Done, small |
| C5 | A second image model per area (PE-Core-G) adds to the answer head where SigLIP2 and MegaLoc share blind spots | +2.66 dev3k / +2.70 commons26 dev at 25 km with PE-G + SigLIP2 cosine + PE-G region probe columns (V5) | Done, not yet in the value network's tables |

## Headline numbers

**Primary benchmark (test, original protocol: the query photographer's photos kept in the index), % < 1 / 25 / 200 / 750 / 2500 km**

| | both | im2gps3k | yfcc4k |
|---|---|---|---|
| Released Pinpoint | 28.9 / 46.4 / 60.7 / 75.5 / 86.7 | 21.8 / 48.9 / 64.6 / 79.6 / 89.8 | 33.5 / 44.8 / 58.2 / 73.0 / 84.8 |
| Pipeline, RL lambda 0 [109] | 29.1 / 51.2 / 62.9 / 75.5 / 86.9 | 22.4 / 52.8 / 66.4 / 78.9 / 90.3 | 33.3 / 50.2 / 60.6 / 73.3 / 84.8 |
| + `--one-km 0.5` [111] | 31.6 / 50.6 / - | 23.2 / 52.4 / - | 37.1 / 49.5 / - |

These test numbers are on the eval halves (3,795 photos); the test set is now all 7,533 photos of both benchmarks. The 200-2500 km columns
are flat against the released model: C3 targets exactly them. commons26 test (continent-balanced, 4,046): released 4.6 / 16.8, pipeline
6.1 / 24.2 (% < 1 / 25 km; [113]).

**Multi-scale answers (V8, dev, 3 seeds, % < 1 / 25 / 200 / 750 / 2500 km, everything revealed)**

| | dev3k | commons26 dev |
|---|---|---|
| Step 1 weights ([< 25] + 0.5 [< 1]) | 30.46 / 47.38 / 58.03 / 72.64 / 84.40 | 7.61 / 24.77 / 45.21 / 70.62 / 85.70 |
| Table weights (0.5 / 1 / 1 / 1 / 1), with hedges | 28.44 / 47.30 / 60.97 / 76.23 / 86.99 | 6.89 / 25.18 / 49.91 / 74.65 / 87.07 |
| Table weights, no hedges | +0.47 at 750 km | +0.85 at 750 km |
| Fixed gated switch (threshold tuned on the other set) | +1.02 at 750 km, -0.53 at 25 | +0.89 at 750 km, -0.07 at 25 |

Hedges are chosen on 44-49% of photos. Headroom (V7): on photos where the network doubts a 25 km hit, some pool candidate is < 750 km on
~90%, the answer on 52-61%; best-of(answer, probe) per photo would reach ~83% at 750 km on both sets. Calibration: dev3k P(< r) within
0.03-0.04 of observed at every r; commons26 dev overconfident (ECE 0.06-0.13), training is MP16 only.

**What moved 25 km, component by component** (train CV / dev3k): candidate agreement +1.0 / +2.0 [68]; areas as hypotheses with restricted
search +2.9 / +3.8 [69]; fine-tuning Pinpoint's reranker +3.1 / +3.3 [70], [74]; an answer head over areas +4.9 / +5.4 [71]; MegaLoc per
area +0.5-1.1 [93]; clue reading +1.0 dev3k, +4.4 commons26 [76], [113]; PE-G (+ SigLIP2 + region probe) per area +2.7 (V5); one head over
areas and text places, commons26 dev +1.8-2.2 (140, 141).

## Findings that shape the paper (including negative results worth reporting)
1. **Where the losses are** (dev3k, [129], [130], R1, R5): retrieval misses 26.5 pts (no pool candidate within 25 km; mostly photos without
   geography in the pixels, located in practice only by the photographer's own uploads), right candidate never an area 11.8, wrong area
   chosen 14.4 (cross-region, median 381 km off). The reachable ceiling is the area oracle (61.7), not the pool oracle (73.1). Misses are depth,
   not absence: the right photo is usually in the index but ranked low (R1). On commons26 dev retrieval misses are 49.1 pts.
2. **Structure over retrieval works; re-describing the photo doesn't** (main point 1): choosers, comparators, attributes, map features, photo
   descriptions and new head inputs from the same embeddings all land within +-1 pt. The answer head is neither data-limited (R8) nor
   structure-limited (R11).
3. **Look-alike verification saturates** at AUC ~0.6-0.79 (main point 2): retrieval picked the wrong areas for looking like the photo, so
   checking them visually against the photo can't separate them. Same for text in the index photos (V4) and reverse geocoding (V2).
4. **Text helps on fresh photos, not on MP16** (main point 3; V1, V3): on MP16 the names the 4B reads sit where retrieval already succeeds;
   on commons26 reading is the largest single gain (+4.4), and the limit there is ranking candidates, not finding them.
5. **Local open VLMs lack calibrated place knowledge beyond retrieval** (main point 4): Qwen 4B to Qwen2.5-VL-72B and Gemma 3 27B, in every
   format tried and trained on ground truth, are near chance on the head's mistakes. A paid LLM given only town names did +0.94 at 25 km
   [105]: the signal exists but our open models don't have it. Supports the design choice of a small value network over cached evidence
   rather than an LLM judge.
6. **A learned controller only pays when outcomes change the next choice** ([94], [98], V6 step 2): over today's cached actions the RL policy
   ties "take everything"; with per-model recognise actions and realistic latency costs it only ties a simple gate on dev3k (<= +0.3 at matched
   cost) and loses 1.4-2.2 pts to it on commons26 dev (a zoom habit learned on MP16); choosing all actions upfront does as well. The claim "the
   controller adds something" is not yet supported; it needs sequential tools (re-reads, candidate creation).
7. **Search results don't verify search proposals** (S1-S3): constrained searches find right regions beyond the area oracle (~+6 pts net of
   luck) but their results can't tell them apart from look-alikes; text-proposed places verify much better (AUC 0.94) but are blind where the
   gallery is sparse.
8. **Better retrieval needs a reranker that can use it** (R17, reported by the retrieval-track session 2026-10-23, not yet in its pushed
   lessons): a trained PE-G projection beats Pinpoint's contrastive search by +5.5 recall at 25 km, but its candidates move the reranker's
   top-1 only +0.10 because the reranker can't see PE-G. The PE-G-aware answer head (C5) is where they could pay.
9. **Protocol matters**: photographer exclusion in training helped the photographer-excluded metric and hurt the original protocol ([74],
   [108]); training only on photos the encoder never saw (bucket 99) is required ([54], [75]).

## Method and protocol notes for the paper
- Benchmarks: im2gps3k + yfcc4k under the original Pinpoint protocol (photographer's photos kept), all 7,533 photos as test; commons26 (fresh
  2026 photos, dev 977 / test 4,046) as the out-of-distribution set. All choices on dev3k / commons26 dev; test once, on request.
- Reward: ground-truth distance only (no distillation from Gemini; Gemini is a benchmark). Base retrieval: Pinpoint (released model, exact port,
  reranker retrained on its own data). Tool costs from measured latency where available (PE-G 50.7 ms / image vs MegaLoc 5.5 on an L40S, V6).
- Reporting rules: measure against the current pipeline, not weaker baselines; pre-declare bars; report fixes and breaks; check candidate-list
  oracles against another photo's truth (luck, [120]); use the area oracle, not the pool oracle, as the reachable ceiling ([130]).
- Value network details (V6, V8): `_net` encoder over area tokens; logistic P(< r) per answer; lambda as an input; states from random reveals
  plus the network's own epsilon-greedy rollouts; paired Monte Carlo targets; a listwise term on the 1 / 25 km outputs of area answers only
  (needed to match 1 km); hedge answers from the PE-G region probe (2,048 cells) and the pool's score mass.

## Related work to check before writing
- Classification with costly features / active feature acquisition (Janisch, Pevny & Lisy, AAAI 2019; Shim et al. 2018; Li & Oliva 2021):
  the closest framing to C2 ("classify" as an action next to feature purchases). Our differences: an open-world answer set that grows during
  the episode, real retrieval / VLM tools, calibrated multi-scale answers.
- Geolocation by classification vs retrieval and hybrids (PlaNet, PIGEON, GeoDecoder-style hierarchies, Pinpoint): C3 combines cells and
  retrieval through decision theory inside one network. Check for prior decision-theoretic or expected-score answer selection in geolocation.
- LLM agents with tool-use RL (Search-R1 style GRPO on the model's own tool calls): the frontier for "RL agents with tools"; position the value
  network as the decision layer over costly tools, not as a replacement.

## Missing before a paper
- **The primary table on test with C3 and C5** (user's go-ahead needed): the multi-scale network with V5 columns, original protocol, all three
  benchmarks, against the released Pinpoint and the current pipeline.
- **The 1 km trade-off**: a weight sweep (1 km weight 0.5 -> 1 -> 2) to show how much 1 km comes back and what coarse gain it costs.
- **Calibration on fresh photos**: Commons training states (the combined-head session's `commons_train` pool) for commons26.
- **Gemini, column by column**: large VLMs tend to be strong at country / region scale, exactly where C3 gains; our Gemini numbers per column
  are needed (OpenRouter credits ran out 2026-09-29: check which Gemini results are stored before planning new calls).
- **Seeds and SEs** for every headline (V8 is 3 seeds; V7 is 1 seed with the switch fraction chosen in-sample).
- **C2's open claim** (a controller that beats fixed recipes) needs outcome-dependent tools: VALUE_PLAN steps 4-5.
