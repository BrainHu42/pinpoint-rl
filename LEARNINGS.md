# Learnings so far (Sep - Oct 2026)

What the base-model diagnostic, SFT, single-turn GRPO and the follow-up headroom checks established, and what they
rule out. The current plan lives in CLAUDE.md ("Research plan"). Numbers are % of
photos within 1 km / 25 km of the truth, with same-photographer gallery images excluded. "Full eval halves" = every
im2gps3k + yfcc4k eval-half query (n = 3,795; SE about 0.6 pts at 1 km, 0.8 at 25 km). "300 subset" = the stratified
study subset (SE about 2 / 3 pts; don't trust gaps under ~4 pts there).

## Summary (read this first; details in "Main lessons" below, numbered)

**Where we are (2026-10-06; next step in `HANDOFF.md`).** Against our one-step reranker's top-1 (the baseline until 2026-10-07; see below), the best confirmed gain is **+0.9 points** [+0.3, +1.6] top-1 within 25 km,
from a fine-tuned 4B comparator (query photo + exemplar photo of a candidate -> same place?) combined with the reranker's rank (lessons 24, 29, 40, 41).
On the 3,713 im2gps3k + yfcc4k eval-half photos (placeholders dropped), nothing fitted on them:

| | < 1 km | < 25 km | < 200 km |
|---|---|---|---|
| One-step reranker top-1 (baseline until 2026-10-07) | 17.1% | 38.9% | 56.4% |
| Pinpoint attention reranker top-1, photographer-filtered (lesson 51) | 17.9% | 38.1% | 55.1% |
| Pinpoint attention reranker top-1, no filter (baseline from 2026-10-07; not comparable with the filtered rows) | 29.5% | 47.4% | 61.9% |
| + comparator-b, 4 exemplars, combiner trained on dev | 17.2% | 39.9% | 56.9% |
| Oracle over the reranker's top 8 | 30.1% | 52.5% | 70.1% |
| Oracle over the whole ~17-candidate pool | 34.2% | 60.2% | 80.3% |

The headroom is in choosing (13.5 pts at 25 km to the top-8 oracle). How much a model could recover is open: a rough by-eye look at 30 photos (one reader, lesson 35)
put it near 3 pts, but Gemini closed ~2/3 of the gap on the 300 subset (lesson 9) and our open models, zero-shot, close none of it. Place knowledge is the likely
lever; we are testing it on the 4B first and scaling up only on a promising signal.
Every lever tried lands between -0.5 and +1 point. Wikimedia (lesson 42): reranker top-1 25.0% within 25 km, top-8 oracle 38.9%, comparator +0.4 to +0.8 (balanced subset): no wikimedia-specific gain.
Near-misses (21.9% of benchmark photos, top-1 1-25 km off; lessons 43-46): 1 km oracle over local candidates 30.1% vs 17.1% today; a comparator trained for the band
reaches AUC 0.8 but the combined top-1 gain at 1 km is +0.8 [+0.2, +1.3] on the benchmarks, 4x the band data (lesson 45) gives the same (+0.9), and better
combiners (incumbent-relative features, a switching margin; lesson 46) give the same (+0.7). The single-exemplar visual signal is the limit, not data or calibration.

**Baseline change (2026-10-07, lesson 51).** "The reranker" in lessons 1-50 is our one-step reranker (fitted on the benchmark tune halves), with the
photographer filter. From now on the baseline is Pinpoint's own attention reranker, and benchmarks are evaluated **without** the photographer filter, as
in prior work (user's decision): 29.5 / 47.4 / 61.9 on the same 3,713 photos (yfcc4k gains 19 pts at 1 km from same-photographer gallery photos;
im2gps3k and wikimedia are unaffected), wikimedia 8.5 / 24.7 / 57.5. With the filter on both, the attention reranker ties the one-step reranker
(17.9 / 38.1 / 55.1 vs 17.0 / 38.9 / 56.4), so the conclusions above stand; the gains above are relative to the filtered one-step reranker.

**One line per lesson.**
- 1-4 (SFT / GRPO): the 4B reaches reranker parity, not better; GRPO sharpens but doesn't discover; the LLM wins only on photos with readable text.
- 5-10: "where to search" is not the bottleneck; GeoNames needs fuzzy matching; rewriting retrieval queries adds < 2 oracle pts; per-candidate evidence helps an
  untrained chooser a little; go/no-go must be measured beyond what we already have (the shown-candidate oracle), on the full eval halves.
- 11-19 (stage 1, new evidence): 4B / 27B search queries, geotagged Wikipedia, transcribed text, an 80M-place name index add 1-3 oracle pts over the pool and
  nothing for a learned chooser; the evidence is right on the photos retrieval is already right on (17, 18, 19).
- 20: train / dev / val audit is sound; Flickr "no longer available" placeholders (flagged) and near-duplicates make absolute numbers ~1.5 pts optimistic.
- 21: the VLM's photo attributes are accurate but redundant with what retrieval encodes.
- 22-23: a zero-shot exemplar judge separates the true place from far away, not from its neighbours; its skill is mostly exemplar similarity.
- 24-26, 29: the fine-tuned comparator is the only positive: +0.7 [+0.3, +1.2] on all benchmark photos; pairwise and 25 km variants are no better.
- 27-28: bigger zero-shot choosers (9B, 27B) and nearby Wikipedia text stay below the reranker.
- 30: the gallery isn't thin; reranking deeper than the pool (50 clusters) is worse (-0.8).
- 31-32: a region-diversified pool adds ~2 oracle pts at matched budget; the region head is the bottleneck and a bigger one barely helps.
- 33-34: the models' own answers and nearby place categories add nothing beyond rank + comparator.
- 35: by eye, of 30 choosing photos 9 are near-misses at the 25 km line, 10 have nothing place-specific, 6 show a specific place, 4 need knowledge, 1 is label noise.
- 36-37: coarse thresholds (200 / 750 / 2500 km) have as much headroom and the same null results; the 4B's candidate-free guess is far below retrieval.
- 38: a geo-aware adapter on SigLIP2 adds ~+1 pt candidate recall over the region prior; Pinpoint's own retriever has worse top-10 recall than raw + prior.
- 39-41: several exemplars per candidate help ~+0.3-0.6 without retraining; comparator-b (3x pairs) adds ~+0.3 more; full benchmark test +0.9 (the rule dev
  picks is null, +0.4; the post-hoc best-of-4 rule +1.7 is a hypothesis, not a result).
- 42: wikimedia is harder (top-1 25.0% < 25 km) with the same choosing headroom; the comparator gives +0.4 to +0.8 there, nothing wikimedia-specific.
- 43: near-misses are the largest 1 km target; 93% have a gallery photo within 1 km of the truth, 73% among cached neighbours, 59% among our local candidates.
- 44-45: a near-band comparator (c: 55k rows, d: 219k rows) lifts within-photo AUC 0.735 -> ~0.80 but top-1 < 1 km only +0.8 / +0.9; 4x data adds nothing.
- 46: combiner variants (relative-to-top-1 features, switching margin) chosen by CV on held-out train photos: +0.7; fixes and breaks move together.
- 47-49 (map search pivot): searching within 25 km of the top-1 reaches 70-82% of near-misses within 100-300 photos (fixed list 56%), but SigLIP2 (AUC
  0.54), keypoint matching (0.52; it finds what is shown, not where the camera stood) and the comparator (0.58) can't separate the right place from its
  look-alikes; 4 exemplars on the fixed list: AUC 0.84, +0.9 at 1 km. Directions 1-3 below are tested and closed for now.
- 50 (open): knowledge SFT, teaching the 4B to name a photo's place and scoring each candidate's name. Base-4B control is at chance (AUC 0.50-0.53,
  -0.7 at 25 km); labels and 250k training photos are built, training has not run yet.
- 51: Pinpoint's attention reranker is the baseline from 2026-10-07, run unfiltered like prior work (29.5 / 47.4 / 61.9); with the photographer filter
  it ties our one-step reranker (+0.9 at 1 km, -0.9 at 25 km). The filter moves yfcc4k by 18 pts at 1 km and nothing else.
- 52: by eye, of 140 random attention-reranker misses (> 25 km; 70 per benchmark), a person places 8 within 25 km (+1 low confidence): landmarks and
  readable text (a race banner, a plaque, a dealer URL); half of them were never among its 12 candidates. Of 45 misses with a right candidate, a person would
  pick it for ~4. Most misses have no place signal (73) or only region-level signal; text sometimes misleads both (a Polish ad in Atlanta).
- 53: im2gps3k and yfcc4k flaws, and a new benchmark (commons26, 5,561 Commons photos taken after 2026-07-01). im2gps3k has 198 photographers
  (design effect ~6: its error bars are ~2.5x too small) and our tune / eval split shares 97.5% of them. The same top-1 MP16 retrieval gets 37.6 / 38.1%
  within 25 km on im2gps3k / yfcc4k but 10.0% on commons26 test (13.0% on locatable photos); 1 km 14.9 / 29.5 vs 3.2.

**Directions as ranked on 2026-10-05.** Status 2026-10-06: 1 and 2 were tested and failed (lessons 47-49), 3 waits for a tool with a strong
signal, 4 is open. The open experiment is knowledge SFT (lesson 50).
1. **Geometric verification as a tool** (never tried). Keypoint matching + RANSAC inliers (SuperPoint / DISK + LightGlue, pretrained, no training) between the
   query and 2-4 exemplars per local candidate. Near-binary when the same structure is visible and silent otherwise, which is what flipping near-misses without
   breaking exact photos needs (today ~8 fixes per 5 breaks). Go test, a few hours on dev: within-photo AUC on the subset where it fires, and the 1 km gain of
   the lesson-46 combiner with inlier features; go if > +2 at 1 km on the benchmarks. Weak on nature / indoor / no-overlap photos.
2. **Candidate recall in the near band** (pairs with 1). Search the whole gallery within 25 km of the top-1, not the top-300 neighbours: the ceiling goes from
   59% to up to 93% of near-misses. Only worth it with a precise scorer, since a noisy one breaks more as candidates grow.
3. **RL over tools** (the project's framing, still untested for stage 2): a policy that decides which candidates to verify (comparator, matcher, search) under a
   budget and when to keep the top-1. Do it after 1 gives a signal stronger than AUC 0.8; RL cannot create a signal the tools don't have.
4. **Scale the comparator** (9B / 27B LoRA) only if 1-3 show the 4B pipeline is the bottleneck.
The 4-exemplar near-band test that was pending here ran (lesson 48): +0.9 at 1 km, below its +1.3 bar, so the comparator-over-the-fixed-list line is closed.
Not promising (tested): more comparator data, combiner tweaks, zero-shot choosers up to 27B, new text / attribute evidence (stage 1), deeper pools.

**Traps worth remembering.** Choose scorers on dev, not by looking at val or the benchmarks (41). A judge's Yes/No logit level can drift in training; always
check the failed-score count and restrict output to the answer tokens (40). Compare against the reranker top-1, not current greedy; the pool oracle is the
baseline for new evidence (10, 13).

## Headline numbers (full eval halves)

| System | Greedy | Single sample | Best of 8 | Right answer among shown candidates |
|---|---|---|---|---|
| One-step reranker #1 | 16.7 / 38.2 | | | |
| sft-34k (reranker top-10 in prompt) | 15.8 / 37.6 | 11.5 / 33.0 | 22.3 / 48.0 | 30.8 / 53.6 |
| sft-34k-retrieval (all ~17 pooled candidates, retrieval order, evidence; no reranker) | 16.3 / 37.2 | 12.6 / 32.8 | 22.3 / 48.2 | 33.4 / 59.0 |
| grpo-kl step 200 (from sft-34k-retrieval) | 16.9 / 37.9 | 15.9 / 36.8 | 21.7 / 45.1 | 33.4 / 59.0 |

300 subset only: base Qwen3.5-4B greedy 12.0 / 38.0, best-of-8 21.7 / 49.7; Gemini 3.8 Flash with top-10 candidates
27.7 / 55.7 (best-of-8 31.3 / 58.0; oracle over its samples + candidates 40.7 / 65.7).

## Main lessons

1. **Selection among one-shot retrieval results is saturated.** SFT and GRPO both land at reranker parity, while the
   right answer is in the shown list for 33 / 59%. The model and the reranker agree on >90% of photos; disagreements
   roughly cancel (model-only right 2.6 / 3.9%, reranker-only 3.0 / 4.8%).
2. **SFT lifts the 4B model from below the reranker to parity, not past it.** On the 300 subset, greedy rose +4–6 pts
   over base, mostly by learning to follow the reranker (#1 copied 41% → 67%). More data (5k → 34.5k) lowered val loss
   (0.349 → 0.328) but MP16-val accuracy stayed flat. Dropping the reranker (retrieval order + evidence numbers) costs
   nothing and raises the candidate ceiling (+2.6 / +5.4).
3. **GRPO sharpens; it doesn't discover.** It converts best-of-8 headroom into single-sample accuracy (+3–4 pts) and
   shrinks best-of-8. With lr 2e-5, no KL and T=0.7 it collapsed (entropy 0.13 → 0.025, 44% of groups with identical
   rewards, best-of-8 −9 pts at 25 km). lr 5e-6, KL 0.04 and T=1.0 kept entropy at ~0.17 but greedy only moved +0.6.
   Geo-R (AAAI'26) reports the same "vanishing advantages" problem and fixes it with hard-photo filtering.
4. **The LLM's advantage is real but narrow.** On photos with readable text it beats the reranker (<25 km 64.9 vs
   60.9; wins 6.9% vs loses 2.9%). On generic photos (no text, no nameable place; nature, events, interiors) it
   loses (18.3 vs 21.0). The model's off-list guesses are almost always wrong (11 of 270 within 25 km).
5. **"Where to search" is not the bottleneck.** Retrieval restricted to the true region
   jumps to 64.4 / 48.9% within 25 km (im2gps3k / yfcc4k, vs 38.4 / 19.7 unrestricted). The VLM picks the right region
   more often than the region head (62 vs 58%; 46 vs 42%), yet retrieving inside the VLM's region scores *below* the
   VLM's own answer (45.6 vs 48.3; 27.0 vs 30.1), and the oracle over its sampled regions adds only ~6 pts. Whole-image
   retrieval matches overall scene appearance (e.g. "crowd in red" → St. Louis for a Kauffman Stadium photo).

6. **Offline geocoding needs fuzzy matching.** Exact GeoNames lookup of Gemini's confident place names: 55% match at
   all; best match within 1 / 25 km for 27 / 47%, most-populous match 22 / 39%. Misses are name variants.

7. **Rewriting the retrieval query doesn't add candidates the list is missing (go/no-go, 2026-09-29).**
   - Text: SigLIP2 text queries of Gemini's confident place names (645 photos) land within 1 / 25 km for 0.8 / 3.1%
     (best of top-10 clusters 2.5 / 6.2%) vs GeoNames 22 / 39% and whole-image retrieval 33 / 70%. SigLIP's text tower
     knows concepts ("a stadium"), not specific places ("Golden Gate Bridge" → Finland, Java, Russia).
   - Crops (10 fixed crops per eval photo, full eval halves): at equal budget, best of 10 crop top-1s 18.3 / 36.7 vs
     whole-image top-10 24.9 / 45.0; the centre crop alone is worse than the whole image (9.9 / 23.9 vs 12.4 / 27.0).
     Adding the 10 crop guesses to the ~17 shown candidates raises the oracle only 33.4 / 59.0 → 34.8 / 60.6.
   - So the "learning what to search for" plan failed its go/no-go. The missing accuracy is in *choosing* among
     candidates retrieval already returns (oracle 59% vs greedy 38% within 25 km), not in finding new ones.

8. **Per-candidate evidence helps an untrained chooser a little (`experiment/evidence_test.py`, base 4B zero-shot,
   reranker top-10, full eval halves).** Candidates only 10.5 / 33.9; + up to 4 GeoNames landmarks within 1 km
   (60% of candidates have one) 12.1 / 35.2; + one 256 px MP16 exemplar photo per candidate (70% the most similar
   retrieved photo within 1 km, else the nearest photo) 11.7 / 35.2. Both gains are consistent on im2gps3k and yfcc4k.
   Photos cut off-list guesses (11% → 5%) and raise model-only wins (3.1 → 3.9%); reranker-only wins fall 7.4 → 6.4-6.8%.
   Open question: does training (SFT on these prompts) turn this into a large gain? That is the proposed next run.

9. **The selection cap is the chooser's knowledge, not the candidate list.** With the same kind of prompt (photo +
   top-10 candidates, no extra evidence), on the 300 subset: reranker 18.7 / 45.0, SFT 4B 17.7 / 43.0, Gemini 3.8
   Flash 27.7 / 55.7, candidate oracle 32.0 / 59.7. A model that knows what places look like closes ~2/3 of the gap.
   For a 4B model, per-candidate evidence (lesson 8) or tools are the substitute for that knowledge.

10. **Method: define go/no-go against what you already have.** The query-rewriting go criterion ("combined oracle beats
    current greedy by ≥ 8 pts") would have passed trivially, because the existing candidate list already beats greedy by
    21 pts. The right comparison was what new queries add beyond the shown-candidate oracle (< 2 pts). Also use the
    full eval halves for decisions; 300-subset differences flipped sign several times.

11. **Base-4B search queries + six SigLIP2 evidence photos don't help (`experiment/query_evidence.py`, 2026-10-01).**
    1,000 MP16 val photos, reranker top-10 as text, base 4B, no training. <25 km: no search 30.8, ask again 31.5,
    whole-image evidence 32.2, caption 31.3, visual-clue queries 31.1, geographic queries 31.2 (reranker top-1 34.2).
    Queries vs whole-image: −1.0 / −1.1 pts, CIs include 0 (gate was ≥ +2). Why:
    - Visual queries are concepts ("ancient stone pyramid ruins"): 3.5% of their photos are within 25 km of the truth.
    - Geographic queries name a shown candidate's city 83% of the time, so they re-fetch the candidates.
    - New coverage (evidence near truth, no shown candidate near it) is 1.3-3.6% of photos; the model converts ~none.
    - The model ignores correct evidence: in ~25% of photos with an evidence photo near the truth it answers wrong
      (e.g. the exact Pittsburgh rhino sculpture retrieved, answer "Jeff Koons Rhino, Washington DC").
    - Prompt facts: a `{"lat": 0.0, "lon": 0.0}` template gets echoed as an answer (35-60%); use `<latitude>`. Without
      "Think briefly (under 120 words)" the base model rambles past 600 tokens.

12. **Geotagged Wikipedia is a discriminating offline backend; the 4B's queries are the bottleneck
    (`experiment/wiki_backend.py`, 2026-10-02).** 1.15M English articles with coordinates, BM25 + bge-base dense.
    - Gemini's 645 high-confidence place names (benchmark photos): Wikipedia top-1 within 25 km 82%, top-3 89-90%
      (GeoNames 39%, SigLIP2 text 3.1%), and +7.3 pts beyond the shown-candidate oracle (86.7 → 94.0).
    - Same 645 photos, the base 4B's three geographic queries: +0.8-1.1 pts beyond the oracle. 71% of its queries name
      a shown candidate's city; its landmark guesses are often wrong (Bargello → Siena Palazzo Pubblico).
    - Dev MP16 queries at equal budget (6 results per photo): new coverage 0.8-1.5% vs SigLIP2 1.3-3.6%.
    - Prompt vs model: the 4B with Gemini's exact labelling prompt (no candidates, one most-specific name) gives no name
      for 37%, finds the truth within 25 km for 42%, and adds +1.7 pts beyond the oracle (best of 8 samples at T=1:
      +2.0). It names famous landmarks right (Sydney Opera House, Wat Arun), which retrieval already finds, and misses
      or mislabels the long tail (Bargello → Palau de la Generalitat). The prompt explains little; the model is the cap.
    - Against the reranker top-1 (the baseline; oracle = perfect choice between reranker and search, <25 km):
      MP16 dev 34.2 → 40.6-42.1 with geographic-query search (SigLIP2, Wikipedia BM25 or dense), 35-37 with visual or
      caption queries, vs 47.2 for the shown top-10. 645 benchmark photos: 77.4 → 79.8-82.0 with the 4B's searches,
      vs 86.7 for the shown top-10. Actual answers with evidence stay below the reranker (lesson 11).
    - Model size (645 benchmark photos, <25 km, oracle choice between reranker top-1 and Wikipedia search, reranker
      alone 77.4): 4B 79.8 greedy / 81.6 best of 8; Qwen3.6-27B (Q4_K_M, llama.cpp, thinking off) 82.0 greedy / 83.6
      best of 8, 80.9 with our candidate-conditioned geo-query prompt; Gemini 89.8. So the gain from search roughly
      doubles from 4B to 27B (+2.5 → +4.7 greedy) and is still about half of Gemini's (+12.4). SE ~1.5 pts, so
      differences of ~2 pts are borderline.
    - So search can add ~7 pts of candidates the list misses, but only with names the 4B doesn't produce (lesson 9's
      knowledge cap again). Article coordinates are the entity centre, so <1 km is lower than for photo retrieval.

13. **Stage-1 baseline (`experiment/stage1_eval.py`, 2026-10-02): over the whole candidate pool, extra retrieval adds ~1
    pt and the 4B's queries add no more than extra whole-image results.** Metric: oracle accuracy of (pooled candidates
    + retrieved coordinates) at <1 / <25 / <200 km; six results per photo; photo set = 1,000 MP16 dev (`dev`) and 1,000
    benchmark eval-half photos (`val`, 500 im2gps3k + 500 yfcc4k). Prompt: the 4B names 3 places, no candidates shown.
    - The pool (~17 candidates/photo) is the baseline, not the reranker's top-10: 53.0 / 64.5% <25 km (dev / val) vs
      47.2 / 59.0 for the top-10 and 34.2 / 42.8 for the reranker top-1; <200 km 72.2 / 82.8. The gap that remains is
      choosing (stage 2), not finding.
    - Gain over the pool at <25 km: whole-image control (6 results) +1.1 / +1.0; 4B queries via SigLIP2 +1.0-1.5,
      Wikipedia BM25 / dense +0.6-1.2 (at 4.2 results per photo, since the 4B often returns fewer than three distinct
      places); matched-budget whole-image +0.9; SigLIP + Wikipedia (12 results) +1.2-2.0. Queries minus matched
      whole-image: +0.1 to +0.6, intervals touching 0. At <200 km the gains are the same size.
    - Prompt yield (queries per photo, dev): as written 2.10; "3 different places" 1.73 (37% of photos get no query);
      placeholders in the JSON template 1.52 (44% answer in prose and run out of tokens). Keep the empty-string template.
    - 4B queries name a shown candidate's city only 12-18% of the time without candidates in the prompt (71-83% with).

14. **Evidence informativeness (`stage1_eval.py`, 2026-10-02): the 4B's search evidence discriminates between candidates
    but adds almost nothing to a simple combiner.** Oracle accuracy only counts new candidates; evidence can also say which
    candidate to trust. Support of a candidate = results within 25 km of it; measured on photos whose pool holds a correct
    candidate (dev 530, val 645), prompt v2.
    - Specificity: text evidence lands on correct candidates 26-34% of the time vs 6-9% on wrong ones (SigLIP2 30 / 9,
      Wikipedia dense 26-31 / 6-7), a 3-5x ratio; matched whole-image results 56-57 / 25 (2.3x, same retrieval the
      reranker already uses).
    - Coverage: text evidence touches the pool for only 41-50% of photos (whole-image 90-100%).
    - Within-photo AUC of support at separating correct from wrong candidates: whole-image 0.66, text 0.58-0.61;
      reranker rank 0.74-0.75.
    - One-parameter combiner, score = -rank + w * support, w fitted on dev: top-1 <25 km changes by 0.0 to +0.6 pts
      (Wikipedia dense, w = 2: dev +0.6, val +0.6, CIs include 0; 11-13 fixed vs 5-7 broken). Other arms keep w = 0.
      Variants v2b / v2c touch less (29-40%) and gain nothing.
    - Caveat: a linear combiner on rank is crude; a learned chooser (stage 2) could use the discrimination better. The
      untrained 4B with exemplar photos or nearby place names gained ~+1.3 pts (lesson 8).

15. **A learned chooser can't use the 4B's search evidence either (`experiment/evidence_ranker.py`, 2026-10-02).** The
    pipeline's per-candidate MLP reranker (same recipe), refit on 34,523 MP16 train photos (bucket 99, train
    photographers; 72,914 queries, 2.1 per photo) with and without evidence features (log1p count of SigLIP2-text or
    Wikipedia-dense results within 1 / 25 / 200 km of each candidate + result count); 3 seeds; top-1 at <25 km:
    dev (MP16 val) A 34.7, + SigLIP2 34.8, + Wikipedia 35.2, + both 34.8, both shuffled across photos 35.0;
    val (benchmarks) A 43.3, 43.1, 43.0, 43.3, 43.2. All changes within +-0.5, intervals include 0, the shuffled control
    does as well as the real evidence, and <200 km is flat or lower. The refit baseline matches the original reranker
    (34.2 / 42.8). Evidence touches 10-13% of candidate cells. Together with lessons 11-14: with this 4B and these
    backends, neither new candidates (+~1 pt oracle) nor candidate discrimination (no learned gain) is there to find.
    Untested: richer evidence features (similarity scores, article text), a larger or better searcher, an LLM reading
    the articles.

16. **A stronger searcher doubles the oracle gain but still doesn't help a simple chooser (Qwen3.6-27B Q4_K_M via llama.cpp,
    same prompt v2, dev and val photos; no Gemini).** The 27B writes an analysis before its JSON: at max_tokens 150, 82%
    of photos got no query (invalid comparison); at 700, 90-93% give three distinct places (2.7-2.8 queries per photo vs
    2.1 for the 4B; 21-25% name a shown candidate's city vs 12-13%).
    - Oracle gain over the pool at <25 km (dev / val): 27B Wikipedia dense +2.5 / +2.0, BM25 +2.4 / +1.9, SigLIP2 +2.1 /
      +0.9, SigLIP2 + dense (12 results) +3.3 / +2.4; 4B dense +1.0 / +1.1, combined +1.2 / +2.0. Against
      matched-budget whole-image results the 27B adds +1.1 to +1.6 (4B: +0.1 to +0.3, combined +0.3 / +1.1).
    - Informativeness: the 27B's evidence touches 53-62% of pools (4B 41-50%), AUC 0.61-0.64 (4B 0.58-0.61), on correct
      vs wrong candidates 35-43% vs 9-14%. One-parameter combiner top-1 change: 0.0 to +0.1 (weights 0-0.25), no
      better than the 4B's +0.0 to +0.6.
    - Not tested: a learned chooser on 27B evidence (needs 27B names for the 34k training photos, ~3 h of generation).
    - Even a perfect chooser over pool + 27B evidence gains only ~2-3 pts; the gap to close is choosing (pool oracle
      64.5% vs reranker top-1 42.8% <25 km on val).

17. **Why stage 1 found so little: the search evidence is right on the same photos retrieval is already right on
    (2026-10-02; dev and val, <25 km, SigLIP2 + Wikipedia-dense results together).** A photo is locatable when it contains
    something identifying (landmark, readable text, distinctive architecture); then image retrieval finds it, the reranker
    keeps it, and an LLM can name it. When it doesn't, none of them can.
    - Evidence lands near the truth for 33-43% of photos whose pool already holds the answer, but for 3-7% of photos whose
      pool misses it (4B 2.6 / 5.6%, 27B 7.0 / 6.8% on dev / val): it recovers 12-33 of 355-470 misses.
    - Where the reranker's top-1 is right it supports that top-1 for 42-55% of photos; where the top-1 is wrong but the
      answer is in the pool ("choosing" photos, 19-22% of photos) it supports a right candidate for only 15-23% and the
      wrong top-1 for 12-19%: mostly silent exactly where help is needed.
    - The pool-miss photos are generic or unlocatable: kayaks in a marsh, a lion cub in grass, a dusk street silhouette
      (named "Toronto", truth Kuwait), a party, a table setting. The LLM's guesses are a regional prior the region head
      already holds. 16 of the 1,000 val photos (15 yfcc4k, 3% of yfcc4k) are Flickr "photo is no longer available"
      placeholders, which no method can locate.
    - Consequence: a text query is the LLM re-describing the same pixels, a smaller and less-informed version of what
      retrieval over ~10M photos already did, so it adds information only where the model knows a place that the photo
      database covers poorly (27B: +2-3 pts oracle; 4B: ~+1). The un-redundant channels left are reading fine detail
      (text, signs) and knowledge about places, which suits per-candidate checking (stage 2) more than search.

18. **Text screen (Qwen3.5-9B transcribes legible text; strings searched in SigLIP2 text space and Wikipedia; `evidence_screen.py`,
    2026-10-02): fails the screen.** Prompt: "Copy any legible text in this photo that could help locate it (signs, shop names,
    street names, banners)"; dev / val, 1,000 photos each.
    - 27% of photos have any text, the same share among pool misses (28%) and "choosing" photos (20-27%), so text isn't
      concentrated where retrieval fails. The reranker is as good on text photos as on others (top-1 34 vs 34% dev, 47 vs 41%
      val; pool holds the answer 52 vs 53% / 64 vs 65%).
    - Most strings are generic: "AF 593", "A10", "73", "AIR FRANCE", "yelp", a URL, Flickr's "photo no longer available".
      The specific ones sometimes work (a gravestone name: evidence 2 km vs pool 214 km; "Loch Dunvegan, Glasgow,
      Caledonian MacBrayne": 5 vs 11 km).
    - Evidence within 25 km of the truth: 28 / 27% when the pool holds the answer, 8.5 / 13.3% when it misses; it recovers
      11 of 470 and 13 of 355 misses = +1.1 / +1.3 pts oracle over the pool, about the 4B naming result (lesson 13). Against
      matched whole-image results: SigLIP2 +0.4 (CI -0.2, +1.0), Wikipedia -0.1 to +0.1 on val.
    - Informativeness: touches 5-11% of pools; one-parameter combiner top-1 change -0.3 to +0.3 (intervals include 0).
      "Choosing" photos with text are few (n = 44-50); there the evidence backs a right candidate 20% vs the wrong top-1
      8-14%, too few to read.
    - Caveat: searching strings is limited by the backend. Shop and street names live in OpenStreetMap, not Wikipedia
      (1.15M notable articles). The earlier readable-text win (lesson 4) was the LLM reading and reasoning about the text
      (language, country), not searching it, and that is untested here.

19. **A real name index doesn't rescue the text channel (`overture_text.py`, Overture Maps places 2026-09-23.1: 80.5M POIs,
    SQLite FTS5, 8.4 GB; Qwen3.5-9B's strings; 2026-10-02).** A candidate is "supported" if a place whose name contains all
    the tokens of one of the photo's strings lies within 3 km of it. dev / val, ~270 photos with text each:
    - A name match exists for 190 / 174 of them (70 / 64%), a specific one (<= 50 places) for 74 / 93.
    - Among photos with any support, a correct candidate is supported 73 / 71% of the time vs 49 / 39% for a wrong one
      (AUC 0.69 / 0.66; Wikipedia-based text evidence was 0.52-0.60). Supports are common for wrong candidates because
      pools hold several candidates in one city and many names are chains or generic words.
    - But the photos that matter are few: "choosing" photos (answer in the pool, top-1 wrong) with any support number
      27 / 29 of 1,000, so even a perfect use of the text could move top-1 by at most ~2.7 pts. There the support backs a
      right candidate 67 / 69% and the wrong top-1 74 / 48%: no usable signal.
    - Specific strings as global evidence: 29 / 36% within 25 km when the pool holds the answer, 14 / 11% when it misses
      (recovers 4 + 4 misses, +0.4 pts oracle). One-parameter combiner (w = 2, fitted on dev): top-1 +0.1 / +0.1
      (fixed/broke 2/1 and 4/3).
    - Crude matching (all-tokens AND, no IDF weighting, no confidence filter), so a tuned version would do somewhat
      better, but the ceiling set by the number of choosing photos with text stays. Same pattern as lessons 13-18:
      photos with identifiable text are the ones retrieval already gets right.

20. **Audit of the train / dev / val sets (`experiment/audit_sets.py`, 2026-10-02): sound, with two small flaws.**
    - Clean: no photographer overlap between train (34,523 photos), dev (1,000) and val (1,000; 656 photographers unknown to
      MP16); near-duplicates (cosine >= 0.95) of train photos: 2 in dev, 0 in val; none inside dev; truths are not on coarse
      grids (0.0-0.1% on a 0.01 deg grid, chance 0.04%); sharing the exact truth coordinate with >= 5 gallery photos (28% of
      dev) doesn't raise reranker top-1 <1 km (15% vs 16%); country mix matches (US 29-30%, UK 10-11%); a photographer-clustered
      standard error equals the per-photo one (design effect 0.99-1.03); 0 of 1,000 val photos pass the burned-in-GPS filter
      (dev is filtered by construction).
    - Flaw 1: 15 val photos (1.5%, 3% of the yfcc4k half) are Flickr's "photo is no longer available" image (none in train
      or dev); unlocatable, and they form 105 of val's 106 duplicate pairs. Without them val top-1 <1/<25/<200 km goes
      18.4 / 42.8 / 60.2 -> 18.7 / 43.5 / 61.0 and the pool oracle <25 km 64.5 -> 65.4. Flagged in
      `artifacts/query_evidence/val/exclude.json`; analyses from the attribute test on drop them (earlier val numbers include
      them).
    - Flaw 2: 3.7% of dev and 5.3% of val photos have a gallery photo with cosine >= 0.95 even after same-photographer
      exclusion (flags in `<tag>/near_duplicate.json`); the reranker's top-1 <25 km on them is 73 / 70% vs 33 / 41% for the
      rest. Without them dev top-1 <25 km is 34.2 -> 32.7 and the pool oracle 53.0 -> 52.2 (val 43.5 -> 42.0 and 65.4 ->
      64.3, also without placeholders). It lifts every method alike, so comparisons are unbiased, but absolute accuracies
      are ~1.5 pts optimistic.

21. **Attribute test: the VLM's photo attributes are accurate but redundant with what retrieval already encodes
    (`photo_attributes.py`, `candidate_attributes.py`, `attribute_ranker.py`, `attribute_check.py`; 2026-10-02).**
    Qwen3.5-9B described every photo (setting, terrain, water, vegetation, weather, text language; 91% fully valid);
    candidates got climate, elevation, ruggedness, coast distance and urbanness from WorldClim, ETOPO and Overture.
    - The attributes track the truth: photos called "sea" lie a median 0.9-1.1 km from the coast (26-34 km for "none"),
      "mountain" 152-322 m elevation std (flat 15-19), "tropical" 22.6-24.9 C (conifer 7.2-7.5), "city" 13.6-17.4k places
      within ~6 km (wild 31-45), and a named text language is official at the true location's country for 86-90% of photos.
    - Learned reranker, MP16-train refit, top-1 <25 km, 3 seeds (dev / val without placeholders): A original 34.7 / 44.0;
      F + candidate attributes 35.0 / 43.3; G + photo attributes, their products with the candidate attributes and a language
      match 34.2 / 43.7; H = G with photo attributes shuffled across photos 34.4 / 42.6. G minus A -0.5 / -0.2 (CIs include 0),
      G minus F -0.7 / +0.4; G minus H -0.1 / +1.1 (val CI +0.1, +2.2), but H falls below A, so G's edge is less harm from
      the extra features, not a gain over the baseline.
    - Why: on "choosing" photos (answer in the pool, top-1 wrong; dev + val) the wrong top-1 fits the photo's attributes as
      well as the right candidate: sea 98% vs 93% coastal, mountain 84% vs 84%, flat 63% vs 66%, city 72% vs 69%, text
      language 89% vs 85%. A rule separates the two for only 4-36% of photos and then favours the right one 0-55% of the time
      (rural 28%, wild 29%, language 17%). Both candidates came from visual-similarity retrieval, so they already agree with
      the photo on climate, terrain, water and setting.
    - Summary of lessons 11-21: search queries (4B, 27B, text, a name index of 80M places) and photo attributes all add at
      most 1-3 oracle points and nothing to a learned chooser. The information they carry is already in the photo embedding.

22. **Exemplar comparison (zero-shot Qwen3.5-9B judges "same place?" for the query photo vs one database photo taken within 1 km
    of a candidate; `exemplar_judge.py`; 2026-10-02): informative per candidate, no gain in top-1.**
    - Screen on "choosing" photos (right answer in the pool, top-1 wrong; 402 photos, dev 187 + val 215): the right candidate's
      best exemplar scores above the wrong top-1's in 58.5% [54, 63] of photos (dev 56, val 60.5); embedding similarity of the
      same exemplars 50.5%. By distance of the right candidate to the truth: within 1 km 69.4% [62, 76] (n = 144), 1-25 km 52.3%
      (n = 258). A second exemplar adds nothing.
    - Full test: top 8 candidates of every dev and val photo, one exemplar each (~16,000 comparisons, all had an exemplar).
      Candidate-level: mean P(same) 0.74 / 0.78 (dev / val) for candidates < 1 km from the truth, 0.57 / 0.65 for 1-25 km,
      0.38 / 0.44 for >= 25 km (AUC 0.79 for < 1 km vs >= 25 km, 0.66-0.67 for 1-25 km vs >= 25 km).
    - But within a photo the judge's highest-scoring candidate is within 1 km in only 35 / 33% of photos that have such a
      candidate among the top 8, vs 62 / 56% for the reranker's top-1. Combined as -rank + w * logit(P): best w = 0.25 gives
      top-1 <25 km +0.1 [-0.2, +0.4] dev and +0.2 [-0.3, +0.8] val; larger w hurts (w = 1: flips to a right answer 13 vs away
      19 on dev, 25 vs 25 on val; w = 8: -0.6 / -1.7). A cross-validated learned combiner (rank + judge features, 5 folds,
      3 seeds, 1,985 photos): -0.1 [-0.5, +0.2]. Shuffled-judge control: 0.0 / -0.3.
    - Reading: the judge separates "the true place" from "far away" but not the exact place from its neighbours among
      plausible candidates, and its noise outweighs the reranker's ordering. Untested: a judge fine-tuned on pairs with
      ground-truth labels (exemplar within 1 km of the truth vs hard negatives from the same pool).

23. **The zero-shot judge's skill is mostly exemplar similarity (2026-10-02, `exemplar_judge.py`, zero-shot 9B, top-8 pairs).** The
    judge's candidate-level AUC (candidates < 1 km from the truth vs >= 25 km) by tercile of the exemplar's embedding similarity to the
    query: low 0.50 / 0.54 (dev / val), mid 0.72 / 0.71, high 0.85 / 0.86. 90% of the positives sit in the mid and high terciles. A
    cross-validated learned combiner (dev + val, 5 folds, 3 seeds) over rank + judge + similarity gives +0.4 [-0.2, +1.0] at <25 km, and
    rank + similarity without the judge gives the same +0.4 [-0.3, +1.0]: the zero-shot judge adds nothing beyond similarity.

24. **A fine-tuned comparator beats the zero-shot judge and gives the first small positive result (`comparator_data.py`,
    `comparator_train.py`, run `comparator-a`; 2026-10-03).** Qwen3.5-4B + LoRA (r 32, all language linears, 448 px images), input
    = query photo + one exemplar (database photo within 1 km of a pool candidate, best by embedding similarity, not by the query's
    photographer), output = Yes/No at the generation position, balanced binary loss, 48,000 pairs (15,947 positive: candidates < 1 km
    from the truth; negatives >= 10 km; the 1-10 km band dropped) from the 34,523 MP16 train photos' top-8 candidates, 1 epoch,
    1.5 h on the shared GPU at 8.7 samples/s (13 GiB). Held-out train-photo pairs: accuracy 77.8%, AUC 0.825.
    - Candidate-level AUC (< 1 km vs >= 25 km from the truth, dev / val): 0.899 / 0.866 (zero-shot 9B 0.787 / 0.789); 1-25 km vs
      >= 25 km 0.789 / 0.778 (0.655 / 0.674).
    - Within a photo the judge's top pick is within 1 km in 48 / 40% of photos that have such a candidate in the top 8 (zero-shot
      35 / 33%), still below the reranker's top-1 (62 / 56%).
    - Top-1 <25 km with score = -rank + w * logit(P(same)), w = 1 fitted on dev: dev +0.8 [-0.1, +1.7], val +0.7 [-0.1, +1.6]
      (im2gps3k +1.4 [+0.2, +2.8], yfcc4k +0.0); flips to a right answer vs away at w = 1: 15 vs 7 (dev), 13 vs 6 (val); shuffled-
      judge control -0.1 / +0.2. Cross-validated learned combiner (5 folds, 3 seeds, 1,985 photos): +0.6 [+0.1, +1.1].
    - Reading: real but small. The exemplar is the most similar photo near the candidate, so the comparator largely re-derives
      what the reranker's similarity features already encode; the judge's own top pick is still worse than the reranker's.

25. **More judge variants don't beat the pointwise comparator (2026-10-03).**
    - Hard cases (reranker top-1 >= 25 km off and a candidate < 1 km from the truth among ranks 2-8; n = 77 of 1,985 photos): the judge
      scores the right candidate above the top-1 in 74% [64, 83] of them for `comparator-a` (dev 82, val 67) vs 67.5% for the zero-shot
      9B. Only ~4% of photos are such cases, which caps what an exact-place judge can add at the strict level.
    - Pairwise comparator (`pairwise-a`, `--mode pairwise`: query + exemplars of two candidates, answer 2 or 3; warm start from
      `comparator-a`; 21,262 rows, 7,135 photos, the reranker's two best-ranked wrong candidates always paired with the right one;
      1 h; held-out hard pairs accuracy 71.4%, AUC 0.802): over the top-4 candidates with Borda scores and -rank + w * logit, top-1
      <25 km +0.9 [-0.4, +2.2] dev, -0.1 [-1.6, +1.5] val, and <1 km -1.5 / -2.3. No better than the pointwise comparator; dropped.
    - Cross-validated learned combiner over dev + val (1,985 photos): `comparator-a` alone +0.7 [+0.2, +1.2] (<1 km +0.3, <200 km
      +0.5); + exemplar similarity +0.3 [-0.3, +0.9]; + similarity + the zero-shot judge +0.4 [-0.2, +1.0]. Extra features only add noise.

26. **A comparator trained for the 25 km metric is no better (run `comparator-25km`, 2026-10-03).** Same trainer, warm start from
    `comparator-a`, pointwise labels positive < 25 km from the truth and negative >= 50 km (`pairs_train_all.jsonl`, 40,000 rows, 31,001
    positive, 1.3 h at 8.8 samples/s; held-out pairs accuracy 67.8%, AUC 0.807). Candidate-level AUC rose where intended (1-25 km vs
    >= 25 km: 0.844 dev / 0.815 val vs 0.789 / 0.778 for `comparator-a`; < 1 km vs >= 25 km fell to 0.855 / 0.822), but the top-1 gain did
    not: scalar rule (w = 2 on dev) +0.7 [-0.5, +1.8] dev, -0.1 [-1.4, +1.2] val; cross-validated combiner +0.5 [+0.1, +0.9] (all 1,985
    photos). Both comparators together in the combiner: +0.5 [-0.0, +1.0]. The exemplar-comparison signal saturates around +0.5 to +0.7
    pts of top-1 <25 km at this scale (48k pairs, 1 epoch).

27. **Model size alone doesn't make a zero-shot chooser beat the reranker (`knowledge_scaling.py`, 2026-10-03).** The `default` prompt
    (photo + the reranker's top-10 candidates as text, think briefly, JSON coordinates), temperature 0, dev / val (placeholders dropped),
    top-1 <25 km: reranker 34.2 / 43.5, candidate oracle 47.2 / 59.8; Qwen3.5-4B 29.1 / 38.5 (-5.1 / -5.0, CIs exclude 0), Qwen3.5-9B
    29.1 / 38.8 (-5.1 / -4.7), Qwen3.6-27B (Q4_K_M, llama.cpp) 32.4 / 40.7 (-1.8 [-3.5, -0.2] / -2.7 [-4.8, -0.7]). At <200 km: reranker
    48.8 / 61.0, 4B 43.9 / 57.0, 9B 41.0 / 56.1, 27B 46.6 / 59.3. 74-93% of answers lie within 25 km of a shown candidate (4B 88 / 90,
    9B 74 / 83, 27B 90 / 93). The 9B is no better than the 4B; the 27B is closest but still below the reranker.

28. **Nearby Wikipedia text per candidate helps the zero-shot chooser a little, not enough (`wiki_nearby.py`, `knowledge_scaling.py --wiki`,
    2026-10-03).** Each shown candidate line gets its nearest geotagged Wikipedia articles within 3 km (up to 3: titles, the first with its first
    sentence cut at 90 characters; 91% of candidates have at least one). Same prompt and models as lesson 27, top-1 <25 km (dev / val):
    4B 29.9 / 38.3 (vs 29.1 / 38.5 without), 9B 30.4 / 39.9 (vs 29.1 / 38.8), 27B 31.6 / 40.6 (vs 32.4 / 40.7); against the reranker
    34.2 / 43.5 they are -4.3 / -5.2, -3.8 / -3.6 and -2.6 / -2.8 (all CIs exclude 0). <200 km: 9B 42.9 / 58.7 (vs 41.0 / 56.1). The 9B gains
    about +1.2 pts, the 4B +0.3 on average, the 27B nothing. Knowledge about what lies at each candidate is not what separates the small
    models from the reranker.

29. **The comparator's gain holds on all benchmark eval-half photos (2026-10-03; `exemplar_judge full-report`, run `comparator-a`).** Top-8
    exemplar pairs for all 3,795 im2gps3k + yfcc4k eval-half photos (3,713 after dropping 82 Flickr placeholders; every candidate had an
    exemplar), scored by the merged `comparator-a`, combiner weight w = 1 fitted on the MP16 dev photos only (nothing on the benchmarks was
    used to fit): top-1 <1 / <25 / <200 km, reranker 17.1 / 38.9 / 56.4 -> 17.3 / 39.7 / 56.9, change +0.2 / +0.7 [+0.3, +1.2] / +0.5;
    shuffled-judge control -0.2. im2gps3k (n = 1,480) 47.8 -> 49.3 (+1.4 [+0.7, +2.2]); yfcc4k (n = 2,233) 33.0 -> 33.3 (+0.3 [-0.3, +0.8]).
    Not a near-duplicate effect: without the 202 photos whose top gallery neighbour has cosine >= 0.95 (n = 3,511) +0.7 [+0.3, +1.2]; on
    those 202 +1.0 [-1.0, +3.0]. Larger weights do slightly better (w = 2 / 4 / 8: +0.9 / +1.0 / +0.9 on all photos; fitted on dev, w = 1 is
    reported). The scoring of 30k pairs takes ~8 minutes on the shared GPU.

30. **The gallery isn't thin, and going deeper than the pool doesn't help either (2026-10-03; `deep_rerank.py`).**
    - Coverage: 79 / 78% of dev / val photos have more than 1,000 MP16 + OSV photos within 25 km of the truth; only 1 / 4% of the photos the pool
      misses are in places with fewer than 10. Misses are retrieval failures, not missing data.
    - A right-place photo (< 25 km) is among the top K raw neighbours (MP16 + OSV merged by similarity) for K = 100 / 300 / 1000: 60 / 69 / 80% of
      dev photos and 71 / 80 / 88% of val, vs 53 / 65% for the ~17-candidate pool. But that recall is deep and spread out: clustering the top-1000
      neighbours at 1 km and keeping the first K clusters gives a right-place cluster for K = 10 / 25 / 50 in only 38 / 47 / 55% (dev) and
      51 / 61 / 68% (val), i.e. +2-3 pts over the pool at 50 clusters.
    - Re-ranking those 50 clusters (each represented by its best photo, ~99k comparisons with `comparator-a`), top-1 <25 km over dev + val (1,985
      photos): original reranker 38.8; raw similarity's best cluster 26.9; comparator alone 27.1; cross-validated learned combiners: similarity +
      comparator 29.3, similarity + reranker rank 36.9, all three 38.0 (-0.8 [-2.0, +0.4]). The reranker's pool (Pinpoint's GPS gallery and the
      region prior) is far better than raw similarity, and over many look-alike candidates the comparator's false positives outweigh its hits.
    - So the deep recall (80-88% at 1,000 neighbours) sits where precision is lowest. Pulling it in needs a better ranking of the raw neighbours
      themselves (a geo-aware embedding) or a better region prior, not a judge over more candidates.
31. **A region-diversified pool adds ~2 oracle pts at matched budget; the region classifier is the bottleneck (2026-10-03; `region_prior.py`).**
    - Same cached top-1000 raw neighbours (CPU only). "Regional" pools cluster inside each of the region head's top-10 (state, country)
      regions and interleave them by P(region) / (rank in region + 1) ** beta; the control extends the pipeline's own raw source
      (similarity + 0.01 * log prior). Dev (1,000) / val (985), <25 km.
    - Alone at the pool's size (17), no pool beats the current one (dev 52.3 / val 64.4%; best regional 52.3 / 63.7%).
    - Current pool + K extra candidates, oracle gain: regional (head, beta = 2) +4.0 / +6.3 / +9.1 (dev, K = 5 / 10 / 20) and +3.6 / +5.9 / +8.1
      (val), vs the control's +2.2 / +4.5 / +8.0 and +2.0 / +3.9 / +6.0. So ~+1.5-2 pts beyond the matched control: real but below the 3-pt bar, and
      past lessons say extra candidates rarely turn into top-1. Mixing in Pinpoint's GPS-kNN region votes ranks the true region higher
      (dev top-1 40 -> 47%) but doesn't improve the pools.
    - Ceiling: clustering only inside the *true* region gives 77.6 / 85.1% at 17 candidates (vs 52.3 / 64.4) and +27 / +22 pts as 10 extras. The
      head ranks the true region 1st for 40 / 53% of photos and in its top 10 for only 72 / 82%. The headroom is in region classification itself,
      and earlier attempts to choose the region better (VLM, retrieval evidence; lesson 5) failed.
32. **A bigger region head, or more data for it, barely moves region accuracy (2026-10-03; `region_head_big.py`).**
    - Heads on frozen SigLIP2, selected on 2,000 held-out MP16 photos (not dev), true region top-1 / 10. Existing setup (1 hidden layer 2048, 2 epochs,
      1.5M photos: all MP16-query photographers held out): 41.5 / 73.4%. Two layers of 4096 trained for 10-20 epochs overfit (39.0 / 71.3%). Holding out only
      benchmark + dev + selection photographers (3.56M photos): 42.7 / 75.3%; two layers of 4096 for 3 epochs: 43.0 / 75.3%; plus 2M OSV-5M street photos: 43.2 / 75.2%.
    - With the best head (wide, 3.56M), dev true region top-1 goes 40 -> 44% and val 53 -> 55% (val's pipeline head already had the bigger data). The
      pool test barely moves: regional pool at 17 is 51.6 / 64.8% (current 52.3 / 64.4%); as 5 extras +4.1 / +4.2 vs the control's +2.8 / +2.4.
    - So region classification from frozen SigLIP2 is saturated at ~43% top-1 on MP16. The perfect-region ceiling (lesson 31) is out of reach for this
      family of classifiers. Getting there would need information the embedding lacks: text, signs, live evidence.
33. **Stage-2 go/no-go: the models' own answers add nothing beyond reranker + comparator (2026-10-03; `model_vote_cv.py`).**
    - Setup: a cross-validated combiner (5 folds, 3 seeds) over each dev + val photo's top-8 candidates (1,985 photos; reranker top-1 <25 km 38.8%,
      top-8 oracle 51.4%). Features: rank, `comparator-a` scores, and a "vote" from each zero-shot chooser's answer (lesson 27): distance to each
      candidate, whether it is within 25 km, whether it is the nearest.
    - Rank + vote: 4B +0.0, 9B +0.0, 27B -0.0. Rank + comparator: +0.6 [+0.1, +1.1]. Adding any vote to rank + comparator: -0.2 / +0.1 / +0.1
      (CIs about ±0.7). The choosers mostly repeat the reranker (lesson 1), so their answers carry no independent signal.
    - Implication: a stage-2 policy that sees only the photo, the candidates and the comparator has about +0.7 pts of learnable headroom over
      the reranker. A combiner that sees every comparator score is an upper bound for a budgeted agent using the same tool. RL with these tools
      would be a framework, not a gain; it needs a new information source (stronger comparator, live street-level photos) to have something to learn.
34. **Kinds of places near a candidate don't discriminate either (2026-10-03; `category_evidence.py`).** Map side: Overture places of 61 visual
    categories (church, castle, stadium, beach, lighthouse, marina...; 8.1M places) counted within 1 / 5 km of every pool candidate. Photo side:
    SigLIP2 zero-shot over the same categories (its sigmoid P is ~0 everywhere; the ranking is sensible, so a per-photo softmax). Match = sum of
    P(visible) * idf * [category nearby]. Top-8 candidates of 1,985 dev + val photos.
    - On 250 "choosing" photos (right candidate in the top 8, top-1 wrong): the right candidate's match beats the wrong top-1's 56% of the time,
      ties 0-3%, loses 41-44%. The photo's most likely category lies near the right candidate in 115 / 168 of them (1 / 5 km) and near the wrong top-1
      in 115 / 161. Within-photo AUC right vs wrong is 0.54 / 0.57.
    - Cross-validated combiner, top-1 <25 km vs the reranker (38.8%): rank + categories -0.3; rank + comparator +0.6; rank + comparator + categories
      +0.1; with shuffled categories +0.2. Categories beyond rank + comparator: -0.5 [-1.2, +0.1].
    - Same mechanism as lesson 21: the wrong candidates come from photos that look like the query, so the same kinds of places are near them. Overture
      is sparse for physical features (lighthouses 3k, castles 15k), so OpenStreetMap would be denser. But the right/wrong split is even (115 vs 115),
      which says density isn't the limiting factor. Not worth the OSM download.
35. **By eye, most of the choosing gap can't be recovered from the photo (2026-10-03; `choosing_sheet.py`, 30 of the 250 choosing photos, 15 dev + 15 val,
    each with the query, the right candidate's exemplar and the wrong top-1's exemplar).** Tags:
    - Near-miss at the 25 km line, 9 / 30: the top-1 is 26-35 km away in the same metro area, park or département, and the right candidate is no more
      convincing. Across all 250 choosing photos, the wrong top-1 is within 25-50 km of the truth for 21% (50-200 km 29%, >= 200 km 50%).
    - Nothing place-specific, 10 / 30: insects, food, fireworks, sky, bowling alleys, gardens, generic coast. A person can't choose either.
    - A specific place is visible, 6 / 30: a sculpture in Milan, Oasis 21 in Nagoya, a Liverpool plaza, a readable café name, a distinctive building, an
      arena. The comparator already scores the right exemplar high in 3 (0.81-0.98). Elsewhere the exemplar shows a different spot (multi-exemplar
      would help), the text needs reading, or the cue misleads (visiting team's jerseys).
    - Needs knowledge, 4 / 30: Thai lacquer art, Gambel oak leaves, granite summits, a Patagonian skyline.
    - Wrong ground truth, 1 / 30: a Disneyland Paris photo geotagged in central Paris.
    - Implication: about 20% of the 12.6-pt gap between top-1 and the top-8 oracle (<25 km) is recoverable from the photo, i.e. ~2.5-3 pts, part of
      which the comparator already takes. That explains why every chooser stalls near +0.7.
36. **Coarse thresholds have as much headroom, and our evidence doesn't help there either (2026-10-03; `threshold_headroom.py`).** 1,985 dev + val
    photos; cross-validated combiners over the top 8 retrained with the reward "within T km".
    - Headroom, as reranker top-1 / top-8 oracle / pool oracle: 25 km 38.8 / 51.4 / 59.1; 200 km 54.9 / 67.9 / 78.0; 750 km 72.8 / 84.0 / 90.9;
      2500 km 85.3 / 92.5 / 97.1. The gap to the pool oracle is 20.4 / 23.2 / 18.1 / 11.8 pts, so region and country have as much room as city.
    - Change vs the reranker (95% CI, ~±0.6-1.0):
      - comparator: 25 km +0.6, 200 km +0.3, 750 km -0.1, 2500 km -0.2;
      - 27B vote: -0.0 / -0.5 / +0.2 / -0.5 (4B and 9B similar);
      - categories: 0.0 / -0.4 / -0.2 / -0.3;
      - everything together: +0.8 / +0.3 / +0.2 / +0.2.
    - Retargeting the ranking alone (rank features with the coarse reward) never changes the pick. The reranker's order is already its best guess
      at every scale.
    - So region- and country-level choosing needs knowledge the reranker lacks, and the zero-shot answers don't supply it: they are anchored on the
      candidate list (lesson 27: 88-93% land near a shown candidate). Untested: a candidate-free answer as an independent coarse vote.
37. **The 4B's own geographic knowledge, without candidates, is far weaker than retrieval at every scale and adds nothing (2026-10-03;
    `free_guess.py`, `scripts/free_guess.sh`).** Base Qwen3.5-4B, photo only ("Where was this photo taken? ... name the country and region"),
    greedy + 8 samples at T = 0.7, 1,985 dev + val photos, ~5 min per 1,000 photos.
    - Greedy guess alone vs the reranker top-1, at 25 / 200 / 750 / 2500 km: 19.4 vs 38.8, 31.8 vs 54.9, 49.1 vs 72.8, 65.9 vs 85.3. It is right
      where the reranker is wrong for only 2.6-4.2% of photos, and wrong where the reranker is right for 22-28%.
    - As a vote (distance to the greedy guess, share of samples within 25 / 200 / 750 km of each candidate) in the per-threshold combiners: rank +
      vote +0.1 / -0.6 / +0.2 / -0.1; rank + comparator + vote +0.6 / +0.1 / +0.1 / -0.6 (vs rank + comparator +0.6 / +0.3 / -0.1 / -0.2).
    - So the anchored answers (lesson 33) weren't hiding independent knowledge. At the 4B scale, the VLM knows less about where a photo is than
      whole-image retrieval over 9M geotagged photos, even at country level.
38. **A geo-aware adapter on SigLIP2 adds about +1 pt of candidate recall over the region prior (2026-10-03; `geo_embed.py`).** Residual MLP on frozen
    SigLIP2 (identity start), multi-positive InfoNCE over each MP16 bucket-99 train photo's cached neighbours. Positives are neighbours < 25 km from the
    truth; negatives are the top-96 look-alikes plus 128 random deeper neighbours. 29,933 train photos have a positive in their top 1000 (2.9M gallery
    rows); dev and benchmark photographers' rows are blocked; selection uses held-out train photographers. Training takes minutes.
    - Look-alike negatives alone are a trap: selection MRR rose 0.32 -> 0.45, but on the full lists the adapter ranked right places *worse* than raw
      similarity (val, right-place cluster in the first 10: 42.8 vs 51.4%). It lifted deep neighbours it had never seen. Adding random deep negatives
      fixed it (selection MRR 0.32 -> 0.41; 4096 hidden, lr 2e-3, best around epoch 5).
    - Right-place cluster (< 25 km) in the first 10 / 50 clusters:

      | Ranking | dev @10 | dev @50 | val @10 | val @50 |
      |---|---|---|---|---|
      | raw | 37.7 | 54.5 | 51.4 | 67.7 |
      | raw + region prior | 43.0 | 59.3 | 57.0 | 70.5 |
      | adapter | 40.9 | 57.6 | 52.7 | 70.3 |
      | raw + adapter + prior | 44.5 | 60.0 | 57.6 | 71.7 |

      The 200 km numbers move the same way.
    - Current pool + 5 / 10 / 20 extra clusters, oracle gain < 25 km: adapter +3.0 / +4.9 / +7.9 (dev) and +3.0 / +5.1 / +7.9 (val), vs the matched
      control (raw + prior) +2.2 / +4.5 / +8.0 and +2.0 / +3.9 / +6.0. That is ~+1 pt at small K, about what the region-diversified pool gave (lesson 31).
    - So geography-aware re-scoring of the same neighbours is real but small, like everything else on this pool. Training data is the binding limit:
      the rules allow only bucket 99 (~30k usable queries).
    - Pinpoint's own retriever is the same idea at scale (image tower vs location encoder on frozen SigLIP2, MP16 buckets 0-98 + OSV-5M, ~200M
      samples). Same metric over its cached top-500 GPS-gallery rows (~40 clusters), right place in the first 10 clusters < 25 km | < 200 km:
      dev 41.2 | 58.4, val 48.4 | 68.8, below raw + prior. Its strength is the top-1 (32.2 / 37.5% < 25 km); its list bunches around one guess. The
      pool already merges it with raw + prior, which is why a smaller copy trained on bucket 99 adds little.
39. **Several exemplars per candidate lift the comparator a little, without retraining (2026-10-03; `multi_exemplar.py`, `scripts/multi_exemplar.sh`).**
    `comparator-a` as is; up to 4 exemplars per top-8 candidate (within 1 km, one per photographer, not the query's; mean 3.25; exemplar 0 is lesson
    24's), 52k comparisons in 13 min. Top-1 < 25 km vs the reranker, -rank + w * aggregate(logit), w fitted on dev:
    - first exemplar (lesson 24): dev +0.9 [+0.0, +1.8], val +0.5 [-0.3, +1.4] (im2gps3k +1.0, yfcc4k +0.0);
    - max over exemplars (w 2): dev +1.2 [+0.2, +2.2], val +1.1 [-0.1, +2.3] (im2gps3k +2.2 [+0.4, +4.0], yfcc4k +0.0);
    - mean of best 2 (w 4): dev +1.5, val +0.9; plain mean (w 8): dev +1.1, val +0.3.
    - Cross-validated combiner over dev + val: first exemplar +0.7 [+0.2, +1.2], all exemplars +0.9 [+0.2, +1.7].
    - So extra exemplars add about +0.3 to +0.6 pts over one, consistently on dev and val, all of it on im2gps3k (landmark-heavy). yfcc4k stays at 0.
      The comparator was trained on one exemplar per candidate; training it on several, and on more pairs, is the untested part.

40. **Training the comparator on 3x the pairs with several exemplars each gains ~+0.3 pts, below the +2 go bar (2026-10-04; run `comparator-b`,
    `comparator_data.py multi`, `comparator_train.py --mode multi`, `scripts/comparator_b.sh`).** Continued from `comparator-a`'s adapter on 141,646
    rows (40,568 positive) from 33,052 bucket-99 train photos: every exemplar (up to 4, one per photographer) of every positive and of the reranker's
    two best-ranked wrong candidates, one exemplar of up to two other wrong candidates; batch 16, lr 1e-4, one pass, 4.5 h at 8.7 pairs/s. Held-out
    train-photo pairs: accuracy 77.5%, AUC 0.816 (`comparator-a`: 77.8% / 0.825, different pairs).
    - Scoring trap: the loss fixes only logit(Yes) - logit(No), so the absolute Yes / No level drifted. For ~12% of photos `comparator-b` put almost
      no mass on either (top token a digit at ~0.5%), and the top-12-logprobs extraction returned nothing for 13% of pairs (7,002 of 52,092). Fixed by
      restricting the output to the two answer tokens (`allowed_token_ids` + `--logprobs-mode processed_logprobs`); all pairs now score. Always check
      the failed count of a judge run.
    - Same scoring, 4 exemplars per top-8 candidate, top-1 < 25 km vs the reranker (dev / val; w fitted on dev):

      | scorer | `comparator-a` | `comparator-b` |
      |---|---|---|
      | first exemplar | +0.9 / +0.6 | +1.6 / -0.6 |
      | max over exemplars | +1.2 / +0.9 | +1.4 / +1.8 [+0.3, +3.6] |
      | mean of best 2 | +1.5 / +0.7 | +1.2 / +1.3 |
      | cross-validated combiner, all exemplars (all 1,985) | +0.8 [+0.0, +1.6] | +1.1 [+0.4, +1.8] |

      im2gps3k (max): a +2.0, b +2.0; yfcc4k: a -0.2, b +1.6 [-0.8, +4.1].
    - So b is at best ~+0.3 over a on the combiner (+1.1 vs +0.8, CIs overlap) and ~+0.9 on the val rule (+1.8 vs +0.9); the val interval still
      contains +0.9 and misses the +2 bar. On its own, b's first-exemplar score is no better than a's. Training mix and data volume change little:
      the exemplar-comparison signal saturates near +1 pt of top-1 < 25 km at the 4B scale, consistent with the ~2.5-3 pt cap from lesson 35.
    - Scores with the broken extraction are kept as `multi_exemplar_scores_comparator-b_top12.json` for reference; do not use.

41. **Full benchmark test of `comparator-b` with 4 exemplars: +0.9 [+0.3, +1.6] from a combiner trained on dev; the scalar rule that dev picks is null
    (2026-10-04; `multi_exemplar.py full-pairs / full-judge / full-report`, `scripts/multi_exemplar_full.sh`).** All 3,713 im2gps3k + yfcc4k eval-half photos
    (82 placeholders dropped), 95,490 comparisons (3.21 exemplars per candidate, 0 failed, 24 min), nothing fitted on them: aggregate and weight chosen on
    the 1,000 MP16 dev photos, the listwise combiner trained on dev only. Reranker top-1 < 1 / 25 / 200 km: 17.1 / 38.9 / 56.4. Change at < 25 km [95% CI]:

    | scorer | all | im2gps3k | yfcc4k |
    |---|---|---|---|
    | rule: first exemplar (w 4) **chosen on dev** | +0.4 [-0.5, +1.2] | +0.7 | +0.1 |
    | rule: max over exemplars (w 4) | +1.7 [+0.9, +2.5] | +2.6 [+1.5, +3.8] | +1.1 [+0.1, +2.1] |
    | rule: mean of best 2 (w 4) | +1.4 [+0.7, +2.2] | +2.3 | +0.9 |
    | rule: mean (w 4) | +0.8 [+0.1, +1.5] | +1.5 | +0.4 |
    | combiner trained on dev, first exemplar | +0.9 [+0.4, +1.4] | +1.1 | +0.7 |
    | combiner trained on dev, all exemplars | +0.9 [+0.3, +1.6] | +1.2 | +0.7 [-0.2, +1.6] |

    - Honest reading: by the pre-declared protocol (best dev scorer) the result is +0.4, not significant. The best row, max over exemplars (+1.7), was the
      best of four scorers on val earlier, not chosen on dev (dev: first +1.6, max +1.4), so treat it as a hypothesis for a fresh test, not as the headline.
      The trained combiner is the clean number: +0.9 [+0.3, +1.6], vs +0.7 [+0.3, +1.2] for `comparator-a` with one exemplar (lesson 29).
    - Without the 202 near-duplicate photos the numbers are the same (max +1.7 [+0.9, +2.6]). < 1 km does not move (+0.1 to +0.7 with exemplars, -1.6 for the
      dev-chosen rule); < 200 km +0.5 for max and the combiners.
    - So the comparator line ends around +1 point of top-1 < 25 km on the benchmarks, whatever the exemplar count or the training volume.

42. **Wikimedia (the third final-test set) is harder, has the same choosing headroom, and the comparator gives nothing there (2026-10-04;
    `wikimedia_eval.py`, `multi_exemplar.py full-pairs / full-judge / full-report --tag wikimedia_balanced`).** Loader: `benchmarks.py` entry `wikimedia`
    (`/data/pinpoint/wikimedia/test.csv`, 6,017 photos, all flagged usable by a Gemma labeller, not Gemini; `test_balanced.csv` is a 3,036-photo density-
    balanced subset), `load_world(benchmarks=("wikimedia",))`, pool and reranker order built as for MP16 photos (reranker fitted on the im2gps3k / yfcc4k tune
    halves; nothing trained on wikimedia). No photographer overlap with MP16 (0 matches), 1.5% near-duplicates of a gallery photo (cosine >= 0.95).
    - Baseline, % within 1 / 25 / 200 / 750 / 2500 km (all 6,017): Pinpoint GPS top-1 4.3 / 22.1 / 57.9 / 82.4 / 92.4; reranker top-1 8.0 / 25.0 / 59.4 / 83.0 /
      93.1; oracle over its top 8 15.2 / 38.9 / 72.6 / 89.4 / 96.0; oracle over the whole pool (18 candidates) 18.2 / 49.4 / 83.3 / 93.8 / 97.9. Balanced
      subset: reranker 7.2 / 20.9 / 49.2 / 75.8 / 88.7, top-8 oracle 13.7 / 33.7 / 63.9 / 84.2 / 93.1, pool oracle 16.7 / 43.1 / 75.4 / 90.2 / 96.2.
      Top-1 at 25 km is 14 pts below the Flickr benchmarks (38.9), but the gap to the top-8 oracle is the same size (13.9 pts at 25 km, 7.2 at 1 km).
    - `comparator-b`, 4 exemplars (2.8 per candidate, 0.5% of candidates have none), balanced subset, everything chosen on MP16 dev as in lesson 41, change in
      top-1 < 25 km: dev-chosen rule +0.7 [-0.1, +1.5]; best-of-4 rule +0.7; combiner trained on dev +0.8 [+0.3, +1.3] with the first exemplar, +0.4 [-0.4, +1.1]
      with all. < 1 km -0.3 to +0.6, < 200 km 0.0 to +0.6. Without the 62 near-duplicate photos the gains are +0.1 to +0.5. On the 62 near-duplicates the
      gain is +14.5 [+4.8, +24.2] (the exemplar is near-identical to the query), which says the comparator matches known photos well and nothing more.
    - So the landmark-heavy set did not turn the comparator into a larger gain: there is no wikimedia-specific lever in the comparator. The +2 go bar was
      not met; the scoring of the other 2,981 photos was not run.
    - Ops note: the shared GPU was taken by another project's back-to-back jobs for ~5 h; the script now retries the vLLM start (a job that grabs memory while
      our server loads kills it) instead of failing.

43. **Near-miss photos (top-1 within 25 km, not within 1 km) are the largest target, and feature-only local refinement gets almost none of it
    (2026-10-04; `near_miss.py ceiling / rules / local-rerank`).** On the 3,713 benchmark photos the reranker top-1 is within 1 km for 17.1% and within
    25 km for 38.9%, so 21.9% (812 photos) are near-misses; another 61.1% are beyond 25 km.
    - Ceiling: for 93.3% of near-misses the gallery has a photo within 1 km of the truth, and for 73.3% one is already among the cached top neighbours
      (raw MP16 + OSV, 2,000); for 47.4% among Pinpoint's top-500 GPS rows. Over all photos 56% have one in the cached neighbours.
    - Model-free local rules (replace the top-1 by the most similar neighbour within 25 km of it, or by the mode of a similarity-weighted kernel density over
      those neighbours, settings picked on dev): < 1 km 17.7% vs 17.1% at best (T 0.03, h 1 km); fixes 100 of 812 near-misses, breaks 78 of 634 exact ones.
    - Trained local re-ranker (candidates = the top-1 plus 1 km clusters of the neighbours within 25 km, mean 14 per photo; 12 features: neighbour counts and
      similarity-weighted support within 0.5 / 1 / 3 km, best-similarity gap, rank of the first neighbour within 1 km, distance to the top-1, OSV share, GPS-row
      support; listwise softmax, reward 1 for < 1 km and 0.25 for < 5 km; 34,530 bucket-99 train photos; 3 seeds): < 1 km on the benchmarks 17.1% -> 17.8%,
      +0.8 [+0.3, +1.2] (fixes 52 of 812 near-misses, breaks 24 of 634 exact); dev +0.1 [-0.7, +0.9]. The oracle over its candidates is 30.1% (benchmarks) /
      26.2% (dev): ~13 pts of headroom at 1 km, of which features take ~6%.
    - So the 1 km headroom (13 pts) is as large as the 25 km choosing headroom, and similarity / density features can't separate the exact spot from its
      neighbours in the same city. Whether a visual judge trained for the 1-25 km band can is the open question (the comparators so far saw negatives >= 10 km).
    - Visual judge, not trained for the band (`comparator-b` as is, one exemplar per local candidate: a gallery photo within 1 km of the candidate, most similar to
      the query, not by the query's photographer; 13,562 of 14,422 dev candidates had one, 14 min; `near_miss.py exemplar-pairs / exemplar-report`, dev only):
      within-photo AUC for a candidate < 1 km vs 1-25 km from the truth 0.735 (5,989 pairs). Its own top local candidate is worse than the top-1 (< 1 km 10.1% vs
      15.9%; fixes 16 of 183 near-misses, breaks 75 of 159 exact). Cross-validated combiner (5 folds, 3 seeds, only 1,000 photos to train on): local features
      alone -0.0 [-0.8, +0.8], + comparator +0.3 [-0.6, +1.1]. So there is a visual signal at this scale before any training for it; its size is unknown.
    - Trap: indexing an NpzFile (`saved["coords"][i]`) inside a loop re-reads the array every time; with 34,530 iterations it used 47 GB of the machine's
      60 GB before I stopped it by PID. Load each array once.

44. **A comparator trained for the near band (`comparator-c`) raises the visual signal a lot but moves 1 km accuracy by < 1 point (2026-10-05;
    `near_miss.py near-pairs / final-report`, `scripts/comparator_c.sh`).** Continued from `comparator-b` on 54,712 rows (20,092 positive; 7,760 bucket-99 train
    photos that have a local candidate within 1 km of the truth; positives = candidates < 1 km from the truth with up to 2 exemplars, negatives = 4 at 1-5 km,
    1 at 5-25 km, 1 at >= 25 km and the top-1 when wrong, 1 exemplar each; batch 16, lr 1e-4, one pass, 1.7 h at 8.9 pairs/s). 4,317 train photos (k % 8 == 1)
    were kept out of its training and fit the combiner. Held-out train-photo pairs: accuracy 71.0%, AUC 0.760 (a harder mix than lesson 40's).
    - Within-photo AUC (a candidate < 1 km from the truth vs one 1-25 km away): `comparator-b` 0.735 on dev -> `comparator-c` 0.795 dev / 0.782 benchmarks /
      0.806 held-out train photos.
    - Listwise combiner (12 local features, with / without the comparator's logit, logit gap and has-exemplar flag; trained on the 4,317 held-out train photos,
      3 seeds; one exemplar per candidate; scored once on dev and the 3,713 benchmark photos), change in top-1 < 1 km vs the reranker top-1:

      | features | dev (n = 1,000) | benchmarks (n = 3,713) | near-misses fixed / exact broken (benchmarks) |
      |---|---|---|---|
      | local features | +0.4 [-0.3, +1.1] | +0.5 [+0.1, +1.0] | 45 of 812 / 25 of 634 |
      | + comparator-c | +0.4 [-0.5, +1.2] | +0.8 [+0.2, +1.3] (17.1% -> 17.9%) | 70 of 812 / 40 of 634 |

      < 25 km moves by -0.2 to +0.1. The go bar (+2 at 1 km on the benchmarks) was not met.
    - Reading: the visual judge adds ~+0.3 over the features alone, and more fixes come with more breaks. The incumbent top-1 is already right for 61% of
      what the candidates can reach (15.9% of 26.2% on dev), so a flip needs a very reliable margin; an AUC of 0.8 within a photo is not that.
    - 58,017 held-out train + 13,562 dev + 51,686 benchmark candidate scorings took 31 min in one server session.

45. **4x more near-band data (`comparator-d`) does not move the near-band signal (2026-10-05; `scripts/comparator_d.sh`).** Continued from `comparator-c` on
    219,000 rows (every local candidate of every non-held-out train photo, `near_miss.py near-pairs-full`; one pass, 6.9 h). Same held-out photos, scoring and
    combiner as lesson 44.
    - Within-photo AUC (< 1 km vs 1-25 km), c -> d: dev 0.795 -> 0.805, benchmarks 0.782 -> 0.777, held-out train 0.806 -> 0.803.
    - Combiner with comparator-d, change in top-1 < 1 km vs the reranker top-1: dev +1.1 [+0.1, +2.2]; benchmarks +0.9 [+0.3, +1.4] (17.1% -> 17.9%; fixes 78
      of 812 near-misses, breaks 46 of 634 exact), vs +0.8 with comparator-c. < 25 km unchanged.
    - Reading: the single-exemplar visual judge has saturated near AUC 0.8 for this band; more of the same pairs is not the lever. Lesson 26's "loss still falling"
      hope is now tested twice (b: 3x rows, +0.3; d: 4x rows, +0.1).

46. **Better combiners don't help either: the signal is the limit (2026-10-05; `near_miss.py combiners`).** With comparator-d's one-exemplar scores, variants
    chosen by 5-fold CV on the 4,317 held-out train photos only: local features; + the logit; + incumbent- and photo-relative views (logit minus the top-1's,
    minus the photo's best, softmax and rank within the photo); each with a margin gate (switch away from the top-1 only if the score margin exceeds 0-3).
    - CV change at 1 km: local +0.2, + logit +0.4 (gate 2.0: +0.5), + relative +0.5 (gate 0.25, chosen).
    - Chosen variant scored once: dev +0.8 [+0.0, +1.7], benchmarks +0.7 [+0.2, +1.2] (fixes 63 of 812, breaks 36 of 634). The gate trades fixes for breaks
      about 1:1 (benchmarks, + logit: gate 0 fixes 77 / breaks 48, gate 2 fixes 68 / breaks 39).

47. **Map search: searching reaches the truth, but neither SigLIP2 nor keypoint matching can tell it apart (2026-10-05; `map_search.py`, `geo_match.py`).**
    Pivot (user's decision): an agent that moves over the map instead of choosing from a fixed list. Cheapest tests first, on the 1,000 MP16 dev photos.
    - Hand-written zoom search with SigLIP2 only (top 3 seeds, 25 -> 5 -> 1.5 km, ~18k gallery photos scored per query, 0.6 s / photo on CPU): visited points
      within 1 km of the truth: top 24 spots 28.6% vs the fixed local list 26.2% and the reranker pool 30.5%; best-spot picks lose to the top-1 (-0.8 to -1.1).
    - Reach with a verifier budget (`truth-rank`): for near-misses (n = 183), the first gallery photo within 1 km of the truth among all photos within 25 km
      of the top-1 (median 13k), by SigLIP2 rank: top 30 56.8% (= the fixed list's 55.7%), top 100 69.9%, top 300 82.0%, any 94.5%. So a verifier that checks
      100-300 photos per query could reach 14-26 pts more near-misses (~+3 to +5 pts of the dev 1 km ceiling) — if it can pick them out.
    - Keypoint verification (DISK + LightGlue + MAGSAC F-matrix inliers, pretrained, query vs each of the top 100; 342 dev photos with top-1 < 25 km, 34k
      pairs, 12 pairs/s): within-photo AUC (photo < 1 km vs 1-25 km from the truth) 0.54 (SigLIP2 0.58 within the same top 100). Best rule (inlier-weighted
      consensus of photos with >= 120 inliers, else keep the top-1): +1.5 [-0.3, +3.2] on the 342 = +0.5 on dev. Moving to the single best-matched photo loses.
    - Why: matching finds *what* is shown, the label is *where the camera stood*. Strong matches are landmarks (a Prague church interior, the Alhambra at
      night; 180-580 inliers) photographed from spots 1-10 km apart or with noisy GPS; non-landmark photos sit at a 20-50 inlier noise floor.

48. **Up to 4 exemplars per local candidate (comparator-d) lifts the AUC but not the top-1 (2026-10-05; `scripts/near_multi_exemplar.sh`, `near_miss.py
    combiners`).** 3.5 exemplars per candidate, 427k comparisons. Within-photo AUC (< 1 km vs 1-25 km), first exemplar -> mean of all: dev 0.805 -> 0.841,
    benchmarks 0.777 -> 0.842, held-out train 0.803 -> 0.855. CV on held-out train picks "4 exemplars + relative, gate 3" (+0.95 CV): dev -0.2 [-1.3, +0.9],
    benchmarks +0.9 [+0.3, +1.4]. The plain 4-exemplar combiner (not chosen, CV +0.83) gives benchmarks +1.3 [+0.7, +1.8], dev +0.8. Below the +1.3 bar set
    beforehand: the comparator-over-the-fixed-list line is closed. An AUC of 0.84 within the photo still flips about 2 near-misses per exact photo broken.

49. **No verifier we have separates the right place among the search's look-alikes, so map search has nothing to steer by (2026-10-05;
    `scripts/map_comparator.sh`, `geo_match.py report --verifier comparator-d`).** comparator-d on the same top-100 gallery photos as lesson 47 (30,777 MP16
    pairs; OSV photos unscored). Within-photo AUC, photo < 1 km vs 1-25 km from the truth, near-misses: comparator-d 0.575, SigLIP2 0.543, keypoint inliers
    0.524 (all photos with top-1 < 25 km: 0.616 / 0.577 / 0.541). The top-scored photo is < 1 km for 15.3% of near-misses (inliers 13.1%, SigLIP2 6.6%).
    Rules on comparator scores: -1.3 to +0.3 on dev.
    - The same comparator reaches 0.84 over the fixed local list (lesson 48) because there the wrong candidates mostly lack a look-alike photo nearby; among
      the 100 most similar photos within 25 km, every wrong one is a look-alike, and its skill falls to near chance. Most of its 0.84 is "is there a
      look-alike near this place", which SigLIP2 already encodes.
    - Consequence for RL map search (the plan's cheapest test, hand-written policy steered by comparator + matcher): not run, since neither signal separates
      the places a policy would have to choose between. Search reach is there (lesson 47: 70-82% of near-misses within 100-300 photos); a signal is not.

50. **Knowledge-grounded choosing, labels and the zero-shot control (2026-10-05; `place_labels.py`, `name_score.py`, `scripts/name_scores.sh`).** Direction
    (user's decision): teach the 4B place knowledge from geotagged photos (MP16 buckets 0-98 for this stage only; the chooser stays on bucket 99), and use it
    as a score per candidate name, combined with the rank.
    - Labels: MP16-Pro's "neighbourhood" field mixes boroughs (Manhattan), the city repeated (Paris | Paris), villages and Japanese city blocks; 126k names,
      half with < 5 photos. Overture divisions (release 2026-09-23.1, land polygons, point-in-polygon in DuckDB; 300k photos in 7 min): locality 70% of
      photos (median 83 km^2), neighbourhood 22% (5 km^2), macrohood 17% (5 km^2), microhood 7% (1 km^2); 38% of photos have a label finer than the city.
    - Label noise: 1,255 near-duplicate pairs by different photographers (SigLIP2 cosine >= 0.95, within 2 km) are a median 198 m apart (p75 722 m, p90
      1.3 km); their labels agree: locality 95%, neighbourhood 83%, macrohood 78%, microhood 62%. Street labels (streets ~100 m apart) would disagree most
      of the time from GPS error alone, so they were not built. Targets: country > region > city > neighbourhood (or macrohood).
    - Scoring: log P(each level of "country > region > city > neighbourhood" | photo, "Where was this photo taken?") from vLLM prompt logprobs, minus the
      same without the photo (name popularity). Token-to-level assignment by the token's last character (" Maryland" starts in the separator).
    - Zero-shot control, base Qwen3.5-4B, dev (11,653 photo-name scorings, ~5 min): within-photo AUC (candidate < 25 km vs >= 25 km) 0.50-0.53 for every
      level and variant; CV combiner rank + name scores -0.7 [-1.3, -0.2] at 25 km. Among the 538 photos whose pool spans >= 2 countries including the
      true one, the base 4B ranks the right country first 48.9% (popularity-corrected 43.5%) vs the reranker top-1 67.5%. The scoring works; the base model
      knows too little (as in lesson 37). The knowledge-trained 4B has to beat this control.
    - Label fix (2026-10-05, before any training): where city-level areas overlap (Chicago and its townships, New York and its boroughs) the first labeller
      took the smallest; now the city is the most populous containing locality, and the finer level is kept only if smaller than the city with a different
      name. All 4.1M MP16 photos relabelled (`place_labels.py label-all`, DuckDB, ~1 h on CPU; `artifacts/place_labels/mp16.parquet`): 23% reach
      neighbourhood, 46% stop at city, 27% at region, 3% unlabelled. The zero-shot numbers above used the first labels; `scripts/knowledge_run.sh` re-scores
      the base model at 448 px on the new ones (first labels at 448 px: -0.6 dev / -0.4 val).
    - Training data (`knowledge_data select`): 250k photos from 36,753 cities in 191 countries (of 3.58M eligible: bucket < 99, photographers of dev, the
      benchmarks and wikimedia blocked, near-duplicates of any evaluated photo removed, at most 400 per city). Training (LoRA, 448x448, batch 8 x 2 without
      gradient checkpointing, 10.9 photos/s and ~27 GB on the 5090) has not run: the overnight run of 2026-10-05 failed at selection (`.venv` has no pyarrow).

51. **Pinpoint's attention reranker is the baseline from now on, evaluated without the photographer filter as in prior work; with the filter it ties
    the one-step reranker (2026-10-07;
    `models/pinpoint_reranker.py`, `pinpoint_reranker_eval.py`, outputs in `artifacts/pinpoint_reranker/`).** The submission's full model (contrastive
    retrieval -> union of 8 image + 2 raw-image + 2 GPS neighbours from MP16 buckets 0-98 -> attention reranker with an OSV support token), run through
    the submission's own code with one change: its exact MP16 search drops the query photographer's rows. Without the filter it matches the submission's
    unchanged inference on all 7,533 im2gps3k + yfcc4k photos (same 12 candidates and top-1 on 100%). One photo per call, as the submission evaluates:
    batches of 64 change the candidate set on ~25% of photos and the top-1 on ~4% (bf16 search numerics).
    - Whole benchmarks, % within 1 / 25 / 200 / 750 / 2500 km:

      | | n | filtered | no filter |
      |---|---|---|---|
      | im2gps3k | 2,997 | 20.3 / 47.2 / 62.9 / 78.4 / 90.1 | same (no im2gps3k photographer is in MP16) |
      | yfcc4k | 4,536 | 14.6 / 29.3 / 45.6 / 65.3 / 80.6 | 32.8 / 44.1 / 56.4 / 71.8 / 84.1 |
      | wikimedia | 6,017 | 8.5 / 24.7 / 57.5 / 82.5 / 92.6 | (no overlap) |

      3,111 of 4,536 yfcc4k photos are by an MP16 photographer; without the filter yfcc4k gains 18 pts at 1 km and 15 at 25 km from the photographer's
      own gallery photos. Any unfiltered number for yfcc4k (including the submission's own benchmark script) is not comparable.
    - Against the one-step reranker on the 3,713 eval-half photos (placeholders dropped), paired change [95% bootstrap CI]: 1 km +0.9 [+0.0, +1.8],
      25 km -0.9 [-1.9, +0.2], 200 km -1.3 [-2.4, -0.2]; im2gps3k +2.0 / +1.1 / +0.1, yfcc4k +0.1 / -2.1 / -2.2. Wikimedia: +0.5 / -0.3 / -1.9.
      The one-step reranker was fitted on the benchmark tune halves, so it has a small home advantage; the attention reranker saw only MP16.
    - Oracle over the attention reranker's 12 candidates: 30.6 / 55.9 / 75.8 (one-step pool of ~17: 34.2 / 60.2 / 80.3); wikimedia 16.4 / 44.9 / 81.2
      (18.1 / 49.4 / 83.3). Choosing headroom is about 18 pts at 25 km with either.
    - Unfiltered (the baseline from now on, user's decision, comparable to prior work), 3,713 eval-half photos: 29.5 / 47.4 / 61.9 / 76.7 / 88.1
      (im2gps3k 21.5 / 48.9 / 64.4, yfcc4k 34.8 / 46.4 / 60.2); oracle over its 12: 43.7 / 64.2 / 80.6.
    - So lessons 1-50 hold against the new baseline to within about a point when both are filtered. The choosers, comparator data and name scores still use the one-step pool
      and order, built with the filter: on yfcc4k they must be rebuilt unfiltered before comparing with the unfiltered baseline (im2gps3k and wikimedia
      are unaffected).

52. **Where Pinpoint's attention reranker fails, by eye (2026-10-07; one reader, the agent; 140 random > 25 km misses, 70 per benchmark, eval half, no
    filter; two scans of 40 and 100).** Misses are ~750 of 1,480 im2gps3k and ~1,190 of 2,233 yfcc4k eval-half photos; a candidate < 25 km is among its
    12 for ~31% of them. Each photo was guessed blind (guesses written down), then checked against the label.
    - Within 25 km by eye, reranker wrong: 8 of 140 (+1 low-confidence: Mendoza's grape-harvest parade). im2gps3k 6 of 70: Qingdao TV tower, Athens'
      Acropoli metro station (Parthenon-frieze copy), Duxbury's Powder Point Bridge, the Prince Charles Edward cairn at Arnish near Stornoway (a plaque),
      Mount Kinabalu's summit, a Porsche dealer in Luxembourg (porsche.lu on the car). yfcc4k 2 of 70: Toledo (a "San Silvestre Toledana" race banner),
      Moab (Colorado River canyon). Right answer among the 12: 4 of 8 (Athens 3rd, Kinabalu 2nd, Toledo 6th, Moab 2nd); never proposed: 4 (Qingdao,
      Duxbury, Stornoway, Luxembourg). Readable text drives 3 of the 8; SigLIP2 retrieval barely uses it.
    - Right country where the reranker had the wrong country or continent: 3 (Thai temple statue vs Arkansas, Spanish graffiti vs Valparaiso, Iberian
      roller hockey vs Switzerland).
    - Shared errors, where the photo points somewhere else: 5 (a German shop window in Shanghai; Chicago-like towers in San Francisco; Seville-style tiled
      pergola in Buenos Aires' Rosedal; a Polish Coca-Cola ad at Atlanta's Coca-Cola museum; a French-plated car at a UK show). Plus my own outright misses
      (a "UNIS" gym I put in Hanoi was in Beijing).
    - Region only, usually the reranker's region too: ~50 (Valais, Scottish Highlands, Tuscany, Patagonia, Morocco, Japan, Vietnam, Taiwan, California...).
      Getting the exact place would need a landmark.
    - No place signal: 73 of 140 (pets, people, interiors, food, sky, close-ups, sports halls, fireworks); yfcc4k 39 of 70, im2gps3k 34 of 70.
    - Choosing: of 45 misses with a right candidate among the 12, a person would pick it for ~4 (Athens, Kinabalu, Toledo, Moab); ~15 of the 45 have no
      place signal at all (the right candidate is there by luck), the rest are look-alike neighbours (Saint-Ouen vs Bordeaux, Evolene vs Gampel, Suzhou vs
      Shanghai). Agrees with lesson 35: little of the gap to the 12-candidate oracle is recoverable by inspection.
    - So the visible wins are landmarks and text, ~6% of misses (~3% of all photos at 25 km), half outside the candidate list. A gain there needs a
      recognizer or a text reader that can propose new places, not a better chooser.
    - Expert-with-tools estimate (same 140, second pass, labels already known so optimistic; methods: web search of visible text, landmark and object
      identification, terrain / peak and coastline matching on satellite or 3D maps, skyline matching, Street View, regional knowledge; reverse image search
      of the photo itself excluded as leakage): better than the reranker on ~30 (13 high + 17 medium confidence; +3 low), im2gps3k 17, yfcc4k 13; 26 of them
      within 25 km, 4 only country-level. Methods: landmark / object ~9, text search ~8, terrain matching ~7, skyline ~4, regional knowledge ~2. 17 of the 30
      had a right answer among the 12 candidates, so verification against candidates (terrain, skyline, a found landmark) could fix about half; the rest need
      new places. Unaided, by eye: 8. Roughly 10 pts at 25 km over all photos (~12 im2gps3k, ~8 yfcc4k) is the expert ceiling on these error types.

53. **The old benchmarks are flawed; commons26 is a replacement built to fix them (2026-10-07/08; `commons_bench.py`, data `/data/pinpoint/commons26/`).**
    - im2gps3k (n = 2,997): 198 photographers, 85% of photos from photographers with >= 10; 51% share an exact coordinate with another photo (43 at one
      point); 62% have a same-photographer photo within 1 km. Errors cluster by photographer: attention-reranker SE at 25 km 2.25 pts clustered vs
      0.91 iid (design effect 6.1; effective n ~500). yfcc4k: 3,481 photographers, design effect 1.3.
    - Our tune / eval halves hash each photo alone (`strategy_search.py:102`): 97.5% of im2gps3k eval-half photos have their photographer in the tune
      half, 47% a same-photographer tune photo within 1 km (yfcc4k 24% / 4%). Anything fitted on the tune halves can learn album locations.
    - Also: 19.5% of yfcc4k has Flickr accuracy <= 12 (city level or coarser; reranker < 1 km 27 vs 34%); US 26-33%, southern hemisphere 6-8%;
      Places365 says 18% of im2gps3k is indoor; Flickr placeholders not flagged by yfcc4k's `Placeholder` column.
    - commons26: every geotagged JPEG uploaded to Commons 2026-07-01 to 10-06 (search `haswbstatement:P1259`, ~10k / day; 4 days hit the 20k read cap),
      kept if EXIF GPS + capture date >= 2026-07-01 (GPS date stamp first) + camera model + long side >= 1024 px + GPS within 50 m of the page's camera
      location, not bots / Mapillary: 227,086 photos from 3,935 uploaders. Pick: <= 3 per uploader (2 gives 4,780), >= 2 km apart, continents in turn,
      no country > 15%: 5,668; 101 non-photos and 6 with printed coordinates dropped -> 5,561 from 3,241 uploaders, 140 countries; split by uploader
      (dev 935, test 4,626). Still 52% Europe, 3% Africa: Commons is European.
    - Gallery near-duplicates are not a problem by construction: 23 of 5,668 have an MP16 photo at cosine >= 0.95 (OSV-5M 1); >= 0.90 (527) are mostly
      look-alikes hundreds of km away (clouds, altars, roads), so they are flagged, not dropped.
    - Tiers (Qwen3.6-27B Q4, no location shown, JSON-schema output; without the schema it writes an analysis and runs out of tokens): landmark 1,794,
      city 608, region 1,583, none 1,576. Against one blind reader (the agent) on 198 photos: exact 113 / 198, locatable vs none 175 / 198 with the v2
      prompt (v1: 99, 159); the 27B over-calls landmark (59 vs 29). Use tiers as coarse slices; the headline excludes `none`.
    - Top-1 MP16 SigLIP2 retrieval, no photographer filter (the only baseline run so far), < 1 / 25 / 200 / 750 / 2500 km:
      | set | n | result |
      |---|---|---|
      | im2gps3k | 2,997 | 14.9 / 37.6 / 50.7 / 67.8 / 83.5 |
      | yfcc4k | 4,536 | 29.5 / 38.1 / 47.0 / 61.9 / 76.5 |
      | commons26 test, all | 4,626 | 3.2 / 10.0 / 26.5 / 57.5 / 80.1 |
      | commons26 test, headline (tier != none) | 3,307 | 4.2 / 13.0 / 33.4 / 68.3 / 88.7 (landmark 7.9 / 18.5; none 0.5 / 2.4) |
      Coarse accuracy is similar; fine-scale accuracy collapses. The old sets reward finding the same scene in a 2010-era Flickr gallery.
    - Not yet run on commons26: Pinpoint's attention reranker (the baseline) and our methods.

- Pinpoint's retriever trained on MP16 md5(image_id) % 100 < 99; its photos get inflated candidates (Pinpoint top-1
  <25 km 40.6% vs 30.4% held out). Train only on the bucket-99 pool (38k; 34.5k train / 3.8k val split by
  photographer); its candidate quality matches yfcc4k eval.
- yfcc4k shares photographers with MP16. Until 2026-10-07 we excluded same-photographer gallery rows everywhere (`tests/test_author_exclusion.py`);
  since then benchmark evaluation is unfiltered, as in prior work (user's decision, lesson 51). Training data still blocks benchmark photographers.
- Burned-in GPS overlays: base-Qwen P(yes) ≥ 0.64 drops 8 photos of the pool.
- Reserve data: yfcc26k (~19.3k usable after photographer exclusion; lat/lon only).

## Engineering facts
- HF processors ignore `chat_template_kwargs`; pass `enable_thinking=False` directly (TRL GRPO passes
  `chat_template_kwargs` straight through, which works). Prompt suffix must be `assistant\n<think>\n\n</think>\n\n`.
- Merged LoRA checkpoints load in vLLM 0.30 (MTP weights dropped; fine without MTP).
- TRL SFT `chunked_nll` computes logits only at labelled positions (LoRA must not wrap lm_head).
- Keep gradient checkpointing on: batch 4 without it OOMs on ~1,250-token prompts.
- GRPO step (~45 s, 8 photos × 8 samples) is dominated by trainer passes over 64 copies of a ~1,200-token prompt
  (old-policy forward for vLLM importance sampling, reference forward for KL, forward+backward); vLLM generation is
  only 4.5 s. Prompt length and duplicated prompt compute are the cost, not generation.
- TRL colocated vLLM needs `mm_processor_kwargs={"max_pixels": 786432}` patched in (`grpo_train.py`), or image-token
  counts disagree with the trainer. Generating 64 answers at once with HF generate OOMs in the prefill.
- `np.load` NpzFile re-reads an array on every key access: wrap in `dict(...)`.
- vLLM eval servers: `--gpu-memory-utilization 0.7` so a small InnerSight job doesn't block startup.
- Temperature: T=0.7 for SFT-model sampling; resolution caps below 786k px change nothing within noise.

## Dead ends (don't redo)
- Verifying candidates by image matching (SigLIP, SIFT/RANSAC) or zero-shot VLM judges (Gemma).
- A region choice learned from retrieval evidence; VLM-chosen region + region-restricted retrieval top-1.
- Zero-shot Gemma 4; Qwen3.5 thinking mode; llama.cpp for rollouts.
- More SFT data or temperature tuning to raise best-of-8 (flat at ~21–23 / 48–55 across all settings).
- GRPO without KL/entropy control; longer runs of the same single-turn GRPO setup (flat after ~25 steps).
- Query rewriting for retrieval: SigLIP2 text queries of place names; fixed image crops (see lesson 7).
- Evidence numbers on the reranker's top-10 (superseded by the no-reranker prompt before it finished training).
- A "LLM picks region → retrieve inside it" pipeline (lesson 5).

## Kept artifacts
- `/data/pinpoint/sft/comparator-b` (adapter): the best chooser signal (lesson 41); `comparator-d`: near-band comparator (45). Also kept: comparator-a,
  comparator-c, comparator-25km, pairwise-a. Merge any of them with `sft_train merge --run <run>`.
- `/data/pinpoint/sft/sft-34k-retrieval` (adapter): best SFT chooser (reranker parity).
- `/data/pinpoint/sft/grpo-kl` (checkpoint-200): best greedy of the GRPO runs.
- Merged weights were deleted to save space (2026-10-02). Rebuild in order: `sft_train merge --run sft-34k-retrieval`,
  then `grpo_train merge --init sft-34k-retrieval --run grpo-kl --adapter checkpoint-200` (GRPO's base is the SFT merge).
- `artifacts/sft/`: MP16 query pool, neighbour/candidate caches, `sft_retrieval.jsonl` (+ `sft.jsonl`), logs.
- `artifacts/strategy_search/`: benchmark caches, all result JSONs, Gemini labels for 1,683 eval photos
  (`llm_advantage_labels.json`; OpenRouter credits ran out before the other ~2,100), crop-query search cache
  (`query_headroom_crops.npz`), evidence-test answers (`evidence_test_vlm_*`).
- `/data/pinpoint/geonames/allCountries.txt` (GeoNames dump; `evidence_test landmarks` rebuilds the landmark index).
- `artifacts/place_labels/mp16.parquet`: place labels of all MP16 photos (lesson 50); `artifacts/knowledge/selected.json`: knowledge training photos.
- Snapshot of `artifacts/` (2026-10-06, 2.6 GB): Hugging Face `kinghorton42/geo-benchmarks`, folder `artifacts/`.
- Environments: `.venv` (analysis), `~/.venvs/sft` (SFT), `~/.venvs/grpo` (GRPO with vLLM), `~/.venvs/vllm` (serving), `geo`, `overture`, `match`
  (tools); package lists in `envs/`.
