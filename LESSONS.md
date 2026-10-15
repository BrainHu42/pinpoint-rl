# Lessons (detailed)

The numbered lessons behind `LEARNINGS.md`'s summary, with full numbers. Conventions (metrics, eval sets, standard
errors) are at the top of `LEARNINGS.md`. Add new lessons here and a one-line entry to the summary there.
Modules and scripts named in lessons 1-51 that are no longer in the tree are at git tag `pre-cleanup-2026-10-07`.
Modules and scripts named in lessons 1-53 that were removed with the one-step reranker (lesson 54) are at commit 57918cb.

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

51. **Concept queries as similarity scores around each candidate don't help either (2026-10-02, full eval halves; code and outputs
    not kept).** Fills in lesson 15's untested "similarity scores". Base 4B wrote 3 visual-clue queries, 3 place queries and a caption per
    photo (22.6k requests, 0 failures). Each candidate was scored by the queries' best SigLIP2 match among its nearest 400 MP16 + 400 OSV
    photos (within 1 / 5 / 25 km), asking "does this place have stadiums with fountains?" instead of searching for the place.
    - Alone the score picks about as well as photo-to-photo matching (top-1 <= 1 / 25 km: visual 9.5 / 27.1, whole image 10.2 / 26.8,
      another photo's queries 4.1 / 25.1), so the signal is real.
    - As reranker features it is redundant: +0.3 / +0.0 (CI -0.6 to +0.6 at 25 km; 3-seed ensembles); visual + image +0.4 / -0.1.
    - As new candidates (6 hits per photo) over the shown-candidate oracle 33.6 / 58.9: visual 33.9 / 59.5, caption 33.9 / 59.6, place
      queries 34.6 / 60.3, 6 extra whole-image hits 34.7 / 60.9 (consistent with lesson 13).
    - Query quality wasn't the problem: visual queries are concrete (6.6 words, 14% use sign text, 11% generic).

52. **The knowledge SFT adds country / region knowledge, but not where retrieval fails; GRPO not run (2026-10-07; `knowledge_check.py`,
    `name_score.py --levels 3`, `knowledge_oracle.py`; scripts `knowledge_check.sh`, `knowledge_ceiling.sh`).** `knowledge-a` = Qwen3.5-4B LoRA-SFT on
    250k MP16 bucket 0-98 photos -> "country > region > city" (1 epoch, 3.5 h, held-out loss 0.99 -> 0.82, still falling), 448 px.
    - Naming the place (full eval halves, greedy): right country / region / city 61.7 / 31.3 / 16.1 % vs base 4B 54.4 / 24.7 / 15.2; on photos where
      the reranker top-1 is >= 25 km off 53.2 / 17.6 / 4.1 vs 45.5 / 13.2 / 4.3 (country +7.7 [+6.1, +9.3], region +4.4 [+3.0, +5.7], city -0.1).
      Only 86.4 % of its answers keep the 3-level format (base 99.6 %).
    - As a chooser, no training (chosen / fitted on MP16 dev, applied once to the eval halves): "best-ranked top-3 candidate in the named country"
      -0.2 [-0.5, +0.1] at 25 km (base -0.1); rank + candidate-name log-prob combiner -0.1 [-0.2, +0.0] (base -0.4). Dev within-photo AUC of the name
      scores: country 0.541 vs base 0.516, region 0.565 vs 0.545, cumulative to city 0.585 vs 0.543. The pre-set GRPO gate (>= +0.5 at 25 km, or
      AUC +0.03 at country / region) failed, so the GRPO A / B runs were not started.
    - Why (`knowledge_oracle.py`): knowledge is the right lever but the model lacks it where it matters. Picking the best-ranked candidate in the
      *true* region would add +10.8 [+9.8, +11.8] at 25 km (fixes 413 of the 792 photos where the top-1 is wrong and a candidate is right, 4 breaks;
      true city +8.4, true country +3.3). On those 413 photos knowledge-a names the right region 10.9 % (base 9.0 %), vs 52.3 % on photos the
      reranker already gets right: its knowledge, like every earlier evidence source (17-19), sits on the photos retrieval already solves.
    - Next lever is far more knowledge per photo at region level, not RL on the current model: a larger model, much more knowledge data, or tools.

53. **The photos retrieval fails on are hard for the VLM too, not under-covered (2026-10-07; `knowledge_coverage.py`).** Region named right
    (knowledge-a / base 4B) by how many of the 248k SFT training photos share the photo's true region, on the 413 region-fixable photos (lesson 52) vs
    photos the reranker gets right: 1-50 training photos 0.0 / 5.7 % vs 11.6 / 8.7 %; 50-300 5.6 / 2.8 vs 42.0 / 21.8; 300-1,500 5.9 / 3.9 vs
    55.4 / 37.6; 1,500+ 20.8 / 17.5 vs 62.0 / 59.2. Median training photos in the true region: fixable 872, reranker-right 1,073. The SFT learned
    these regions (the reranker-right column gains most), but the fixable photos don't show them; more or harder knowledge data of the same kind
    is unlikely to help. Next direction: clues that aren't overall appearance, used through tools (`AGENT_DIRECTION.md`).

54. **Pinpoint's attention reranker is the reference; our one-step reranker and pool are retired (2026-10-07; `models/pinpoint_reranker.py`,
    `pinpoint_reference.py`, `photo_sets.py`).** Ported from pinpoint-yogi (27a9bfe) without Lightning, the index streamed from disk. Exact: on 320
    benchmark photos the original code (Lightning stubbed, the full index fed lazily) gives identical query encodings, candidate rows and rank
    features, and identical bf16 scores given the same features. Tie order matters: many index scores tie exactly (photos at one coordinate
    share a GPS embedding), the reranker reads each hit's rank, and a naive top-k changed the top-1 on 5 of 64 photos; the port follows the
    original's chunked top-k sequence. Index photos by the query's photographer are excluded (the original doesn't).
    - Eval halves (n = 3,795), % within 1 / 25 / 200 / 750 / 2500 km, both protocols (`pinpoint_reference bench` / `bench-own`, the user asked
      for both, 2026-10-13):

      | Index | Set | Top-1 | Oracle over its pool |
      |---|---|---|---|
      | query photographer's photos excluded (our rule; all our results) | im2gps3k | 21.8 / 48.9 / 64.6 / 79.6 / 89.7 | 36.1 / 64.6 / 80.9 / 91.9 / 97.2 |
      | | yfcc4k | 14.8 / 29.8 / 47.0 / 66.1 / 81.2 | 26.3 / 48.4 / 70.2 / 88.2 / 96.5 |
      | | both | 17.5 / 37.2 / 53.9 / 71.4 / 84.5 | 30.1 / 54.7 / 74.4 / 89.6 / 96.8 |
      | kept (the original Pinpoint's protocol) | im2gps3k | 21.8 / 48.9 / 64.6 / 79.6 / 89.7 | 36.1 / 64.6 / 80.9 / 91.9 / 97.2 |
      | | yfcc4k | 33.5 / 44.8 / 58.2 / 73.0 / 84.8 | 47.3 / 61.9 / 77.8 / 90.7 / 97.1 |
      | | both | 28.9 / 46.4 / 60.7 / 75.5 / 86.7 | 43.0 / 62.9 / 79.0 / 91.1 / 97.1 |

      im2gps3k is unaffected: none of its photographers appear in MP16 (0% matched; MP-16 is said to exclude im2gps photographers). yfcc4k
      and MP16 both come from YFCC100M: 68.9% of yfcc4k eval photos have a photographer with photos in the index (median 162 each), and
      keeping them more than doubles 1 km (14.8 -> 33.5%) and adds 15 pts at 25 km. That gain is retrieval of the photographer's own
      photos of the same trip, not geolocation, which is why our rule excludes them; compare published numbers only under the same protocol.
      Pool size 11.6 (excluded) / 11.4 (kept). One-step reranker: 16.7 / 38.2, oracle over
      its ~17-candidate pool 33.4 / 59.0; the union of both pools (28.7) would reach 37.4 / 63.1. Pinpoint's pool alone was chosen (user).
    - MP16: dev (3,836) 16.0 / 33.9, oracle 27.0 / 49.5; held-out rest (34,531) 15.4 / 32.3; trained-on photos 19.0 / 38.9, oracle 32.4 /
      57.7: its index still favours photos its retriever trained on, so training stays on bucket 99.
    - Target photos (`knowledge_oracle`): 665 choosing photos (top-1 >= 25 km off, a pool candidate < 25 km; was 792), 371 region-fixable
      (was 413): true-region knowledge +9.5 [+8.6, +10.6] at 25 km (city +6.9, country +3.0). knowledge-a names the region on 13.2 % of
      the 371 (base 10.8 %), vs 52.1 % on reranker-right photos: lesson 52's conclusion holds. MP16 dev (`devall`): 600 choosing, 347
      region-fixable (+8.9).

55. **Constraint tools from photo clues: precise but rare, and smaller than first reported (measured with the old one-step reranker's ~17-candidate pool, before lesson 54) (2026-10-07; `observe.py`, `geo_tools.py`, `poi_query.py`, `agent_targets.py`).** A VLM reads
    each photo into JSON clues (text, language, script, driving side, features from a GeoNames-mapped vocabulary, place names, country); tools score
    each retrieval candidate; listwise combiner of rank + tool features (CV on MP16 dev + val, n = 2,000, 208 fixable photos).
    - **Correction:** "val" (query_evidence `val`) is 1,000 benchmark eval-half photos, not MP16, so every "dev + val" number here mixes in test photos.
      Clean (combiner fitted on MP16 dev only, `agent_clean.py`): text matched to a GeoNames name near the candidate (9B clues) **+0.2 [+0.0, +0.3]**
      at 25 km on all eval halves (7 fixes / 1 break), **+0.1 [-0.0, +0.3] on the 2,795 never in val** (4 / 1). Withdrawn: +0.4 [+0.2, +0.6] (fit on dev + val).
    - Overture POI names near the candidate +0.2 [-0.1, +0.7] (13 / 8); text -> GeoNames region +0.1; visible features near candidate -0.4 (AUC <= 0.545);
      country rules and the VLM's country guess 0.0 even with the 9B's fuller language / driving-side fields; all tools -0.1 (overfits).
    - Ceiling: read text names a place in the true region specifically (<= 5 regions) on 4 / 102 dev and 7 / 106 val fixable photos.
    - Extraction: greedy JSON from the 4B looped on long text lists (8.8% unparsed); text last + capped + repetition penalty 1.05 -> 0%.
    - Crops: 9B text from 4 overlapping tiles per photo lifts text coverage on fixable photos 43% -> 63-73% but rank + text stays +0.4 [-0.0, +0.8]
      (14 / 6): reading quality isn't the limit, the text isn't geographic.

56. **Capacity and resolution barely move region knowledge on hard photos (measured with the old one-step reranker's ~17-candidate pool, before lesson 54) (2026-10-07; `agent_capacity.py`, `scripts/capacity.sh`).** Right region
    (country > region) on dev + val fixable photos (val = benchmark photos; diagnostic only, nothing fitted): Qwen3.5-4B 13.9%, Qwen3.5-9B 16.8% (+2.9 [-1.9, +7.7]); city 6.3 vs 10.7 (+4.4 [+0.6, +8.8]).
    Same 4B, dev fixable: 448 -> 786k px +3.9 [+1.0, +7.8] region, +6.9 country (no change on reranker-right photos); 1.6M px = 786k. Agreement of
    4B and 9B as a switch rule: best +0.4 [-0.1, +0.9]; requiring agreement halves fixes and breaks alike (`agent_vote.py`).

57. **What the photos retrieval fails on contain (measured with the old one-step reranker's ~17-candidate pool, before lesson 54) (2026-10-07; by eye, one reader, 24 random dev fixable photos).** Nothing place-specific 58% (indoor,
    people, food, sky, generic nature); a readable business / organisation name ~15% (POI search; the 9B misread one); a recognisable landmark 12%
    (landmark matching); country-level cultural objects 12% (e.g. Strandkorb beach chairs). GeoNames suits places, not the business names hard photos
    show; Overture places covers more but not all (2,907 vs 1,946 of the read n-grams matched).

58. **Landmark / object identification by crop search over MP16 doesn't work (2026-10-07; dev3k with Pinpoint's pool; `observe.py objects`,
    `landmark_id.py`).** Motivation: by the user's own analysis ~6% of Pinpoint's >= 25 km mistakes are fixable by landmark / object identification
    (Qingdao TV tower, Athens metro frieze, Duxbury bridge). The 9B *names* a landmark correctly (city within 25 km) on only 0.6% of dev3k mistakes,
    so recognition, not the idea, is the gap.
    - Tool: the 9B boxes the most distinctive object (2,557 / 3,000 photos; often generic: "rainbow", "brown insect"); the crop is searched with
      SigLIP2 against all 4.1M MP16 photos (same-photographer rows excluded); hits verified by SuperPoint + LightGlue.
    - Crop search alone: top-1 hit < 25 km on 14.3% of searched photos vs Pinpoint 34.5% on the same photos. On Pinpoint's mistakes a hit within
      25 km of the truth is among the top-1 / 5 / 20 for 2.4 / 7.2 / 17.1%, but mostly nearby photos, not the same object.
    - Verification can't pick them: raw LightGlue matches, right vs best wrong hit median 8 vs 10 (right wins 39%); RANSAC (MAGSAC) inliers 11 vs 21
      (right wins 12%). Switch rules over thresholds and hit agreement: best +0.0 at 25 km, looser ones negative (e.g. -1.1, 13 fixes / 47 breaks).
    - Reading: strong geometric matches come from the same *kind* of thing elsewhere (repetitive facades, chain signage). Identifying the user's
      examples needs real recognition (a much stronger model, or a curated landmark collection), not crop retrieval in MP16.

59. **A 27B VLM names more landmarks, but not enough of the right ones (2026-10-08; dev3k, Pinpoint's pool; Qwen3.6-27B-FP8 via `observe.py
    landmarks`).** Same landmark prompt as the 9B (lesson 58). On Pinpoint's 1,997 >= 25 km mistakes: landmark named 10.5% (9B 2.5%), city right
    (a GeoNames place of that name within 25 km of the truth) 1.7% (9B 0.6%), vs the ~6% the user judged identifiable by landmark / object.
    On the 1,003 photos Pinpoint gets right it names a city away from Pinpoint's top-1 189 times, never right: following every answer would cost
    ~-5 pts. Switching only to a pool candidate near the named city: +0.2 pts at 25 km (16 fixes / 10 breaks at "high" confidence; nearly every
    answer is "high"). Recognition triples from 9B to 27B, but precision and coverage stay far below the >= 1 pt target.

60. **A specialised instance-retrieval descriptor (DINOv2-large) barely helps landmark / object identification (2026-10-08; dev3k, Pinpoint's pool;
    `landmark_id.py rescore`, `scripts/landmark_rescore.sh`).** Lesson 58's crop search kept to SigLIP2's top 100 MP16 hits, re-scored with DINOv2-large
    (CLS token; GeM over patch tokens; 336 px). On Pinpoint's 1,587 searched mistakes a hit within 25 km of the truth is among the top 100 for 34.2%,
    mostly nearby photos rather than the same object. Top-1 hit right on mistakes: SigLIP 2.3%, DINOv2 CLS 3.3%, GeM 2.8% (on photos Pinpoint gets
    right 36.8 / 39.6 / 33.1%). Best switch rule (DINOv2 CLS score >= median and 3 of its top 5 hits within 25 km): +0.2 [-0.0, +0.5] at 25 km
    (12 fixes / 5 breaks); every other rule +0.2 or worse. No GLDv2-trained descriptor (CVNet, SuperGlobal) is on Hugging Face. With lessons 58-59,
    the landmark / object line tops out near +0.2 pts with open models and MP16 as the reference collection.

61. **Deciding by place doesn't rescue the landmark crop search (2026-10-08; dev3k, Pinpoint's pool; `landmark_vote.py`).** From the top-100 MP16
    crop hits and their SigLIP2 / DINOv2 scores (lesson 60):
    - Location voting (each hit's place scored by similarity-weighted hits within 1 km; switch when the best place is far from Pinpoint's top-1,
      scores high and beats the next place >= 25 km away): no rule beats Pinpoint; best -0.0 (2 fixes / 3 breaks), looser rules -1.6 to -6.3.
      Voting favours densely photographed places, which Pinpoint already gets right.
    - Scoring Pinpoint's candidates (max / summed crop-hit similarity within 1 / 5 km, hit counts; listwise combiner with rank, CV on dev3k):
      +0.1 [-0.4, +0.5] at 25 km (27 fixes / 25 breaks), -0.7 at 1 km.
    With lessons 58-60: the crop search's hits carry almost no information beyond Pinpoint's own retrieval.

62. **The landmark / text target exists in dev3k; recognition and Pinpoint's pool are what cap the tool (2026-10-08; dev3k).** By-eye check of 40 random
    Pinpoint mistakes (>= 25 km; one reader who knew the truth, so counting only clues identifiable without it): clear landmark / object 2 (5%:
    Wittenham Clumps, Bagan), +4 borderline (Sebring paddock, Gran Sasso, Trieste's Piazza Unita, a San Diego office block); searchable text 3 (7.5%:
    "City Island Lobster House", a Swiss board "Bellinzona Lugano Chiasso", "AIK Banka - Posetite nas"), +1 borderline; nothing usable ~31 (77%).
    Close to the user's test-set estimate (~6% landmark / object, ~5% text).
    - On the clear cases: 9B and 27B named City Island Lobster House (and read its text), the 27B named Bagan; none recognised Wittenham Clumps; the
      Swiss board needs reasoning about the line. The DINOv2-best crop hit was 146-8,719 km off on all four.
    - Pool restriction: Lobster House and Bagan weren't fixed because no Pinpoint candidate lies near them. Geocoding the 27B's named city (most
      populous GeoNames place of that name in the stated country) and answering outside the pool: 30 fixes but 151 breaks (-4.0).
    - Gate on Pinpoint's confidence: top-1 reranker score median 0.46 when right, 0.31 when wrong (margin to the 2nd candidate doesn't separate).
      Switching only when Pinpoint's score is in its bottom 20%: +0.2 [+0.0, +0.4] (8 fixes / 2 breaks); bottom 50%: +0.2 (18 / 13). Even perfect
      gating caps 27B naming at 30 fixes = +1.0 pt: more correct recognitions are needed.

63. **YFCC100M as a name source: extracted, and its names locate the same few photos (2026-10-08; dev3k, Pinpoint's pool; `yfcc_names.py`,
    `scripts/yfcc_extract.sh`, `scripts/yfcc_extract_nvme.sh`).** YFCC100M metadata (official SQLite, 65.6 GB) -> `data/yfcc100m/geo.npz`: 48,469,829
    geotagged photos from 214,600 users, with title + tags (4.1 GB), description (first 300 chars, 2.0 GB), machine tags (0.2 GB), GPS accuracy,
    year, video flag, image URL parts (README next to the data).
    - Extraction: the SQLite's 1 KB pages make a scan IOPS-bound on EBS (~12-15 MB/s, parallel readers don't help: 3+ h projected). Copying the
      file to the local NVMe instance store (/opt/dlami/nvme, 89 MB/s, 12 min) and extracting there with 4 processes: 37 min.
    - Name lookup on dev3k: phrases from the 27B / 9B landmark names, 9B search phrases (observe `queries`) and 9B full-image + crop text: 10,168
      phrases, 5,756 found (83.2M title/tag, 29.7M description, 1.7M machine-tag hits). The photo itself and its photographer are excluded.
      Proposals (densest 25 km cluster of the best phrase) on 1,567 photos, right on 26.5%; right on 68 of Pinpoint's mistakes (ceiling ~+2.3 pts).
      Gated switch rules (cluster share / size, Pinpoint's top-1 score): best +0.1 [-0.1, +0.3] (6 fixes / 2 breaks); ungated 61 / 328 (-8.9).
      Dropping coarse-GPS matches (accuracy < 11) changes nothing.
    - Same ceiling as the text-search tool's region evidence (branch `worktree-text-search`, `region_evidence.py`): YFCC adds descriptions and
      photographer-written names but not reliable selection. The clue-selection problem is visual (does the text belong to the place?), which is
      the case for the agent in the target design (`TEXT_SEARCH_PLAN.md`). Count-based selection is not worth more work.

64. **Pinpoint restricted to the right region is the large lever: 33.4 -> 61.3% within 25 km (2026-10-08; dev3k; `pinpoint_constrained.py`,
    region filter in `models/pinpoint_reranker.py`).** Pinpoint's retrieval + reranker rerun with each query restricted to index photos of one
    area (Overture division labels of every MP16 photo, `artifacts/place_labels`); unrestricted, the rerun equals the saved top-1 on 99.9%.
    - True country: 41.2% < 25 km, +7.8 [+6.7, +8.9] (255 fixes / 21 breaks). True (country, region): 61.3%, **+27.9** [+26.2, +29.5]
      (859 / 23). Above the oracle over Pinpoint's own pool (50.3%): restricted search finds candidates the original <= 12 never held.
    - So a clue only has to name the region; visual retrieval finds the place inside it. This is the target design in `TEXT_SEARCH_PLAN.md`:
      the agent keeps Pinpoint's answer by default and calls Pinpoint as a tool, restricted to the area its clues point to.
    - Caveat: a perfect-region upper bound; regions are first-level divisions (a US state), so in large ones part of the gain is still
      Pinpoint's ordering. The question becomes where right-region evidence comes from on the photos where Pinpoint's region is wrong.

65. **Text search gives the right region too rarely; a VLM judge fixes its precision but leaves +0.3 pts (2026-10-08; dev3k; `text_search.py`,
    `region_evidence.py`, `clue_judge.py`, `scripts/clue_judge.sh`).** The text-search tool (SQLite FTS5 over 24.7M GeoNames names, 81.5M
    Overture places with websites and phones, 48.5M YFCC100M titles + tags; 8-190 ms per search) on the 9B / 27B phrases of dev3k (landmark
    names, text read on the full image + 4 crops, read place names, search queries; 13,510 searches).
    - Blind pick (heaviest top cluster holding >= half its query's matches): right on 65 of Pinpoint's mistakes, 337 traps; shuffled phrases 4 /
      415. The right place is somewhere in the results for 231 mistakes (7.7 pts; 172 without YFCC).
    - Region evidence, on (photo, region) pairs that differ from Pinpoint's region: query right 148 vs Pinpoint right 2,721; no group beats
      Pinpoint (best: 2+ queries agree and share >= 0.9, 47 / 195; websites / phones 5 / 36). Clues are right where Pinpoint is right (with 3+
      agreeing queries the region is right 53%, yet loses 39 / 193 on disagreements), as in lessons 17-19. A perfect selector + restricted
      Pinpoint: 82 fixes (+2.7).
    - Eye check of the 147 photos where a query names the true region and Pinpoint's region is wrong: 53 luck (a generic word), 39 read in the
      photo (a shop shutter, a kiosk sign, a battlefield plaque), 55 recalled (Glastonbury, VDNKh). Specific traps outnumber them: read text
      354 vs 89 (brands, clothing, posters for other cities), recalled names 953 vs 107 (look-alikes: Atlanta airport for San Jose's).
    - VLM judge (photo + clue + where it points: is the name on a sign / building / plaque or a recognised landmark, or could it refer to
      somewhere else?), on 4,591 specific clues: 4B at P(Yes) >= 0.8 clue right 14 vs Pinpoint right 6 (threshold chosen on the other half: 14 / 6),
      >= 0.9 8 / 1; the 9B is no better. End to end (4B, >= 0.8, 25 photos): the clue's own coordinate +0.3 [+0.0, +0.5] (12 / 4); Pinpoint
      restricted to the clue's region +0.1 (7 / 4); at >= 0.5 -0.1 to -0.3.
    - Judging clues by looking at the photo is the right mechanism, but right clues are few: text search with these readers and offline sources
      is worth about +0.3 pts on dev3k (ceiling +2.7), not enough to carry the agent. Named places are better answered by their own coordinate.

66. **A fine-tuned clue judge lifts read-text clues to +0.7 pts, then stalls (2026-10-09; dev3k, 4B clues; `clue_judge.py sft-data / combine`,
    `sft_train`, runs `clue-judge-4b`, `clue-judge-vs-4b`, `clue-judge-vs-all-4b`).** Clues from the base 4B (read text, read place names,
    search queries) for the 35,359 train photos (169,734 clues; 60,617 specific), searched with `text_search`; labels from GPS only (the
    clue's top place < 25 km from the truth). Perfect-judge ceilings (clue's own coordinate): all clues +2.9 pts, read clues +1.7,
    recalled queries +2.0.
    - Judge on all specific clues (30k rows, natural 28% Yes): held-out precision on read clues 20 / 11 (clue right / Pinpoint right,
      where they disagree; zero-shot 15 / 15); end to end, text clues +0.5 [+0.2, +0.9] (21 fixes / 5 breaks). Recalled names no better:
      the target taught "recognisable place", right mostly where Pinpoint is right.
    - Judge trained only on clues disagreeing with Pinpoint, with Pinpoint's region in the prompt: read clues 26 / 11, end to end +0.6
      [+0.3, +1.0] (23 / 4).
    - With 35k extra photos from MP16 buckets 0-98 (2x the positives): read clues 27 / 12, end to end +0.7 [+0.4, +1.1] (25 / 4).
    - Recalled names lose with every judge (best 10 / 9). Thresholds and clue type were chosen on dev3k, so the +0.6-0.7 is slightly
      optimistic. Fine-tuning roughly doubles the zero-shot gain on read text (about half the read ceiling), but more data and the
      decision-conditioned target add only ~0.1 each: the clues, not the judge, are the limit. Text search tops out near +0.7 pts.

67. **Locatability ceiling: about half the photos carry nothing place-specific (2026-10-09; 120 random dev3k photos; `scripts/locatability_score.py`,
    labels `fixtures/locatability_dev3k.json`).** Blind by-eye pass (truth and Pinpoint unseen); tier = the finest level the content
    supports. P specific place 14%, C city 8%, R region 6%, N country 13%, K continent 12%, **X nothing place-specific 47%** (food, pets,
    flowers, interiors, concerts).
    - Within 25 km: blind guesses 28%, Pinpoint 37%, either 42%; oracle over Pinpoint's pool 52%; Pinpoint restricted to the true region 62%.
    - P + C photos: Pinpoint 21 / 26, blind 24 / 26 (mostly solved). R + N + K: Pinpoint 15 / 38, pool oracle 25 / 38 (the gap). X: Pinpoint
      8 / 56, true-region restriction 30 / 56, but that region knowledge isn't in those images.
    - Estimated ceiling from image content plus today's retrieval: about 42-48% < 25 km (Pinpoint 33.4% on all of dev3k). One reader,
      coarse guesses, +-4 pts.
    - Eye check of 48 dev3k photos where Pinpoint's top-1 is wrong but its pool holds the right place: 40% near-misses (top-1 25-100 km
      away, same kind of place), 42% generic photos, 17% explicit clues (text, recognisable places); physical differences between the
      top-1 and the right candidate (coast, terrain) essentially never. Physical-constraint elimination dropped.

68. **Agreement among Pinpoint's own candidates adds +1.0 pts (2026-10-09; `pool_agreement.py`).** Pinpoint's reranker scores each
    candidate independently. Re-scoring each candidate by the softmax score mass (T = 0.2) of all candidates within 25 km of it, and picking
    the most supported one (setting chosen on train; also the best held-out setting on dev3k):
    - train (35,359 bucket-99 photos, never used for dev choices): 32.4 -> 33.3% < 25 km, **+1.0** [+0.7, +1.2] (1,019 fixes / 682 breaks);
      < 1 km about -0.6.
    - dev3k: 33.4 -> 35.4%, **+2.0** [+1.2, +2.8] (106 / 46); < 1 km -0.7 [-1.3, -0.2].
    - Picking Pinpoint's best-scored candidate inside the most supported area instead keeps 1 km accuracy: train +0.7 (< 1 km -0.1),
      dev3k +1.7 (< 1 km +0.2).
    - The first change in this project to clear +1 pt on independent data, free (Pinpoint's own outputs only). It works where several
      candidates cluster: near-misses and generic photos, the bulk of Pinpoint's recoverable misses.

69. **Geolocation as hypothesis testing adds +2.9 / +3.5 pts on train, +3.8 / +4.6 on dev3k (2026-10-09/10; `HYPOTHESIS_PLAN.md`,
    `hypotheses.py`, `hypothesis_compare.py`, `hypothesis_learn.py`).** Hypotheses = Pinpoint's top-5 agreement areas + clue places
    (text search + zero-shot 4B judge) + the 4B's own place proposals; Pinpoint restricted to a +-25 km box around each; a learned set-level
    comparator (DeepSets MLP, listwise loss, GPS labels, 5-model ensemble) picks one; the answer is its restricted top-1 (or the agreement
    pick inside it: "/ agree").
    - Hypothesis oracle (K = 5 areas + clue places): 46.4% train, 48.5% dev3k (Pinpoint 32.4 / 33.4, pool oracle 49.2 / 50.3); + the 4B's
      place proposals 48.8 / 50.6.
    - Ladder (train 5-fold CV; dev3k trained on all of train): agreement +1.0 / +2.0; hand comparator (mass x support^0.5 x photographers)
      +1.6 / +2.2; learned +2.1 / +2.7; + clue hypotheses and clue features +2.5 / +3.3 (one model) and +2.7 / +3.9 (ensemble); + VLM
      place hypotheses and distance to Pinpoint's top-1 **+2.9 / +3.5 train, +3.8 / +4.6 dev3k** (top-1 / agree; dev3k seeds 1-2 +3.4 / +4.2;
      59-69 breaks; < 1 km +0.5 / -0.7 train). Saved: `artifacts/agent/hyp_final_comparator.pt`.
    - Gains by locatability tier come from R / N / K photos (lesson 67), where the region is recognisable but the spot isn't.
    - What did not help (each within +-0.2 on train CV): areas ranked 6-10, richer restricted-retrieval features, attention between
      hypotheses, training towards the agreement pick's label, 35k extra training photos from buckets 0-98 (Pinpoint trained on them: its
      top-1 is 40% < 25 km there), the zero-shot 4B's "could this photo be from {place}?" P(Yes) (AUC 0.63), and the same question
      fine-tuned on 48k or 110k train_extra hypotheses (AUC 0.78 on dev3k, vs 0.84 for restricted score mass; +0.0-0.1 train, -0.1 to -0.4 dev3k).
    - Why it stops here: of 1,519 dev3k photos with a right hypothesis, the comparator misses 426; in ~70% of those the right hypothesis has
      the weaker retrieval evidence (score mass larger in only 26%). The fine-tuned judge prefers the right one in 58% of such pairs, too
      weak to move the choice. The remaining headroom (oracle 48-51% vs 37-38%) needs evidence the gallery doesn't have.

70. **Look-zoom-rerank (2026-10-12, autonomous run; `PINPOINT_SET_PLAN.md`): retraining Pinpoint's reranker is the lever; zoom choice is
    learnable.** % within 25 km vs the released Pinpoint, train CV / dev3k:
    - Step 1, one flat set model over global + zoomed candidates (~92 per photo): +1.4 / +1.3, the same as the global wide set layer.
      Zoomed candidates' scores are not comparable with global ones; with only 35k photos a flat model doesn't learn area-level reasoning.
    - Step 2, area model (`area_model.py`): one token per hypothesis, attention-pooled from Pinpoint's CLS states in its zoom, plus hand
      features. +4.4 / +4.9 (agreement pick; set pick +4.2 / +4.7, 1 km +1.7 / +1.6), vs the comparator's +4.1 / +4.7.
      - With 16 Pinpoint-only features + states: +4.0 / +4.5.
      - States without hand features: +3.3 / +3.2.
      - Clue places add +0.7 over Pinpoint areas only. The 4B place proposals add +0.2 / -0.3: areas + clues reach **+4.2 / +5.2**.
    - Step 4, Pinpoint's reranker fine-tuned with pinpoint-main's code (`pinpoint_train/`; 6,000 steps x 256 photos from its 4M training
      photos, lr 5e-5, wide 32 / 8 / 8 pools with original-list rank features, same-photographer exclusion, co-trained set head):
      - its own pick over the wide pool: **+3.1 / +3.3** (1 km +1.6 / +1.0), one model, no LLM;
      - control without the set head: +2.1 / +2.3;
      - the co-trained set head's picks: +2.6 / +2.9. Trained on photos the reranker has seen, its picks don't transfer, but co-training
        improves the reranker's own scores by +1.0;
      - the offline set layer on the retrained states: no gain.
    - Setup trap (not a Pinpoint bug): training against the shipped submission index, whose source is named "source_0" while the data
      module tags samples "mp16", silently disables leave-one-out (training accuracy 100%). `train.py`'s own index build names the
      source "mp16", and the released reranker used its own index. Fixed with an alias in `retrieval_index._build_source_lookup`; all
      reported numbers are after the fix.
    - Step 5, RL zoom agent (`zoom_agent.py`): REINFORCE policy over pre-zoom evidence picks which hypotheses to zoom. The chooser sees
      only zoomed hypotheses' own evidence, so there is no leak.
      - 1 zoom: +3.5 / +3.8, vs +1.1 / +2.1 for Pinpoint's top area.
      - 2 zooms: +4.0 / +4.2, equal to zooming all ~6.2 hypotheses.

71. **Merged pipeline, steps 1-2 (2026-10-12; `PINPOINT_SET_PLAN.md`, `hypothesis_agent.py`): the gains stack partly; zoom choice barely
    matters.**
    - Step 1: the area pipeline on the retrained reranker (its pool, zooms and softmax replace Pinpoint's and the set layer):
      area model **+4.9 / +5.4** (train CV / dev3k, agreement pick), vs +4.2 / +5.2 on released Pinpoint. Retrained alone: +3.1 / +3.3.
    - Step 2: one network with an answer head (random zoom subsets in training) and an RL action head:
      - all areas zoomed +4.9 / +5.5; 2 zooms by the agent +4.7 / +5.5;
      - zooming Pinpoint's top-2 areas: +4.7 / +5.2;
      - no zoom at all: +3.9 / +3.8.
    - Most of the gain is area-level choice, not zooming. The policy adds +0.0 to +0.3 over a fixed rule, so an action head needs
      costlier, more varied actions to matter.

72. **Replacing the agreement rule: soft kernels are free, learned versions don't help (2026-10-13; `location_density.py`).** On the
    retrained reranker's ~46 candidates (train CV / dev3k):
    - The fixed rule adds little there: agreement T 0.2 picks 35.0 / 36.9% < 25 km vs the reranker's top-1 35.5 / 36.8%.
    - Hand-made kernels (box, Gaussian 2-50 km, Laplace, multi-scale; NMS at 25 km) are all within ~0.5 at 25 km.
      Small soft kernels keep 1 km: Laplace 10 km, T 0.1 gives 17.4 / 18.4% vs 15.0 / 15.8% for the box rule.
      Right area among the top 5 proposals: ~51 / 53% for every kernel.
    - Learned location distribution: per-candidate weight + Gaussian spread from the reranker's states, mixture NLL with a 10% uniform
      floor.
      - 25 km-mass pick: -0.3 / -0.1 vs the reranker's top-1.
      - Peak pick: +0.4 / +0.1 at 1 km, -0.9 at 25 km.
      - Top-5 coverage: 50.9 / 53.0%, the same as the fixed rule.
    - Training proposals for coverage hurt, using either a soft best-of-5 loss or multiple-choice learning with 5 winner-takes-all
      heads: 48.3 / 50.8% and 47.9 / 50.0% top-5 coverage.
    - The candidate list holds a right answer for 61.6 / 63.4%. The ~10-point gap to the proposals is not recoverable from the reranker's
      states with 35k photos.
    - Decision: swap the box rule for a small soft kernel (elegance, 1 km); look for coverage in new evidence, not in a better grouping.

73. **Merged pipeline on the test set ties the previous pipeline (2026-10-13, the user asked; `PINPOINT_SET_PLAN.md`).** Frozen models
    trained on all of train, eval halves (3,795 photos), % within 25 km vs Pinpoint's 37.2:
    - 2 zooms by the agent, agreement pick: 42.1, **+4.9** [+4.0, +5.9] (im2gps3k +4.3, yfcc4k +5.2); 1 km -0.7.
    - Set pick: +4.1 at 25 km, +1.5 [+0.8, +2.2] at 1 km. Retrained reranker alone: +2.8 / +1.3.
    - Paired vs the previous pipeline (41.8): +0.2 [-0.5, +1.0]. The +0.8 dev3k edge did not transfer, so the merged pipeline is a
      simplification (one network, 2 zooms, no offline set layer), not an accuracy gain.
    - Lesson: gains below ~1 pt between pipelines on dev3k are within its noise for decisions; the next gain has to come from new evidence.
    - Rerun with the gated final pick (lesson 74's default, chosen on train / dev3k; the user asked, 2026-10-13): 2 zooms 42.0% < 25 km,
      +4.8 [+3.9, +5.8] (im2gps3k +4.2, yfcc4k +5.2), 1 km 18.3% (+0.8, was -0.7 with the agreement pick); set pick +4.1 / +1.6.
      The gated rule's dev3k behaviour transferred: same 25 km, the 1 km loss gone.

74. **Final pick, longer lists, reranker training (2026-10-13; `PINPOINT_SET_PLAN.md`).** Train CV / dev3k, % within 25 km vs Pinpoint:
    - Final pick inside a zoomed area: the box founder (a central point) is better at 25 km, a kernel peak (a sharp point) at 1 km. Gating the
      kernel peak to within 5 km of the founder keeps 25 km (+4.7 / +5.3, 2 agent zooms) and lifts 1 km from +0.1 / -1.0 to +1.3 / +0.5. Default.
    - 123-candidate lists raise the in-list ceiling from 63.4% to 72.9% (dev3k) but the reranker's pick (+0.1) and the top-5 areas (+0.4) don't
      find the extra right answers. The gap is ranking, not list length.
    - Reranker ablations (6,000 steps): continued training alone +1.5 / +2.0; + same-photographer exclusion +2.4 / +2.2; wide lists without
      exclusion +0.3 / +0.5 (more chances for the same-photographer shortcut); wide + exclusion + set head +3.1 / +3.3. A full pass
      (16,000 steps) +2.8 / +3.1: validation loss is flat from ~4,500 steps.
    - Training throughput (4 CPU cores): ~400 ms per step, GPU ~65-70% busy, the rest fixed CPU work in the main process. The NVMe index
      gives -8%; chunk size, loader workers and batch size don't fix it without rewriting pinpoint-main's pool union.
    - Lesson: the reranker fine-tune saturates at ~+3. More steps and more candidates don't move it; the next gain needs new training signal
      (cross-fitted labels for the choosing models) or new evidence.

75. **Cross-fitted labels don't help; near-duplicate exclusion has no asymmetry to fix (2026-10-13; `PINPOINT_SET_PLAN.md`).**
    - Near-duplicates (raw SigLIP2 > 0.95 to the closest gallery photo, query photographer excluded) are equally common for dev3k, train and
      Pinpoint-trained photos (3.4-3.7%). Photographer exclusion worked by removing a train / evaluation asymmetry; there is none left.
    - Answer-head learning curve: 33.7 -> 36.9% (held-out train) from 3k to 25k photos, ~+0.5-0.8 per doubling.
    - Cross-fitting by photographer halves (two fine-tuned rerankers, each scoring the other half's 67k trained-on photos): train + 135k
      cross-fitted photos 36.8 / 38.7% vs train alone 37.0 / 38.8% (held-out train / dev3k). The photos are easier than the evaluation
      distribution because Pinpoint's retrieval encoder and the released checkpoint were trained on them: in-list ceiling 69% vs 62-63%,
      top-1 45% vs 36-37%.
    - Lesson: any photo Pinpoint's encoder trained on is out of distribution for training models on top of it. The only in-distribution labels
      are bucket 99 (train + dev3k) unless the encoder is retrained or new geotagged photos are added. More labels is closed as a lever;
      new evidence is next.

76. **A learned read policy gets most of the clue value for a fifth of the reads (2026-10-13; `read_agent.py`, `PINPOINT_SET_PLAN.md`).**
    The 4B clue reader as a costly action in the merged pipeline (before reading: Pinpoint's areas + the photo's SigLIP2 embedding as a free
    glance; reading reveals clue places and clue evidence; 2 zooms; gated pick). Train CV / dev3k, % within 25 km:
    - Never read 36.5 / 37.5%, always read +0.5 / +1.0. Reading helps 0.9 / 1.1% of photos and hurts 0.4 / 0.2%.
    - Learned policy (gain classifier, out of fold; reads when E[gain] > cost): +0.39 / +0.57 at 5% reads, +0.51 / +0.80 at 20%.
    - "Read when Pinpoint is unsure" is no better than random (+0.12 / +0.13 at 20%). Glance only +0.45 / +0.63, Pinpoint features only
      +0.32 / +0.60 at 20%: what is in the photo decides whether reading pays, not retrieval confidence. (Caveat: "unsure" was one feature,
      the normalised support of Pinpoint's best area; a better confidence score such as the reranker's softmax margin is untested.)
    - The gain is small in absolute terms (reading's ceiling is ~1 pt); the policy is a classifier on both known outcomes (a one-step
      bandit), not RL over a sequence.
    - First accuracy-vs-cost result for the tool-using agent; the same recipe extends to costlier tools (geocoders, maps) as more actions.

77. **Map checks of what the 4B sees add nothing beyond retrieval (2026-10-13; `map_checks.py`, `maps_search.py`, `scripts/answer_mapchecks.py`).**
    `features_near` (osm-search's OSM extract, 128M objects as boxes; batch version `maps_search.FeatureIndex` over `features.npz`, 40M
    feature rows, KD-trees by box size, exact against brute force) asked for each feature the 4B reported (observe.py, 67% of photos have one
    the tool knows) near the truth, the right area's restricted top-1 and each wrong area's.
    - The observation is real: within 1 km of the truth, beach 86% when observed vs 16% when not; cliff 56 / 16, harbour 57 / 15.
    - But Pinpoint's wrong areas mostly have it too (they look like the photo): beach 92% at the right area vs 78% at wrong ones; the largest
      gaps are stadium 86 / 59, ruins 65 / 40, castle 63 / 46. Hard elimination (drop areas missing an observed feature) fails the true spot for
      48% (1 km) / 31% (3 km) of photos and nets zero on the targets (top area wrong, a right one exists: helps 1.4%, hurts 1.3%).
    - As soft evidence in the answer head (held-out 10k train / dev3k, 3 seeds, paired): density -0.05 / -0.08, share of observed features
      present -0.04 / -0.19, per-word present / absent columns -0.64 / -0.46 (overfit), all -0.88 / -0.84.
    - Lesson: generic visual features are what retrieval already matches on; a map tool only helps with evidence retrieval can't see, e.g.
      names read in the photo searched inside each candidate area (maps_search near=), next.
    - Engineering: `hypothesis_agent.fit_answers` trains many answer heads as one vmapped model (18 heads in 121 s, GPU 60-66% vs ~19% for one;
      reproduces fit_answer's 37.0 / 38.8-39.0). Don't run parallel training processes next to the osm.db build: a global OOM killed it once.

78. **OSM name search inside areas is redundant with the text-search clue evidence (2026-10-13; `name_checks.py`, `scripts/answer_mapchecks.py`).**
    `maps_search` (osm.db: 128M OSM objects, FTS5 over all name variants; staged on the NVMe drive) asked whether the names a photo shows (4B
    read text + place names, 40% of photos) or evokes (4B recalled landmark queries, 78%) exist near each area's restricted top-1. 137k
    distinct names, one phrase lookup each (~90 ms; the full maps_search query takes 0.5-2 s, 14-18 s for common names with near=), 25%
    found in OSM.
    - Names separate areas far better than visible features (lesson 77): a name within 5 km of 30% of right areas vs 11% of wrong ones
      (train; read 22 / 8%, recall 19 / 4%). But on the targets (top area wrong, a right one exists, 15.6% of photos) a name favours a right
      area for 1.0% of photos and the wrong top area for 0.8%.
    - Answer head, paired at 25 km (held-out 10k train / dev3k, 3 seeds): read names +0.04 / +0.33, recalled -0.07 / +0.09, both -0.09 / -0.02.
    - Redundancy check: masking the existing clue features (text search over GeoNames / Overture / YFCC titles, judged by the 4B) costs
      -0.45 / -0.77; OSM names put back only +0.07 / +0.25 of it. The name evidence is already in the pipeline, in a stronger form.
    - Lesson: names and visible features from the photo, checked against maps, add nothing beyond what retrieval + text search already use;
      the clue reader's information caps these tools at ~1 pt. A new tool has to bring information the photo's text and look don't.
    - Why (`scripts/diagnose_names.py`, a funnel over the baseline answer head's own mistakes; held-out train / dev3k, % of photos): choice
      errors (wrong, a right area exists) 14.0 / 13.9; with a name 12.2 / 12.2; a name found in OSM 7.9 / 7.9; a name within 5 km of a right
      area 2.0 / 2.0; and not of the chosen area 0.7 / 1.0; of those without existing clue evidence 0.4 / 0.8 (the ceiling, matching the
      +0.04 / +0.33). The reverse (baseline right, a name near a wrong area but not the chosen one) is 2.3 / 2.4%, 3x the fixable cases.
      Found names are mostly not this place's: only 24 / 36 / 48% of photos with a found name have a match within 1 / 5 / 25 km of the truth
      (generic words: 'archive', 'departures', 'in memory of'; brands: 'fiat', 'visa'; countries: 'japan', 'españa'; look-alike landmarks).
      Names that do match the right area usually match the chosen one too (cities, chains, regions), and the specific ones were already found
      by text search + the clue judge.
    - As a candidate generator instead (photos where no pipeline area is right: 49.5 / 48.1%, train / dev3k): a found name with <= 20 worldwide
      matches lands within 25 km of the truth on 0.6 / 0.8% of all photos (~9 places proposed per photo with such names); <= 100 matches
      1.2 / 1.4% (~35 places); <= 1000 matches 2.8 / 3.2% (~310 places). Photos currently unreachable, more than the 0.4-0.8% of choice errors
      names could fix; using them needs a visual check among the matches (Pinpoint restricted to the union of a name's match locations).
    - Engineering: `answer_mapchecks.py --variants` takes column groups from map_checks / name_checks and 'noclue' (masks the clue columns).

79. **Names as a candidate generator: a real +1.3 ceiling that can't be selected (2026-10-13; `name_search.py`, `scripts/name_switch.py`).**
    Each photo name with 1-1000 OSM matches (lesson 78's lookups) restricts Pinpoint (retrained reranker setft_a, photographer excluded) to the
    index photos within 2 km of any match (`PinpointReranker.rank_subset` over a GPU-resident index; vs the box search: same top-1 location
    within 1 km on 98% of photos). 46.5k / 4.0k (photo, name) queries on 49% of train / dev3k photos, median ~3,000 allowed photos, ~15 min.
    - Yield: on photos with no right pipeline area (49.5 / 48.1%), some name's top-1 lands < 25 km on 1.8 / 2.1% of them (~1% of all photos;
      the name-match ceiling was 2.8 / 3.2%: Pinpoint's look picks the right match place about a third of the time).
    - Switching the pipeline's answer to the best name candidate helps 1.2 / 1.4% of photos and hurts 12.2 / 12.6%. Oracle switch +1.25 /
      +1.37 at 25 km; learned switch (gain classifier, out of fold; threshold from train CV) +0.03 / -0.03.
    - Separation of helps from hurts (AUC): name-vs-chosen reranker score gap 0.88, learned value 0.88; clue judge P(Yes) 0.58; name
      specificity (match count) 0.54, read vs recalled 0.49. At a 1:10 base rate, AUC 0.88 doesn't reach the >50% precision a switch needs.
    - Lesson: the generator finds new right places, but the only verifier we have is Pinpoint's look, and the photos that need new candidates
      are exactly the ones whose look is ambiguous. Neither name specificity nor the judge says which name candidates are right.
    - Hard constraints at two scales (the user's design, measured on cached data, train / dev3k): (1) country / region codes in the 4B's read
      text: country-code domain 0.7 / 0.6%, phone number 0.5 / 0.5%, postcode-like 0.6 / 0.6%, any 1.6 / 1.6% of photos (TEXT_SEARCH_PLAN
      option 1: truth allowed by the domain 75%, plates 1.5% and no better than Pinpoint). (2) Co-located name pairs (two of the photo's names
      matching within 0.3 km of each other, a verifier independent of the look): 14.2 / 13.7% of photos, median 2 spots, a spot < 25 km of the
      truth 5.0 / 4.8% of all photos; but on photos with no right pipeline area only 0.3 / 0.2%. The constraints are right where retrieval is
      already right (lessons 17-19, 65, 67): on these Flickr benchmarks the photos retrieval fails carry no text to constrain them.

80. **A stronger reader is the bottleneck, mostly through recognition, not text (2026-10-13; the user asked for a test with Claude as the reader;
    `scripts/reader_test_export.py`, `scripts/reader_test_score.py`, readings `fixtures/reader_test_dev3k.json`).** 40 random dev3k photos
    where none of the pipeline's areas is right (< 25 km), at the 4B's pixel budget; read blind (no truth, no 4B output, no Pinpoint) before
    scoring; countries from the nearest GeoNames town (pop >= 5,000), place guesses geocoded automatically.
    - Content: 22 of 40 carry nothing place-specific (X: flowers, pets, portraits, interiors), as lesson 67; explicit place-specific text on 3.
    - Within 25 km: my guesses 4 of 40 (Hamburg's Schanzenpark water tower 0 km, Levoca's Thurzo House 0 km, Manresa's Seu 1 km, a Breton war
      memorial 17 km); the pipeline is wrong on all 40 by construction, the 4B's names and queries reach none (Hermannplatz Berlin, Girona
      cathedral, Vilnius for the Breton memorial).
    - Country: me right on 20 of 24 answered (50% of all), the 4B 11 of 19 (28%), Pinpoint's top-1 19 of 40 (48%). Where Pinpoint's country is
      wrong (21): me right 8, wrong 4, no answer 9; the 4B right 1. Lesson 64: Pinpoint restricted to the true country +7.8, region +27.9.
    - Only 1 of the 4 hits is explicit text (the Latin THURZO inscription with Slovak shop signs; the 4B read it as Romanian). The rest is
      knowledge: recognising a landmark or a regional costume, a Danish Army helicopter type, a Malaysian Sabah plate prefix (country right,
      town 271 km off).
    - So the clues the 4B misses are recognition and regional knowledge more than unread text. One reader (Claude), 40 photos (4 / 40 has a
      95% interval of about 3-24%); the knowledge is a frontier model's, so a 4B reaching it would need training (RL on ground truth), not prompting.
    - Are those queries in the corpus? (`scripts/reader_test_search.py`; text_search with the photo and its photographer excluded, and OSM.)
      Every place name I produced resolves: 'Wasserturm Schanzenpark' 0 km (share 1.00), 'Sternschanze' 0 km (0.90), 'La Seu de Manresa'
      1 km (1.00; OSM 0 km), 'Thurzov dom' 0 km, 'Levoca' 0 km (0.98). The raw text strings mostly don't (book titles, slogans, 'Get Smart'
      17,000 km, the Slovak word 'OBUV', the Latin inscription). The corpus is not the limit: the right name, which comes from recognition,
      is; the 4B produced 'Hermannplatz Berlin', 'Girona cathedral' and 'PANDURIA' for the same photos.

81. **Tool consolidation (2026-10-14, the user: the tools overlapped; landmark matching archived).** The agent's tools are now three, each
    with a text face the agent reads and structured batch calls for training (`src/geo_search_env/tools/`):
    - `pinpoint`: one `search` with per-photo constraints (none, box, country, region, near points, rows), replacing the restricted searches
      of `pinpoint_constrained` (lesson 64), the zoom boxes (70) and `name_search` (79). Self-test on 64 dev3k photos with setft_a: unconstrained
      top-1 = the saved pipeline pool's on 98% (the rest fp16 ties); box (streamed) vs the same rows (resident subset) 98% within 1 km;
      restricted to the truth's country 34% < 25 km vs 30% unconstrained. Wide pools are `PinpointReranker.widen` (was a class inside
      `pinpoint_set.extract`).
    - `maps` (was `experiment/maps_search`): name search with near=, features_near, and now phone / website -> the Overture businesses that
      list it (the places.db tables text_search already had). Overture's website field is noisy (louvre.fr is listed by shops in Mexico and the US).
    - `web`: Wikipedia / Wikivoyage pages plus text_search's Flickr / place clusters for the same words.
    - Archived at tag `pre-tool-consolidation`: landmark matching (lessons 58-63), the GeoNames / Overture POI tools and later clue checks
      (55-67), and the map / name checks and name switch (77-79). Kept: the read chain the pipeline's clue features come from (observe,
      text_search, region_evidence, clue_judge, yfcc_names, vlm_hypotheses, pinpoint_tool).

82. **Gate for query RL: the base 4B sometimes names the right place, rarely beyond Pinpoint's pool, and rarely on the first try (2026-10-14;
    `query_passk.py`, dev3k only).** The 4B (temperature 1, 16 samples) proposes up to 2 place queries or none; each goes through tools.maps
    (top 3) and text_search (top cluster, photo and photographer excluded); a hit = a result < 25 km of the truth. 302 fixable dev3k photos +
    300 random others (control); "novel" = a hit > 25 km from all 5 Pinpoint areas (setft_a pool) shown or not shown to the 4B.
    | prompt | group (top-1 wrong) | queries per sample | pass@16 | novel pass@16 | novel pass@1 |
    |---|---|---|---|---|---|
    | with Pinpoint's top areas | fixable (226) | 62% | 19.5% | 4.4% | |
    | blind (photo only) | fixable (226) | 43% | 12.8% | 8.8% | 2.0% |
    | blind | control (196) | 45% | 3.6% | 2.6% | 0.6% |
    - Shown Pinpoint's areas, the 4B mostly echoes them (Calistoga, Briancon from the prompt); its hits are then choices among the pool, which
      the area model already makes. Blind, it proposes less but finds more places outside the pool. Either prompt: novel 9.7% of fixable /
      4.1% of control photos with a wrong top-1.
    - Hits are recognisable landmarks and districts (Albertina Vienna, Carnegie Mellon, Keukenhof, Lantau Peak, Bermondsey Town Hall) and the
      odd read slogan (Que se lixe a Troika -> Lisbon).
    - Scaled to dev3k: about 94 of 3,000 photos (~3 pts) have a novel hit somewhere in 32 samples where the top-1 is wrong; at pass@1 about
      0.4 pts. So RL has signal above the 1-pt target only if it turns pass@16 into pass@1 and learns to abstain (wrong overrides are not
      counted here). Next: the GRPO run with the Pinpoint-relative reward, blind-style proposals, areas only at the answer step.

83. **Nested proposals (country / region / place) double the ceiling to +6.1 pts, but the 4B overrides wrongly and simple verifiers recover
    +0.1 (2026-10-14; `nested_passk.py`; dev3k, lesson 82's photos).** The base 4B, blind, 16 samples of {country, region, place?} or keep; each
    sample -> one answer: a place whose OSM matches lie in the proposed region (else country) -> Pinpoint near them (5 km); else Pinpoint
    restricted to the region; else the country; else Pinpoint's top-1. Names matched to the Overture labels of lesson 64 (Prešov Region ->
    Region of Prešov etc.). Reference: the retrained reranker's top-1.
    | group | samples: place / region / country / keep | true country / region in 16 (top-1 wrong) | first sample | best of Pinpoint + 16 |
    |---|---|---|---|---|
    | fixable (302) | 14 / 50 / 16 / 20% | 70% / 32% | -12.9 | **+28.1** |
    | control (300) | 15 / 49 / 13 / 22% | 59% / 27% | -20.3 | +3.7 |
    - dev3k-weighted: best of 16 **+6.1 pts** (lesson 82's named places: ~3); a single sample -19.6: the 4B overrides on 80% of samples and is
      usually wrong (region samples on control: 13 fixes, 631 breaks).
    - No-training selector (the most voted answer, override if its vote share and its restricted top-1 score minus Pinpoint's clear
      thresholds): +0.12 held out (2-fold), +0.40 in-sample. Votes and reranker scores don't separate right overrides from wrong ones (as
      lesson 79).
    - So the proposals now hold enough (+6.1); the bottleneck is verification. Next: feed sampled proposals as area tokens to the pipeline's
      trained answer head (it already verifies clue-place areas) and measure; RL on the proposer with that head as the verifier only if it adds.

84. **Retrieval proposes regions far better than the 4B: best-of-16 +19.0 pts vs +6.1; the 4B is confidently wrong where it disagrees with
    Pinpoint (2026-10-14; `scripts/region_candidates.py`, `scripts/proposal_calibration.py`; dev3k, lesson 83's photos, weighted to dev3k).**
    - Calibration of the 4B's majority answer over 16 samples where it differs from Pinpoint's top-1: country precision 16% at >= 25% vote
      share, falling to 4% at >= 75% (Pinpoint's country is right on 66% of photos); region 6-17%. More agreement among samples = more often
      wrong, so the 4B's own confidence can't verify its proposals.
    - Retrieval regions: the 16 most frequent (country, region) labels among the 1,000 nearest index photos, Pinpoint restricted to each.
      True region among them where the top-1 is wrong: 65.8% (4B samples 27.1%). Best of Pinpoint + 16: **+19.0** (fixable +60.9, control
      +14.3); the 4B adds +1.0 on top (union +20.1); retrieval-only fixes +13.9.
    - The pool's own regions (5.6 per photo, TEXT_SEARCH_PLAN option 2) had an oracle below the pool's; a 1,000-neighbour list holds the true
      region far more often. Selection by restricted reranker score gave -0.1 there, so choosing among regions needs a learned verifier.
    - Best-of ceilings grow with the number of draws; +19 is an upper bound, not a forecast. Next: a learned region verifier over these
      candidates (per-region area tokens, neighbour mass, clue features, the 4B scoring a given region), RL on which regions to examine.

85. **Region candidates and tool checks in the answer head: +0.3 at most; the head converts ~2 of the ~15 pts its own areas hold (2026-10-14;
    `region_hyps.py`, `region_checks.py`, `scripts/region_pipeline.sh`, `HYP_VARIANT=regionhyps`; train + dev3k, retrained reranker).**
    - Oracles, dev3k (train): retrained top-1 36.8% (35.5); best restricted top-1 over Pinpoint's ~4.6 areas 51.7% (50.3), + clue places
      52.1 (50.7), + 16 retrieval regions (lesson 84; 34k / 405k kept after dropping those within 25 km of an area) 61.7 (59.7); wide-pool
      oracle 63.4 (61.6).
    - Answer head, all hypotheses examined, % < 25 km dev3k (train CV), vs the released Pinpoint:
      | hypotheses | dev3k | train CV |
      |---|---|---|
      | pipeline (areas + clue places) | 38.7 (+5.3) | 37.3 (+4.9) |
      | + region hypotheses | 39.0 (+5.5) | 37.7 (+5.3) |
      | + region hypotheses + text checks | 38.7 (+5.3) | 37.6 (+5.2) |
      | + region hypotheses + text + maps checks | 38.8 (+5.4) | 37.6 (+5.2) |
      Budget-2 agent: 38.9 / 38.5 / 38.7 / 38.5 (the zoom policy wasn't adapted to region hypotheses).
    - Checks (soft features per hypothesis: read / recalled clue regions and countries from region_evidence, phone / domain country hints,
      tools.maps top-10 matches of read names labelled by area): non-zero on 3-30% of hypotheses; add nothing over region hypotheses.
    - So more candidates and squashed tool evidence don't move the choice. The head sees visual-match statistics and a few clue numbers;
      look-alike areas need evidence of a different kind (lessons 67, 77). Next: the names attached to each area's neighbour photos (YFCC
      titles / tags, `data/yfcc100m/geo.npz`) read by the 4B, the tool-reading agent the user asked for.

86. **The 4B reading each area's neighbour titles / tags picks near chance; the titles name the area, not the query's match (2026-10-14;
    `tag_reader.py`, dev3k, the user asked for the tool-reading test).** Per photo Pinpoint's ~4.6 areas (restricted searches, photographer
    excluded) as letters A.. (A = Pinpoint's top area), each with its nearest town and up to 8 YFCC titles + tags of its restricted candidates
    (`data/yfcc100m/geo.npz`; all 142k found, 7.9 per area); the 4B answers one letter, P(letter) from the first-token log-probabilities.
    - % < 25 km: area A 36.2%, best area 51.5%, a random area 17.5%, the answer head 38.7%.
    - 4B pick: labels only 18.7%, labels + titles 21.7% (picks another area than A on 77-79%, letter D most often: position bias). Switching
      only at P >= 0.9: 36.1% (6 fixes, 7 breaks). Held-out (2-fold) mix of the rank prior and log P: 32.5% vs 32.4% for the prior alone.
    - Signal: AUC of P for right vs wrong areas 0.61 with titles, 0.56 without. Titles name their own area's town on 41% of right and 38% of
      wrong areas, and 19% are camera file names or one or two words: they say where the neighbours are (known from coordinates), rarely
      what links the query photo to one area.
    - With lessons 77-79, 83-85: none of our tool outputs (map features, names in areas, region checks, neighbour titles) separates the look-
      alike areas Pinpoint confuses. The remaining gap (head 38.7% vs best area 51.5%) is visual near-misses and generic photos (lesson 67).

87. **GRPO on the 4B's clue output, pilot: it learns silence and caution, not precision (2026-10-14; `clue_grpo.py`; the user's design:
    say nothing on photos with nothing place-specific).** The 4B sees the photo and Pinpoint's top-1 region and answers {"evidence": "none"},
    a kept region, or an overriding country / region; reward from labels only (right override where Pinpoint's region is wrong +1, wrong
    override -1 where Pinpoint was right, -0.5 where both wrong, none / keep 0). LoRA r 32 on the base Qwen3.5-4B, TRL GRPO with colocated vLLM,
    8 samples, lr 5e-6, beta 0.04; 287 steps in 2.5 h (~30 s per step, ~2,300 train photos). Dev3k, greedy, overrides -> Pinpoint restricted
    to the region (tools.pinpoint):
    | | none | keep | override (changes the region) | region right | % < 25 km vs retrained top-1 | silent on lesson-67 X photos |
    |---|---|---|---|---|---|---|
    | zero-shot | 27.8% | 54.4% | 15.6% (5.9%) | 23.2% | -1.13 (17 fixes / 51 breaks) | 45% (override 5%) |
    | pilot | 39.9% | 43.5% | 16.4% (2.4%) | 23.6% | -0.37 (6 / 17) | 55% (override 0%) |
    - Training reward -0.062 -> -0.033 (48-step means), overrides 21% -> 18%, entropy stable (0.96 -> 0.90).
    - It learned to stay out: more "none", fewer region changes (many "overrides" name Pinpoint's own region, a safe 0), no overrides on
      empty photos; precision of real overrides did not move (23-24%, break-even ~50%).
    - Per the user, the full train set runs regardless (`clue-grpo-full`, from the pilot's adapter; ~36 h at this speed, resumed after
      the daily shutdown with --resume).

88. **A stronger open reader (Qwen3.8-27B-FP8) produces fewer right place names and countries than the 4B (2026-10-14; `reader_compare.py`,
    `scripts/reader_compare.sh`; dev3k, the user's 1-hour test).** observe.py's clue prompt (v2, country guess) and search-query prompt, same
    786k-px images, thinking off; every phrase (read text, place names, queries) resolved with text_search (top cluster, photo and photographer
    excluded; maps skipped for time). % of photos with some phrase within 25 km of the truth, 4B vs 27B:
    | photos | n | 4B | Qwen3.8-27B |
    |---|---|---|---|
    | all | 3,000 | 17.4 (phrases on 81%) | 12.8 (49%) |
    | retrained top-1 wrong | 1,897 | 5.1 | 3.6 |
    | no Pinpoint area right | 1,454 | 3.2 | 2.1 |
    - Country guess right 40.2 vs 34.0% (Pinpoint 69.2%); where it differs from Pinpoint's country, right 13.4% (91 fixes / 412 breaks) vs
      10.0% (96 / 637). The 27B guesses confidently on generic photos (Peru for Tanzania, Iceland for Sweden) and emits phrases on fewer photos.
    - With lessons 56 and 59 (9B, Qwen3.6-27B): open-model size doesn't buy the recognition Claude showed in lesson 80; a bigger reader is not
      the fix for the tools. Landmark prompt also run (landmark named on 423 photos) but not scored (the landmark line is closed).

89. **Looking at parts of the photo adds nothing: crop views compared with each area's photos, +0.0 / +0.4 in the answer head (2026-10-14;
    `crop_look.py`; the user's no-LLM RL line, step 1 = ceiling with all views and all areas).** 10 SigLIP2-giant views per photo (full image
    + a 3 x 3 grid of half-size windows, stride a quarter; the full view matches the cached MP16 embedding, cosine median 0.9999), 389k photos
    of views in ~1.5 h. Per hypothesis: each view's max / top-3 cosine to the area's 12 restricted-search photos (photographer excluded), the
    margin over the photo's other areas, softmax share, argmax fraction; groups "global" (4 columns) and "crop" (9).
    - Answer head (`hypothesis_agent --extra`), % < 25 km train CV / dev3k, all areas zoomed: baseline 37.3 / 38.7; + global 37.4 / 38.9;
      + global + crop 37.3 / 39.1. Budget-2 agent 37.1 / 38.9 -> 37.2 / 39.1. Below the 1-pt go line; no RL look policy built.
    - dev3k, 1,416 photos with a right and a wrong area: picking the area each column prefers is right 40-50% vs 65.7% for the top area;
      AUC right vs wrong areas 0.66 (crop margin) vs 0.68 (full view); a conditional logit over rank features is 36.9% with or without them.
    - Reading: a crop matched against whole index photos says what the full view already says (the area's photos look like the scene);
      detail the 384-px embedding loses doesn't come back by comparing crops with other photos' global embeddings.

90. **Round-trip verification fails the gate: the reverse rank separates right from wrong areas no better than forward similarity (2026-10-14;
    `round_trip.py`, dev3k).** For each of the 12 restricted-search photos r of each hypothesis: how many of the 4.08M index photos r is closer
    to than to the query, in Pinpoint's contrastive retrieval space (query's photographer excluded; a variant also drops r's photographer).
    152,606 pairs in ~3 min on the GPU; reverse rank median 772, < 10 on 5.9% of pairs.
    - 1,416 dev3k photos with a right and a wrong hypothesis: AUC right vs wrong areas, reverse rank (min / median / # photos with the query in
      their top 10 / 100) 0.647-0.665 vs forward similarity 0.655; argmax picks 51-59% vs 65.7% for the top area. Dropping r's photographer
      changes nothing.
    - Conditional logit over rank / source / support (2-fold x 3 seeds): 36.9%; + forward similarity 36.8; + reverse ranks 37.0. No answer-head
      run or RL policy (go line was a clear AUC gain first).
    - Reading: the true area's photos are mostly not same-spot matches either (only 5.9% of pairs have the query in their top 10); on these
      Flickr photos the right area is "the same kind of scene nearby", as generic from its side as the look-alikes are. Neighbourhood
      structure in the index carries the same information as similarity.

91. **Refining to the next bucket has large headroom but no usable signal: a local chooser nets +-0.1 (2026-10-14; `refine_headroom.py`,
    `refine_gate.py`; dev3k; the user's idea: move answers from < 25 to < 1 km, < 200 to < 25 km).** Pinpoint (retrained reranker) restricted to
    R km around its own top-1, photographer excluded.
    - Headroom per current bucket (share of dev3k): 1-25 km (18.9%) has a candidate < 1 km in the R = 25 list 62.8% (oracle +11.9 pts at 1 km);
      25-200 km (13.6%) one < 25 km in the R = 200 list 82.8% (+11.2 at 25 km); 200-750 km (18.7%) one < 200 km 93.1% (+17.4). The restricted
      search's own top-1 is the old answer again (lands in the next bucket 3-4%).
    - All photos: R = 25 list oracle at 1 km 29.9% vs 17.9% kept; R = 200 at 25 km 48.3% vs 36.8%.
    - Chooser over the list (2-fold x 3 seeds, net change at 1 / 25 / 200 / 750 / 2500 km vs keeping the top-1): reranker score / rank /
      retrieval features, + agreement kernels (0.5 / 1 / 5 km), + forward cosine and reverse rank (lesson 90), + best-crop cosine (lesson 89);
      linear or MLP; target-bucket likelihood or expected thresholds met (exact policy gradient over the known outcomes). Best +0.1 at 1 km
      (R = 25); R = 200 -0.2 to -0.6 at 25 km. Target-likelihood choosers move to other in-bucket candidates and lose the tighter bucket
      (-1.6 to -5.5 at 1 km).
    - Why: on the photos a move could fix (12% each), every feature separates right from wrong candidates in the list at chance (within-list
      AUC 0.44-0.58; only "distance from the old answer" 0.64 at R = 200, true by construction once the answer is wrong). The right local
      candidates are other scenes of the same area, not matches of this one.
    - With lessons 85-90: once retrieval has proposed candidates, nothing we can compute from the embeddings (global, crops, neighbourhood
      structure) or the 4B / tools separates the right ones, at any scale.

92. **Word-location statistics from YFCC for all read text add +0.1 over the existing clue features (2026-10-14; `text_geo.py`, dev3k; the user
    chose to build on the clue-reading success).** Every word the 4B reads (36% of dev3k photos, 4,120 distinct words) as soft evidence: from
    the 48M YFCC titles + tags (yfcc.db FTS5), distinct (photographer, 0.25-degree cell) pairs per word (2.0M background pairs); per area a
    naive-Bayes log ratio sum_w log P(w | area) / P(w) at 25 / 200 km, smoothed (alpha 20), clipped per word; the query's photographer removed
    from every count. ~4 min per build.
    - Per-area separation on photos with read words and a right + wrong area: margin of the 25 km log ratio over the photo's other areas AUC
      0.678 (the existing read-clue columns 0.591).
    - Conditional logit, % < 25 km (2-fold x 3 seeds): Pinpoint features 37.2; + existing clue features 37.7; + text statistics 37.4; + both
      37.8. Same with the 9B + crops reader (49% of photos with text): 37.7 vs 37.7. Specific words only (<= 2,000 pairs, positive evidence):
      37.8. Gate failed (go was a clear gain beyond the clue features); no answer-head run.
    - Why: on photos where the top area is wrong and a right one exists (189), the right area has the higher text score in 43%; but where
      the top area is right (379, twice as many), a wrong area outscores it in 42%. Most read words (watermarks, brands, "cinema", English
      phrases) are spread over many areas; language narrows to a country Pinpoint mostly has right already. The names among them are what
      the existing clue chain already uses.

93. **Same-place recognition (MegaLoc) is the first new signal in a while: +0.8 in the simple chooser, on top of the clue features too
    (2026-10-14; `megaloc_gate.py`, dev3k gate; the user's question: better recognition without a frontier LLM).** MegaLoc (gberton/MegaLoc,
    CVPR-W 2025; DINOv2 ViT-B/14 + optimal-transport aggregation, 8,448-d, trained for visual place recognition), float32 at 322 px; the 3,000
    queries + 142,697 area photos (the zoom file's 12 restricted-search photos per hypothesis) in ~13 min (~190 photos/s).
    - Per-area best / top-3 MegaLoc cosine and its margin over the photo's other areas. AUC right vs wrong areas (1,416 photos with both):
      margin 0.71-0.72 (SigLIP2 forward similarity 0.655-0.68, lessons 89-90; YFCC word statistics 0.68); where the top area is wrong (485)
      0.60-0.61.
    - Right area beats the wrong top area by > 0.05: 120 / 485; a wrong area beats the right top area: 163 / 938 (a better ratio than any
      earlier signal).
    - Conditional logit, % < 25 km (2-fold x 3 seeds): Pinpoint features 37.2; + clue features 37.7; + MegaLoc 38.0; + both 38.5. So MegaLoc
      adds +0.8 independently of the clue features, where crops / round trips / word statistics added 0.0-0.2.
    - Answer head (`hypothesis_agent --extra megaloc`: best / top-3 cosine and their margins as pre-zoom columns; train: 1.18M area photos,
      streamed and resumable, ~1.5 h incl. a restart), % < 25 km vs the same head without them (crop_heads baseline, same seeds):
      | | train CV | dev3k |
      |---|---|---|
      | budget 0 (no zoom) | 36.3 -> 37.1 (+0.8) | 37.3 -> 38.3 (+1.0) |
      | budget 2, agent | 37.1 -> 38.0 (+0.9) | 38.9 -> 39.4 (+0.5) |
      | budget 2, top areas | 37.0 -> 38.0 (+1.0) | 38.5 -> 39.6 (+1.1) |
      | all areas zoomed | 37.3 -> 38.2 (+0.9) | 38.7 -> 39.3 (+0.6) |
      1 km: train CV +0.5 to +0.6, dev3k +0.2 to +0.5. vs the released Pinpoint: dev3k +5.8 to +6.1 (was +5.3 to +5.4).
    - The largest evidence gain since the zoom pipeline (lesson 70) and consistent: +0.8 to +1.0 on train CV (35k photos), +0.5 to +1.1 on
      dev3k (noisier). At the 1-pt line on train, just under it on dev3k for the default (budget-2 agent) configuration.
    - Caveat: the columns use every area's restricted-search photos, so in the budgeted rows they leak post-zoom information; as an agent
      action, MegaLoc should be its own costly look (RL deciding when).

94. **An RL action policy over zoom / read / recognise (MegaLoc) traces an accuracy-vs-cost curve at or above every fixed tool combination;
    best pipeline 39.7% on dev3k (+6.3 over Pinpoint) (2026-10-15; `action_agent.py`; the user: fit text clues and MegaLoc in as actions,
    train the policy with RL).** Area tokens as hypothesis_agent; actions zoom area k (1 unit), read (2: clue-place tokens + clue columns of
    every area), recognise (1: MegaLoc columns of every area; hidden until taken, which removes lesson 93's free-information caveat), stop.
    Answer head supervised on random reveal patterns; policy (per-area zoom logits + read / recognise / stop from the frozen encoding)
    trained with REINFORCE, greedy-rollout baseline, entropy 0.01; return = [< 25 km] - lambda x units. Trained on all of train (dev3k) or
    four fifths (held-out fifth).
    - Fixed combinations, dev3k (held-out fifth) % < 25 km at cost: nothing 36.6 (35.6) at 0; recognise 37.7 (36.5) at 1; read 37.2 (36.3)
      at 2; zoom top 2 38.0 (36.0) at 1.9; recognise + zoom 2 38.7 (37.0) at 2.9; read + recognise + zoom 2 39.5 (37.6) at 4.9; all 39.4
      (38.0) at 7.6. Per unit, recognise is the most efficient action (+1.1 for 1 unit), then zoom (+1.4 for 2), then read (+0.6 for 2).
    - Policy on area features only: lambda 0 = everything (39.5); lambda 0.005 38.7 at 2.2 units (= recognise + zoom 2 at 2.9); lambda >= 0.01
      collapses to no actions (36.6): each action is worth ~0.01 on average, so mid costs need knowing which photos benefit.
    - With the photo's SigLIP2 embedding as a free glance (read_agent's finding): dev3k (held-out fifth)
      | lambda | < 25 km | cost | zooms | read | recognise |
      |---|---|---|---|---|---|
      | 0 | 39.7 (38.1) | 6.6 (6.7) | 4.4 | 77% | 69% |
      | 0.0025 | 39.2 (38.0) | 3.7 (4.9) | 2.2 | 39% | 64% |
      | 0.005 | 39.0 (36.8) | 2.9 (1.0) | 1.5 | 36% | 65% |
      | 0.0075-0.015 | 37.5-37.7 (35.9-36.7) | 0.2-0.5 | ~0.1 | 0-10% | 6-19% |
      No collapse; at each cost the policy matches or beats the fixed combination of the same cost (dev3k: 39.0 vs 38.7 at 2.9; 37.7 at
      0.2-0.45 units = recognise-for-everyone at 1.0; held-out fifth: 38.0 vs 37.6 at ~4.9, 36.7 at 0.46 vs 35.6 for nothing). Margins of
      0.3-0.4 are within noise; the robust result is the low-cost end (MegaLoc's gain with 12-19% of photos recognised).
    - 1 km: the policy keeps 17.8-18.4% (fixed 17.2-18.7).

95. **MegaLoc as a proposer fails: its same-place search finds right new places on ~1% of photos; it is a verifier, not a generator
    (2026-10-15; `megaloc_deep.py`, dev3k; step 1 of the user's "proposer, then free-text queries").** Per query the 200 nearest index photos in
    raw SigLIP2 space (photographer excluded; 517k distinct photos), re-ranked by MegaLoc (~35 min).
    - < 25 km: retrained top-1 36.8%; any Pinpoint area 51.5%; any of the 200 67.9%; top-1 of the shortlist by SigLIP2 21.9%, by MegaLoc 22.4%
      (< 1 km: any 37.3%, MegaLoc top-1 13.1% vs retrained 17.9%).
    - Photos where no Pinpoint area is right (1,454, 48.5%): the shortlist holds a right photo for 42.8%, but MegaLoc's top match is right for
      2.1% (SigLIP2's 1.3%).
    - MegaLoc's top match as a new area (> 25 km from every Pinpoint area): 2,006 photos, right 1.4% (28 fixes, 0.9% of photos); higher
      similarity doesn't help (>= 0.3: 1.6%; >= 0.5: 0 of 52).
    - Reading: the right photos in the shortlist are nearby scenes, not the same place, so a same-place model has nothing to latch onto; where
      the same place is in the index, Pinpoint's areas already hold it (lesson 93's verifier gain). The full-index MegaLoc build is not worth it.

96. **GRPO on free-text search queries with the tool's result as reward learns precision by silence, not new places (2026-10-15;
    `query_grpo.py`; step 2 of the user's "proposer, then free-text queries").** The 4B sees the photo only (lesson 82's blind prompt) and
    writes up to 2 queries or none; each goes through text_search (top cluster, photo and photographer excluded). Reward vs the truth: a query
    < 25 km of the truth and > 25 km from every Pinpoint area +1, < 25 km inside an area +0.2, wrong -0.1, unresolved -0.05, none 0. GRPO,
    LoRA r32, 8 samples, 8 photos / step, ~28 s / step; stopped at step ~130 of a 2.5 h pilot.
    - Training: samples with no queries 57% -> 80-88% within 40 steps; new-place rate flat at ~0.1% of samples (about one per 16 steps).
    - dev3k, greedy, base -> step 100: queries on 46.1% -> 8.4% of photos; precision (< 25 km) 26.0% -> 63.6%; unresolvable queries
      23.6% -> 2.3% of photos; new right places 0.60% (18) -> 0.27% (8).
    - Reading: the new-place reward is too rare to give a gradient (lesson 82: pass@1 ~2% on fixable photos); the penalties dominate, so the
      policy keeps only the safe queries, which confirm Pinpoint's areas. Same failure as lesson 87. Queries as an RL action would need a
      denser proposer signal than the open-model 4B's recognition provides.
