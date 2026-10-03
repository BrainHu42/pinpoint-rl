# Learnings so far (Sep 2026)

What the base-model diagnostic, SFT, single-turn GRPO and the follow-up headroom checks established, and what they
rule out. The current plan lives in CLAUDE.md ("Research plan"). Numbers are % of
photos within 1 km / 25 km of the truth, with same-photographer gallery images excluded. "Full eval halves" = every
im2gps3k + yfcc4k eval-half query (n = 3,795; SE about 0.6 pts at 1 km, 0.8 at 25 km). "300 subset" = the stratified
study subset (SE about 2 / 3 pts; don't trust gaps under ~4 pts there).

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

## Data and leakage rules we established
- Pinpoint's retriever trained on MP16 md5(image_id) % 100 < 99; its photos get inflated candidates (Pinpoint top-1
  <25 km 40.6% vs 30.4% held out). Train only on the bucket-99 pool (38k; 34.5k train / 3.8k val split by
  photographer); its candidate quality matches yfcc4k eval.
- yfcc4k shares photographers with MP16: always exclude same-photographer gallery rows (`tests/test_author_exclusion.py`).
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
- `/data/pinpoint/sft/sft-34k-retrieval` (adapter): best SFT model, starting point for the agent.
- `/data/pinpoint/sft/grpo-kl` (checkpoint-200): best greedy so far.
- Merged weights were deleted to save space (2026-10-02). Rebuild in order: `sft_train merge --run sft-34k-retrieval`,
  then `grpo_train merge --init sft-34k-retrieval --run grpo-kl --adapter checkpoint-200` (GRPO's base is the SFT merge).
- `artifacts/sft/`: MP16 query pool, neighbour/candidate caches, `sft_retrieval.jsonl` (+ `sft.jsonl`), logs.
- `artifacts/strategy_search/`: benchmark caches, all result JSONs, Gemini labels for 1,683 eval photos
  (`llm_advantage_labels.json`; OpenRouter credits ran out before the other ~2,100), crop-query search cache
  (`query_headroom_crops.npz`), evidence-test answers (`evidence_test_vlm_*`).
- `/data/pinpoint/geonames/allCountries.txt` (GeoNames dump; `evidence_test landmarks` rebuilds the landmark index).
- Environments: `.venv` (analysis), `~/.venvs/sft` (SFT), `~/.venvs/grpo` (GRPO with vLLM), `~/.venvs/vllm` (serving).
