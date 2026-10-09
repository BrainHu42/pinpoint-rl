# Geolocation as hypothesis testing (plan, agreed 2026-10-09)

**One-line idea.** Don't pick the single most similar photo. Propose a few *area hypotheses*, gather evidence *inside each* with retrieval
restricted to that area, and commit to the hypothesis the evidence supports best. That's the professional's loop (hypothesise, search
inside each hypothesis, compare, commit), with visual retrieval as the evidence-gathering tool.

## Why (our evidence)
| Finding | Number | Lesson |
|---|---|---|
| Pinpoint scores candidates independently and loses set-level evidence; a fixed agreement rule recovers some | **+1.0 pts** < 25 km on 35k train photos, +2.0 on dev3k | 68 |
| Retrieval inside the right area is very strong | dev3k 33.4% -> **61.3%** < 25 km with Pinpoint restricted to the true region (pool oracle 50.3%) | 64 |
| About half the photos carry nothing place-specific; the reachable gap is photos with a recognisable area but no spot | ceiling ~42-48% < 25 km from content + today's retrieval | 67 |
| Explicit clues are rare but precise when judged by eye | read-text clue judge +0.6-0.7 | 65-66 |
| Choosing among pool regions by the *best single score* of a restricted search doesn't work | -0.1 | 64 follow-up |

The method has to explain the gap between the second row (61%: retrieval given the right area) and Pinpoint (33%). It does that by
finding and verifying the right area. The last row says the comparison has to use evidence *mass*, not one best match.

## The system
For a query photo x:

**1. Hypotheses H(x) (K <= 5 areas).** Each hypothesis h is an area: a 25 km disc around a point, or an admin region. Sources:
- **Pinpoint areas:** the agreement clusters of Pinpoint's pool (`pinpoint_tool.areas`), top-K by support.
- **Clue hypotheses:** places from accepted text clues (`text_search` + the fine-tuned judge).
- **Later:** VLM-proposed regions (country / region guesses, landmark recalls), only as extra hypotheses to test, never as answers.
- **Default:** Pinpoint's top-1 is always a hypothesis, so the system can't do worse than "keep Pinpoint" by construction.

**2. Retrieval inside each hypothesis R(x, h).** Pinpoint's retrieval + reranker restricted to h (`models/pinpoint_reranker.py`, `query_box`
/ `query_region`, built). It returns up to M = 12 candidates inside h with reranker scores and the 9 retrieval features.

**3. Evidence features phi(x, h), per hypothesis.**
- **Restricted-retrieval strength:** top-1 score, mean of the top 3, margin to Pinpoint's unrestricted top-1.
- **Agreement mass:** softmax score mass of the densest 25 km cluster inside h; share of the unrestricted pool's mass inside h.
- **Independence:** distinct photographers among the matches inside h. Agreement from one photographer's photo series is weak.
- **Popularity correction:** log of the number of gallery photos in h. Agreement partly acts as a popularity prior and the breaks come
  from crowded places (lesson 68), so the comparator gets the area's photo density to discount it.
- **Clue support:** accepted clues pointing into h and their judge P(Yes).
- **Source:** Pinpoint area / clue / VLM.

**4. Comparator C (the learned part).** A small set model (2-layer transformer, or a listwise MLP) over {phi(x, h) : h in H(x)}, scoring
hypotheses jointly.
- **Labels:** GPS only. The target hypothesis is the one whose restricted best candidate is < 25 km from the truth (none -> keep Pinpoint).
- **Loss:** listwise softmax over hypotheses + a "none" option.
- **Data:** train (35,359 bucket-99 photos), optionally + the 35k extra photos from buckets 0-98 (lesson 66 data; Pinpoint is
  optimistic there, flag it).
- **Answer:** the chosen hypothesis's best restricted candidate, or its agreement pick inside h.

**5. Agent version (stage 2, keeps the RL framing).** A Qwen3.5-4B policy with tools `pinpoint_areas()`, `pinpoint_search(area)`,
`text_search(q)` and `compare(hypotheses)` (= C).
- **What RL adds:** it decides which hypotheses to propose and test (e.g. from text it reads or a landmark it suspects) and when to
  commit.
- **Reward:** distance (`geo_reward`), plus a small cost per restricted search.
- **When:** only after stage 1 shows that hypothesis testing beats agreement.

**Ablation:** a set-aware Pinpoint reranker (Pinpoint retrained with attention across candidates, 2-3 h) as an alternative comparator,
and as the "is it just a better reranker?" control.

## Evaluation
- **Development:** dev3k (choose and tune); train for setting selection and comparator training. **Test** (im2gps3k + yfcc4k eval halves):
  once, frozen, when the user asks.
- **Metrics:** % within 1 / 25 / 200 / 750 / 2500 km; fixes / breaks vs the reference; bootstrap intervals.
- **References:** Pinpoint top-1 (published baseline); **Pinpoint + agreement (development reference, lesson 68)**; Im2GPS-style kNN
  mean-shift over Pinpoint's neighbours; oracles (pool oracle; hypothesis oracle = best of the K hypotheses; true-region restriction).
  Optionally an LMM-over-candidates baseline (Img2Loc / GeoReasoner style) for the paper.

## Experiments, in order (each with a go / no-go)
| # | Experiment | Cost | Go if |
|---|---|---|---|
| E0 | Literature check: retrieval-augmented / agentic geolocation, set / distance-aware reranking (GeoRanker), hypothesis-conditioned retrieval | 30 min | the hypothesis-testing framing isn't already published as such |
| E1 | **Hypothesis oracle**: K = 1..5 Pinpoint areas (+ clue places); how often does H(x) hold the truth, and what if the right one is picked (its restricted best)? | 1 h GPU (restricted search for ~5 areas x 38k photos) | oracle >= ~45% < 25 km on dev3k (Pinpoint 33.4%, agreement 35.4%) |
| E2 | **Hand comparator**: agreement mass inside the restricted results, with and without popularity correction | 1 h CPU | beats agreement-at-25-km on train (> +1.0) |
| E3 | **Learned comparator** on train, evaluated on dev3k (+ the set-aware Pinpoint ablation) | 3-5 h | >= +1 pt beyond Pinpoint + agreement on dev3k, and positive on a train hold-out |
| E4 | Add clue hypotheses and clue features (text search + judge) | 2 h | adds on top of E3 |
| E5 | Agent (stage 2): 4B policy proposes and tests hypotheses, GRPO with distance reward | 1-2 days | beats E4 on dev3k; base-4B best-of-8 headroom first (lesson 3) |
| E6 | Frozen system on the test set, once | 1 h | user asks |

## Paper outline (if E1-E4 hold)
1. **Motivation:** the locatability ceiling (lesson 67): half the photos are unlocatable from content; the reachable gap is "area known,
   spot unknown". Retrieval inside the right area reaches 61% (lesson 64).
2. **Finding:** independent candidate scoring loses set-level evidence; agreement recovers +1 pt (lesson 68), echoing Im2GPS's kNN density
   over 1-NN.
3. **Method:** hypothesis generation, hypothesis-conditioned retrieval, evidence comparison (popularity-corrected), clue hypotheses;
   the agent version.
4. **Results:** vs Pinpoint, Pinpoint + agreement, kNN mean-shift, oracles; ablations (no restriction, no popularity correction, no clues,
   K, radius, set-aware reranker); breakdown by locatability tier.

## Risks
- **Hypothesis recall:** if the right area isn't among the K hypotheses, nothing downstream helps (E1 measures it).
- **Popularity bias:** crowded areas produce strong restricted matches for anything; the correction feature must work (E2 ablation).
- **Overlap with a better reranker:** the set-aware Pinpoint ablation tells whether the gain is "hypothesis testing" or "joint scoring".
- **Cost:** restricted retrieval per hypothesis is one index pass per batch of (photo, hypothesis) pairs. Fine offline; for the agent,
  cache or use a box index.
- **Novelty:** depends on E0.

## E0: literature check (2026-10-09; arXiv abstracts and pages, some papers only partly read, Semantic Scholar rate-limited)
**Already done (cite and differentiate):**
- **Density of retrieved neighbours as evidence:** Im2GPS (Hays & Efros 2008, mean-shift over kNN); Revisiting IM2GPS (Vo, Jacobs &
  Hays, ICCV 2017, kernel density over kNN). Lesson 68's agreement is this idea applied to a learned reranker's pool.
- **Retrieval restricted to predicted areas:**
  - Kordopatis-Zilos et al. (ICMR 2021) search within one predicted cell and cluster spatially.
  - **PIGEON / PIGEOTTO** (CVPR 2024) take the top-K geocells, cluster training locations inside each, pick the best cluster per cell
    by embedding distance and combine with a fixed product of softmax x geocell probability. This is the closest prior work to steps
    1-3, but the comparison is fixed, and there is no photographer, popularity or clue evidence.
- **Retrieval-augmented VLMs scoring candidates one at a time:**
  - Img2Loc (SIGIR 2024) and G3 (NeurIPS 2024).
  - **GeoRanker** (arXiv 2505.13731, 2025): VLM value head, one (query, candidate) forward pass per candidate, listwise loss in
    training but pointwise at inference, no agreement across candidates.
- **RL / tool agents that verify hypotheses:**
  - **Thinking with Map** (arXiv 2601.05432, 2026): GRPO, map / POI tools, parallel trajectories + verifier.
  - **GeoVista** (2511.15705): zoom + web search, SFT + RL.
  - **LocationAgent** (2601.19155): explicit hypothesis-verification cycles, zero-shot.
  - GEO-Detective, GeoSearch, GeoBrowse, GeoAgent: web or Street View tools.
  - None of them uses region-restricted retrieval over a geotagged gallery.

**What looks novel (no prior work found):**
1. A **learned set-level comparator across area hypotheses** using evidence from restricted retrieval: match mass, distinct
   photographers, popularity correction, clue support.
2. Text-read and recalled hypotheses tested by restricted image retrieval.
3. An **RL agent whose tool is region-restricted gallery retrieval**, choosing which hypotheses to test and when to commit.
4. The diagnosis that strong pointwise rerankers (Pinpoint, GeoRanker) discard set-level evidence, shown on top of a learned reranker.

**Implications for the paper:**
- PIGEON is the baseline the comparator must beat. Reproduce its fixed combination on our hypotheses: per hypothesis, the best
  restricted cluster's similarity x the hypothesis prior.
- The novelty claim rests on the learned evidence comparison (E2-E3) and the agent (E5), not on restricted search itself.
- The hypothesis-testing framing must be distinguished from LocationAgent (theirs: web and map clues, zero-shot; ours: gallery
  retrieval inside hypotheses, trained).
- Next comparator baselines: PIGEON-style product, Vo et al. KDE, Pinpoint + agreement.

## E1 / E2 results (2026-10-09; `hypotheses.py`, `hypothesis_compare.py`)
- **E1, hypothesis oracle** (pick the right one of the top-K Pinpoint areas; its restricted top-1):

| K | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| dev3k | 34.9% | 41.3% | 44.6% | 46.5% | 48.0% (48.5% + clue places) |
| train | 33.1% | 39.5% | 42.7% | 44.7% | 46.0% |

  Pinpoint 33.4 / 32.4%; pool oracle 50.3 / 49.2%. "Any restricted candidate" at K = 5: 51.6 / 49.6%. **Go** (>= 45% on dev3k).
- **E2, comparators** (K = 5; answer = the chosen hypothesis's restricted top-1; change vs Pinpoint's top-1 < 25 km):

| Comparator | train (35,359) | dev3k (3,000) |
|---|---|---|
| Pinpoint + agreement on its pool (lesson 68, the reference) | +1.0 [+0.7, +1.2] | +2.0 [+1.2, +2.8] |
| area 1 (best pool support), restricted top-1 | +0.7 | +1.5 |
| max restricted top-1 score | -0.5 | +0.1 |
| restricted score mass | -0.1 | +0.5 |
| PIGEON-style: pool support x softmax(restricted top-1 / 0.2) | +0.8 [+0.6, +1.0] | +1.5 |
| Vo et al. KDE over the top-100 neighbours alone | -1.0 [-1.3, -0.6] | -0.1 |
| **restricted mass (T 0.2) x support^0.5 x distinct photographers** (parameters chosen on train) | **+1.6** [+1.3, +1.9] (1,315 fixes / 754 breaks) | **+2.2** [+1.3, +3.1] (129 / 64) |
| same without photographers | +0.8 | +1.9 |
| same without support | +0.9 | +1.0 |

- **Reading:**
  - **E2 passes on train:** the hand comparator beats Pinpoint + agreement by +0.6 pts on 35k photos (+1.6 vs +1.0). On dev3k it's
    +2.2 vs +2.0.
  - **Distinct photographers is the key evidence:** +0.8 on train. Matches from many different people inside an area are
    independent votes; a photo series by one person isn't. This feature is new for geolocation (E0).
  - **Gallery density (the popularity correction) wasn't selected.**
  - **Prior work is weaker on our hypotheses:** the PIGEON-style fixed product gets +0.8 and kNN density -1.0.
  - The oracle (46-48%) leaves a lot for a learned comparator (E3).
