# Lessons

New lessons, one entry each (2-4 lines: the claim, the deciding numbers, the module). Lessons 1-137 are archived in
`docs/archive/LESSONS_1-137.md` (anchor `#lN` per lesson); their main points are in `LEARNINGS.md`. When a new lesson changes a main point,
update `LEARNINGS.md` too.

The two tracks run in parallel sessions (2026-10-20), so each numbers its own lessons and appends only to its own section: R1, R2, ...
for `RETRIEVAL_PLAN.md`, S1, S2, ... for `SEARCH_PLAN.md`, V1, V2, ... for `VALUE_PLAN.md`. Later single-track work continues from 138.

## Retrieval track (R)

## Search-agent track (S)

S1. **Search-tree ceiling (step 1): +14.0 pts beyond the area oracle with ~37 searches per photo, but about half is spread, not evidence**
    (`search_tree.py`; dev3k, setft_own, original protocol). Vocabulary from each photo's own pool: its candidates' countries (6.3 per photo),
    regions (14.5) and +-25 km boxes around candidates outside every area (16.0). Some restricted top-1 < 25 km: 70.3% (area oracle 61.4, pool
    oracle 73.1); beyond the area oracle country +3.6, region +9.0, box +11.2, all +14.0, of which +3.3 beyond the pool oracle too (new places).
    Luck (lesson 120, same vocabulary vs another photo's truth): a random photo +3.3; a same-country photo > 100 km away +7.7 (area oracle there
    6.0), so net ~+6 pts, and the extras are only 1.8x chance where the areas are 10x. By budget (constraints in pool-mass order) real / same-
    country luck: 8 searches +4.0 / +1.6, 16 +8.0 / +3.6, 64 +13.7 / +7.7. By bucket: never an area 89.9% reachable (boxes: they restate pool
    candidates as areas), wrong area 75.0% (regions), retrieval miss only 12.5% (3.3 pts). True region as a reference search: 71.6% (+18.1).
    Speed of `tools/pinpoint.py`: a streamed pass (none / box / country / region) costs ~1.2-1.7 s whatever the batch, 2.4-3.4 ms / query at
    batch 1,024 (~1.5 ms at 4,096; GPU 4 GB); rows / near (box-sized, ~15k rows) 2.6 / 6.5 ms per query on the 10.5 GB resident index.

S2. **A search's result is not evidence the head can act on (step 2, no-go)** (`search_tree.py evidence`, `evidence-megaloc`; 1.31M cached
    searches on all 35,359 train photos, evaluated on dev3k). P(restricted top-1 < 25 km) from prior features A (the constraint's pool mass,
    candidates inside, best score / rank inside) vs A + result features B (restricted top-1 score and drop vs the unconstrained top, margin,
    agreement inside the restricted list, novelty, agreement with the photo's other searches): AUC among alternatives (top-1 >= 25 km from
    the pipeline's answer) 0.686 -> 0.748 / 0.707 -> 0.764 / 0.783 -> 0.829 at 8 / 16 / all searches. But the best-scored alternative is right
    on 6% of photos while some alternative is right on 14 / 19 / 25%, and no switch threshold nets > 0 even in hindsight (cross-fitted -0.01 to
    -0.10 pts; the 30 most confident switches: 3-6 fixes / 10-17 breaks). Step 2b, MegaLoc (query vs restricted top-1; 78k photos, 7 min):
    alone AUC 0.68; added to A+B (dev3k 5-fold) AUC among alternatives +0.002 to +0.006, switches -0.07 to -0.04 pts. (Numbers after the
    `_fit` fix of S3; before it, the same conclusions.) As lessons 111-112 and main point 2: searching finds right regions the
    evidence can't pick. A sequential agent over pool-derived constraints has nothing to condition on; closed in this form.

S3. **Text-proposed places verify far better than pool ones, but not where text could fix the pipeline** (`text_verify.py`; commons26
    dev, 977 photos). Lesson 119's text-search candidates (<= 64 per photo; some right 35.0%, luck 3.7%) and, on the same photos, boxes
    around the pool candidates; Pinpoint restricted to +-25 km around each (77k searches). The restricted result alone separates right from
    wrong text candidates at AUC 0.944 (0.914 away from the free top-1) vs 0.847 / 0.788 for pool boxes: independence from retrieval
    matters (lesson S2's selection effect). But it adds nothing to lesson 119's ranker (top-1 20.7 -> 20.5 linear, 19.1 -> 19.2 MLP),
    whose text features already hold "Pinpoint points here", and the switch against the pipeline stays below the bar (text only +0.41 /
    +0.72, with verification +0.41 / +0.20; SE 0.5-0.7; a blank MegaLoc column had shown +1.1 / +1.2, so these switches move ~1 pt on noise).
    Why: on the 116 photos with a right text candidate and a wrong pipeline, the right place verifies worse than the ranker's wrong top
    (score drop vs the free search -0.24 vs -0.03, right ahead on 31%; index photos within 25 km e^7.2 vs e^8.9, ahead on 16%; MegaLoc ahead
    on 37%). Fresh photos at the right place have few MP16 references; the wrong tops sit in dense, look-alike-rich places. The limit is
    reference coverage at the candidate, which a live imagery tool would supply and the offline gallery can't.
    Found on the way: `megaloc_queries_commons26_setft_own.npy` and `megaloc_commons26_setft_own.npz` (Oct 16, before the bd2d12d path fix)
    are blank (200 photos -> 4 distinct vectors, cosine 0.99999); only the commons26dev files were redone, so the commons26 test run of
    lesson 113 had no MegaLoc signal. Also, `search_tree._fit` ran only 16-250 Adam steps on small sets (now >= 2,000); S2b re-run with it.

## Value-agent track (V)
V1. **Read place names on the pipeline's wrong photos are mostly generic or not the photo's place; misreads are few** (`scripts/name_miss_probe.py`;
    setft_own, `--one-km 0.5` RL lambda 0 answers; each name classed with `tools.maps` (OSM + GeoNames + Overture); luck = the same names near
    another photo's truth, lesson 120). dev3k: wrong with a read place name 196 photos (6.5 pts); some name within 25 km of the truth 2.9 pts vs
    luck 1.4; specific names (<= 3 countries, <= 50 matches) 0.3 pts vs 0.0; a close misspelling (rapidfuzz >= 80) of an OSM name within 5 km
    of the truth 0.3 pts (half spurious: Aarau / vaara); found only elsewhere 2.2 pts (56% of those names not in the verbatim text: inferred lists,
    e.g. a dozen Swiss cities), found nowhere 1.2. commons26 dev: 178 photos (18.2 pts); near the truth 8.9 vs luck 1.4, specific 2.6 vs 0.1;
    misspelling 1.3 (real ones: "Gregorio Arzate" / Aznárez, 昆山门东站 / 艮山门东站, "Hanka Nevada Road" / Nevala); elsewhere 4.2; nowhere 3.8. Of the 160
    wrong commons26 dev photos whose text names the place: name near truth 62, misspelt 13, nowhere 25, elsewhere 16, text but no place name 39,
    nothing read 5. MP16-like photos: better OCR has < 1 pt to fix; commons26: re-reads alone can't reach VALUE_PLAN's 3-pt bar.
V2. **Reverse geocoding as a check: the photo's text fuzzy-matched against each candidate area's names separates right from wrong areas
    barely, and no switch pays** (`scripts/reverse_text_check.py`; the user's idea: per area, every OSM name within 10 km of its centre as a
    closed vocabulary; the 4B's verbatim text + place names matched with rapidfuzz, so misreads and partial strings count; setft_own, `--one-km
    0.5` RL lambda 0 answers). Whole-string ratio, commons26 dev (484 photos with text, 4.7 areas): AUC right vs wrong areas 0.657; a >= 90
    match at the truth 38.4% vs another photo's truth 4.1%; on the 71 photos (7.3 pts) the pipeline gets wrong with a right area, the right area
    outscores the answer on 39% (tied 35%); best switch with in-sample thresholds 8 fixes / 5 breaks (+0.31, upper bound). dev3k (1,175 photos):
    AUC 0.595, truth 32.0% vs luck 7.0%, right above the answer 40% of 145 photos (4.8 pts), every switch loses (best 3 / 33, -1.00). Word-level
    matching (>= 6 letters) is noise (AUC 0.54-0.55; luck 31-32%). The truth's names do match the photo, but wrong areas are name-dense places
    that match too: main point 2 again, now for text; lessons 118 / 140's exact rare-name columns keep what signal there is.
V3. **SigLIP2's text side as a place-name proposer: right names sit where the pipeline is already right** (`scripts/siglip_names_probe.py`;
    dev3k, setft_own, `--one-km 0.5` RL lambda 0 answers; each photo's cached SigLIP2 image embedding scored against 1.39M prompts "this is a
    photo of {name}." for 148k GeoNames towns (pop >= 1000) and 1.24M geotagged English Wikipedia pages, top names merged within 25 km; luck =
    the same proposals vs another photo's truth > 100 km away). SigLIP2 needs lowercased prompts (20-city check: right city first 24 / 40
    lowercased vs 4 / 40 cased). Top-1 / 5 / 10 proposal < 25 km: 14.0 / 26.9 / 31.9% (luck 0.0 / 0.6 / 1.0; Wikipedia pages are 93% of top-1s).
    On the pipeline's misses 1.5 / 4.4 / 6.3 pts (luck 0.0 / 0.4 / 0.6); where no candidate area is right 0.3 / 1.6 / 2.5 (luck 0.0 / 0.3 / 0.5),
    no pool candidate right 0.1 / 0.9 / 1.4. Pre-declared bar (no area right, top-5, net of luck >= 3 pts): +1.3, no-go. The text tower knows
    the places Pinpoint's gallery already covers; it adds no new candidates where retrieval fails.
V4. **Text visible in the index photos as area evidence: the right area's photos show the query's text only 14% of the time where it would
    matter, and no switch pays** (`scripts/gallery_text_check.py`, `scripts/gallery_text.sh`; the user picked it from a list of sources of new
    information: the strings people actually see at a place (shop fronts, posters, brands) rather than gazetteer names; dev3k, setft_own,
    `--one-km 0.5` RL lambda 0 answers). The base 4B read all visible text in the 12 index photos of every candidate area of the 1,117 dev3k
    photos with read text (50,209 photos, 48,885 answered, 88% with some text; ~1 h on the L40S); the query's verbatim text + place names
    (>= 5 letters) fuzzy-matched per area (rapidfuzz ratio). AUC right vs wrong areas 0.650 (673 / 2,324); a >= 90 match in right areas 25.7%,
    wrong 7.4%, another photo's area 0.4%. On the 141 photos (4.7 pts) the pipeline gets wrong with a right area: right area above the answer's
    area 48%, tied 17%; a >= 90 match in the right area only 14% (the answer's area 10%), with the answer's area < 85 on 11 photos, vs 26 of 656
    pipeline-right photos where a wrong area matches >= 90 and the answer's doesn't. Best in-sample switch 1 fix / 2 breaks (-0.03 pts, bar +1).
    Matching text concentrates where the pipeline is already right (32% of its right answers vs 10% of its wrong ones; main point 3 again):
    the index's text sits where retrieval succeeds.
V5. **A second image model per area: PE-Core-G adds +1.4 pts < 25 km in the trained answer head, +2.0 with SigLIP2's own cosine columns
    beside it; the first gain over the head since lesson 140** (`scripts/second_model_gate.py`, `second_model_where.py`; dev3k, setft_own,
    `--one-km 0.5` head, every tool revealed). Meta's Perception Encoder (PE-Core-G 448 px / L 336 px, timm) and DINOv3 (ViT-7B / L, CLS at
    224 px) embed each query and its areas' 12 restricted-search photos; per area the best / top-3 cosine and both relative to the other areas
    (MegaLoc's columns, lesson 93). Gate (dev3k CV stack on the head's option logits; head 47.20 / 30.77, MegaLoc re-added +0.04): PE-L +0.23,
    PE-G +0.79 / +0.63 (SE 0.32), DINOv3-L +0.02, DINOv3-7B +0.23, PE-G + DINOv3-7B +1.04; fewer photos per area costs (PE-G at 6 / 4 / 3:
    +0.58 / +0.40 / +0.37). The retrieval-track session's control: SigLIP2-g's own pooled embedding in the same columns -0.03 alone, +1.19 with
    PE-G; PE-G's top area is right on 30.9% of the head's misses with a right area vs SigLIP2 24.6% (DINOv3-7B 31.2%). The gain is switches
    to another area on cross-region misses (200-750 km +1.01), mostly urban (+0.72), not near-duplicates (cosine > 0.9: +0.14), not the
    photographer's photos (dropping them from the columns: +1.23). **Trained head** (train-side columns for the 19,321 train photos whose
    options are neither all right nor all wrong, the only ones with answer-head gradient, 746k area photos, PE-G ~7 h + DINOv3-7B ~4 h on the
    L40S; others mean-filled; 3 seeds; % < 25 / 1 km): head 47.17 / 30.63; + PE-G 48.53 / 31.60 (+1.37 / +0.97, SE 0.34); + DINOv3-7B
    +0.78 / +0.43 (SE 0.23); + SigLIP2 +0.29 / +0.26; **+ PE-G + DINOv3-7B 48.76 / 31.66 (+1.59 / +1.02, SE 0.35): pre-declared bar (>= +1.0,
    1 km >= -0.3) passed**; + PE-G + SigLIP2 49.21 / 31.91 (+2.04 / +1.28, SE 0.38), all three 49.29 / 31.94 (+2.12 / +1.31); the last two
    were chosen after the gate. DINOv3 adds +0.2 over PE-G (+0.1 over PE-G + SigLIP2): marginal. PE-G sees what SigLIP2 and MegaLoc share
    blind spots on (look-alike townscapes); SigLIP2's cosine helps as the reference PE-G disagrees with. Not yet: commons26, the RL policy's
    cost curve with the new columns, the test set.
