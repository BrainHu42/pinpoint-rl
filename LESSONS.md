# Lessons

New lessons, one entry each (2-4 lines: the claim, the deciding numbers, the module). Lessons 1-137 are archived in
`docs/archive/LESSONS_1-137.md` (anchor `#lN` per lesson); their main points are in `LEARNINGS.md`. When a new lesson changes a main point,
update `LEARNINGS.md` too.

The two tracks run in parallel sessions (2026-10-20), so each numbers its own lessons and appends only to its own section: R1, R2, ...
for `RETRIEVAL_PLAN.md`, S1, S2, ... for `SEARCH_PLAN.md`. Later single-track work continues from 138.

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
    the pipeline's answer) 0.694 -> 0.745 / 0.706 -> 0.760 / 0.783 -> 0.825 at 8 / 16 / all searches. But the best-scored alternative is right
    on 6% of photos while some alternative is right on 14 / 19 / 25%, and no switch threshold nets > 0 even in hindsight (cross-fitted -0.02 to
    -0.09 pts, precision 10-34%). Step 2b, MegaLoc (query vs restricted top-1; 78k photos, 7 min): alone AUC 0.68; added to A+B (dev3k 5-fold)
    AUC among alternatives -0.02 to 0.00, best switch +0.11 pts. As lessons 111-112 and main point 2: searching finds right regions the
    evidence can't pick. A sequential agent over pool-derived constraints has nothing to condition on; closed in this form.
