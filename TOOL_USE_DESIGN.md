# Tool-use agent: design inputs (seeded 2026-10-20 from track S)

The project's target is an agent that calls live data tools at inference (CLAUDE.md). The tool contract is `src/geo_search_env/tools/__init__.py`
(offline stand-ins: `pinpoint`, `maps`, `web`). This file collects what experiments say the live agent should do; it starts from track S
(`SEARCH_PLAN.md`, closed offline; lessons S1-S3 in `LESSONS.md`; code `experiment/search_tree.py`, `experiment/text_verify.py`; cached
outputs `artifacts/search/`). Add inputs from other tracks here.

## What the evidence says
- **Search where retrieval didn't already look, and verify only there.** Constrained Pinpoint searches over a photo's own candidates find
  right regions (net of luck ~+6 pts beyond the area oracle on dev3k, S1), but their results can't verify them: every pool region holds a
  look-alike by construction, so no switch nets > 0 even in hindsight (S2, 1.31M cached train searches). Places proposed by evidence retrieval
  didn't produce (read text -> name search) verify far better: restricted-search AUC 0.944 vs 0.847 for pool boxes on the same commons26
  dev photos (S3).
- **The binding limit is reference imagery at the candidate.** Where text is right and the pipeline wrong (116 commons26 dev photos), the right
  place has few MP16 photos (within 25 km e^7.2 vs e^8.9 at the wrong pick) and verifies worse than the wrong pick (score drop ahead on 31%,
  MegaLoc 37%). Old Flickr (YFCC) and OSV-5M were too sparse or off-view (lessons 135, 136). The offline gallery can't supply it; a live tool can.
- **Headroom:** a right text candidate exists on 35.0% of commons26 dev photos (luck 3.7%), 16.8% where the pipeline is wrong; a perfect
  choice among text candidates is +4.2 pts (lesson 121). On MP16-like photos text sits where retrieval already succeeds (main point 3).

## Proposed design
- **New tool: `imagery(lat, lon, radius)`**: recent photos near a point (Street View, Places photos, Commons / Flickr APIs), compared with
  the query by MegaLoc (same spot) and SigLIP2 (same kind of scene). Costly and rate-limited, so the policy decides when to call it.
- **Loop:** read the photo -> propose places from text / names (`maps`, `web`) -> free offline check (`pinpoint` restricted to +-25 km;
  batched ~1.5-3 ms, one streamed query ~1.2 s, 2.6 ms on the GPU-resident index) -> if the check is blind (few index photos near the place,
  `Pinpoint.rows_near`) call `imagery` -> answer, or try the next candidate.
- **Where RL earns its place:** the next action depends on the last outcome (offline check confident / blind, imagery match / no match), and
  calls have costs. Judge against fixed recipes and heuristic gates at matched call budget with realistic live-tool costs (the user's rule,
  2026-10-16; SEARCH_PLAN.md go / no-go).
- **Don't spend calls on pool-derived candidates** (S2): their verification is uninformative.

## First test for a live verifier
The 116 commons26 dev photos with a right text candidate and a wrong pipeline answer (`text_verify.py evaluate`, section 4). A verifier pays
if it ranks the right place above the text ranker's wrong pick on well over half of them (offline checks: 16-37%), then measure the switch
against the pipeline on all 977 dev photos (bar +1 pt at 25 km), with dev3k for regressions.

**Result (lesson 138, 2026-10-20): no-go as tested.** With the 20 nearest photos, Commons 50% (AUC 0.49), Mapillary 39% (0.40); imagery at
the photo's own location reaches only 64% / 54%. When Commons covers the view, MegaLoc matches it unambiguously (0.47-0.58); usually the
nearest files show other things, and right text places are a median 8.3 km off. Mapillary at 200 frames changes nothing. Open: dense
Commons (or a search over the candidate's whole area), blocked by upload.wikimedia.org's bot limit (~1 cold thumbnail / s, less after
bursts), so it needs a multi-day crawl or a higher limit.

## Known data issues
- `artifacts/agent/megaloc_queries_commons26_setft_own.npy` and `megaloc_commons26_setft_own.npz` (full commons26, Oct 16) are blank
  (written before the bd2d12d image-path fix); only the commons26dev files are valid. Lesson 113's commons26 test run had no MegaLoc signal.
