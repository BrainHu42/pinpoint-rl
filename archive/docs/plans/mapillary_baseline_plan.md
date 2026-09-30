# Mapillary-First Baseline Experiment Plan

## Objective

Train a small RL search controller on an audited, API-shaped OSV-5M simulator
to improve the frozen Pinpoint submission retrieval model's geolocation
results, then test whether that improvement transfers to bounded live
Mapillary search.

The controller starts from ranked geographic hypotheses produced by the frozen
Pinpoint contrastive retrieval model and may spend budget to surface additional
candidate locations from the frozen MP16-Pro and OSV-5M corpora. Under a fixed
acquisition budget, it decides where to search, which results to inspect, and
when to stop. It may finish with a surfaced location, the coordinate of an
opened eligible reference, or Pinpoint's rank-1 retrieval result.

The current experiment is offline. Live Mapillary characterization and transfer
remain deferred until candidate acquisition, fixed search, and learned control
pass their offline gates.

## Primary research question

Under the same acquisition budget, does an RL controller trained on historical
OSV-5M interactions improve all-query 1-kilometer geolocation accuracy on live
Mapillary relative to frozen Pinpoint retrieval, fixed search over its
candidates, and supervised imitation?

The experiment is successful only if the improvement survives held-out
locations, independent captures, and the simulator-to-live transition without
an unacceptable increase in harmful refinements.

## Hypotheses

- **H1 — Search improves retrieval:** a fixed geographic-search policy improves
  on Pinpoint's rank-1 retrieval result when independent Mapillary evidence is
  available.
- **H2 — Adaptation helps:** a controller that conditions later actions on
  returned evidence outperforms fixed search at matched cost.
- **H3 — Learning helps:** supervised or RL controllers outperform the best
  scripted policy at matched cost on held-out simulator episodes.
- **H4 — RL adds value:** RL outperforms supervised imitation at matched cost.
- **H5 — Strategy transfers:** the ordering of useful policies and a meaningful
  part of the learned improvement survive bounded live Mapillary evaluation.

Failure of a later hypothesis does not invalidate results for earlier ones.

## Scope

### Included

- OSV-5M as the historical Mapillary-derived training corpus.
- Frozen OSV-5M and MP16-Pro retrieval as candidate-location generators.
- An offline OSV-5M simulator for the current evaluation stage.
- Live Mapillary geographic image discovery only in the deferred transfer
  stage.
- The frozen Pinpoint submission contrastive retriever, including its SigLIP2
  image encoder, checkpoint, exact location index, and ranked candidates.
- A frozen query/reference visual matcher that compares normalized outputs from
  the Pinpoint retrieval model's image tower, rather than raw SigLIP2 features.
- Bounded coverage inspection, geographic radius search, bounded pagination,
  selective image opening, and terminal selection or fallback.
- A small model policy trained first by supervised imitation and then, if the
  preceding gates pass, by offline RL.
- A 1 km primary success threshold, with 25 m and 100 m reported only as
  secondary diagnostics.

### Excluded from the baseline experiment

- Flickr and MP16-Pro as simulator worlds, trainable components, or image
  evidence sources. Frozen MP16-Pro retrieval may emit clustered candidate
  locations, but never retrieved images or asset identifiers.
- Global SigLIP text-to-image archive retrieval.
- Text search, reverse-image search, maps, aerial imagery, and 3D
  reconstruction.
- Wikimedia/Wikipedia place discovery.
- Updating the Pinpoint retrieval checkpoint, SigLIP2 image encoder, location
  index, or visual matcher during policy training.
- Unrestricted live-provider calls during training.

Sequence browsing is a planned extension. Add it only after the minimal
geographic-search experiment passes its simulator and live-provider gates.

## Unit of evaluation

One episode contains:

1. a metadata-sanitized query image;
2. ranked initial coordinates from frozen Pinpoint retrieval plus candidate
   locations surfaced from frozen OSV-5M or MP16-Pro retrieval;
3. Pinpoint's rank-1 result as the designated baseline candidate;
4. a fixed credit and action budget;
5. public search cards and opened-image evidence produced by the selected
   backend; and
6. a private query coordinate used only after termination.

Every policy receives the same frozen retrieval capabilities, backend budget,
and public evidence for a given episode. Ground truth,
availability diagnostics, and reward never appear in policy observations or
prompts. Retrieval-only performance is always measured from the untouched
rank-1 candidate, never from a candidate selected after seeing search evidence.

## Minimal action surface

```text
surface_candidate_locations(corpus, cursor)
inspect_coverage(anchor_id, radius_m)
search_near(anchor_id, radius_m, cursor)
open_results(asset_ids)
finish(candidate_id)
```

`surface_candidate_locations` performs frozen corpus retrieval internally,
clusters retrieved items geographically, and returns location cards. Cards may
contain coordinates, corpus provenance, source-local rank and score, support
count, and dispersion. They never expose the retrieved images, thumbnails, or
asset identifiers. MP16-Pro and OSV-5M scores are not assumed comparable.

`inspect_coverage` cheaply summarizes local Mapillary image and sequence
density, capture-time range, and panorama fraction. It does not discover or
open terminal candidates. Simulator summaries are exact over the audited
catalog; live summaries are explicitly approximate because they aggregate
bounded public vector tiles.

`search_near` queries Mapillary around either an initial hypothesis or an
eligible public coordinate discovered during the episode. `open_results`
reveals the permitted image and computes one frozen query/reference match
score. `finish` selects an initial candidate or an opened reference with an
eligible camera coordinate.

The baseline experiment does not expose `search_images` or `verify_matches`.
Repeated opening must not incur another matching charge. Invalid and
unsupported actions consume a bounded decision attempt but reveal no evidence.

## Fixed experimental components

Freeze and version the following before producing training trajectories:

- Pinpoint retrieval checkpoint, SigLIP2 revision and preprocessing, MP16
  location-index manifest, exact-search implementation, inference settings,
  candidate count, and normalized-score transformation;
- query/reference matcher, Pinpoint query/reference source adapters,
  preprocessing, output normalization, and score calibration;
- OSV-5M revision, approved asset manifest, and catalog indexes;
- train, validation, snapshot-test, and live-test location lists;
- action schema, provider limits, budgets, and cost schedule;
- coverage zoom, maximum tiles per request, and aggregation rules;
- observation and prompt schema;
- reward definition and all evaluation metrics; and
- random seeds and policy-training configurations.

Changes after freezing create a new experiment version and require rerunning
all baselines.

## Data construction and audit

Build one shared OSV-5M reference world rather than an episode-specific world
centered on the hidden truth. The world must contain useful and distracting
regions so that searches around wrong hypotheses produce realistic results.

Use the joined raw-image/embedding loader as the asset layer, then construct
audited geographic and sequence-safe split indexes separately; cache-row order
or ID-hash partitions are not valid experimental splits.

For every retained asset, record:

- opaque environment ID and original Mapillary/OSV identifier;
- source revision and checksums;
- published coordinate and its provenance;
- whether the coordinate represents an original or provider-computed camera
  position;
- coordinate uncertainty where available;
- capture time, creator, and sequence ID;
- image availability and corruption status; and
- license, attribution, retention basis, and review status.

### Leakage controls

- Split by geographic area before optimization.
- Keep Mapillary sequences, same captures, and near-duplicate imagery within a
  single split.
- Exclude exact and perceptual query/reference duplicates.
- Exclude query images used to train the frozen Pinpoint checkpoint; if its
  training provenance cannot establish this, treat the pilot as diagnostic
  rather than held-out evidence.
- Exclude same-sequence query/reference pairs from headline evaluation.
- Strip EXIF, filenames, paths, coordinates, and place-derived metadata from
  query inputs.
- Do not expose OSV-derived country, region, climate, or environment labels to
  the policy.
- Do not use private query coordinates to construct result pages or rankings.

### Feasibility sample

Before building the full simulator, run frozen Pinpoint retrieval on 20–50
representative, checkpoint-independent queries and use those ranked outputs as
episode anchors. Keep queries with no qualifying reference in the denominator.
Report:

- independent eligible-reference availability within 1 km;
- availability by geography and urban/rural category;
- missing or corrupt assets;
- same-sequence and duplicate rejection rates;
- coordinate provenance and uncertainty;
- rights-review rejection rate;
- frozen Pinpoint-tower matcher recall on eligible independent references; and
- the fraction of queries for which geographic search can acquire a useful
  reference within a relaxed budget.

Do not proceed to policy training if evidence availability or matcher recall is
too low to support a meaningful search-policy comparison.

## Simulator contract

The simulator must answer arbitrary supported geographic searches over the
audited world through the same public contract used by the live backend. It
must not replay only a preselected set of favorable queries.

Model these measured Mapillary behaviors:

- bounding-box retrieval followed by radius filtering;
- coverage-tile selection, radius filtering, feature deduplication, and empty
  coverage summaries;
- page size, cursors, and maximum radius;
- result ordering and page density;
- empty and sparse pages;
- unavailable or stale assets;
- coordinate and sequence metadata visibility;
- latency and charged provider failures; and
- antimeridian and bounding-box edge cases.

Search returns bounded cards and thumbnails. Full image evidence and matcher
scores become visible only after `open_results`. The simulator never reads the
private query coordinate.

## Deferred live Mapillary calibration

Do not execute this stage during offline baseline development. Begin it only
after the fixed offline search policy and selected learned policy are frozen.

Before training, run time-stamped, credentialed probes through the normalized
live backend. Include correct and wrong anchors, several radii, coverage-tile
boundaries, multiple pages, zero-result areas, deleted IDs, sparse areas, and
provider failures.

The live implementation uses Mapillary's authenticated public vector-tile
endpoint, `mly1_public/2/{z}/{x}/{y}`, and the `image` and `sequence` source
layers demonstrated by Mapillary's first-party
[API demo](https://github.com/mapillary/api-demo/blob/main/app.js). Coverage is
therefore an adapter-level aggregate, not a provider JSON summary endpoint.
Pin the decoder version and revalidate the layer schema during each live probe.

Compare simulator and live behavior on:

- supported-action coverage;
- image/sequence coverage counts, capture ranges, panorama fractions, tile
  counts, and false-empty rates;
- result count and page density;
- rank/order behavior;
- coordinate and sequence-field availability;
- radius-filter behavior;
- cursor behavior;
- missing/deleted asset frequency;
- error categories;
- latency; and
- result overlap where historical and live IDs can legitimately be compared.

Record probe time, public responses, adapter configuration, and aggregate
statistics without secrets, provider URLs, or private labels in policy-visible
traces.

## Policies and comparisons

Evaluate these policies in order:

1. **Pinpoint retrieval only:** immediately retain Pinpoint's rank-1 result.
2. **Fixed retrieval-anchored search:** search a predeclared radius around
   Pinpoint candidates in rank order, open a fixed number of results, then
   apply a fixed selection rule.
3. **Adaptive scripted search:** use coverage summaries and public match
   evidence to decide whether to search, expand, inspect another candidate, or
   stop.
4. **Supervised controller:** imitate valid scripted and curated trajectories.
5. **RL controller:** initialize from the supervised controller and optimize
   terminal reward in the offline simulator.

All comparisons use identical episode inputs and matched budgets. Also report
budget-response curves rather than results at only one budget.

## Reward and metrics

Use a terminal-only reward that balances task success, improvement over the
baseline, acquisition cost, and failed final answers. Freeze its coefficients
before the final training runs.

### Primary metrics

- all-query accuracy within 1 km;
- change in 1 km accuracy relative to frozen Pinpoint rank-1 retrieval;
- mean acquisition credits per query; and
- harmful-refinement rate: baseline correct within 1 km but final answer
  incorrect within 1 km.

### Diagnostic metrics

- reference availability within 1 km;
- search acquisition success conditional on availability;
- final selection success conditional on acquisition;
- fallback rate;
- final and baseline distance-error distributions;
- accuracy within 25 m and 100 m;
- coverage inspections, images opened, searches issued, and pages consumed;
- invalid and unsupported action rate;
- provider failure rate and latency; and
- simulator-to-live deltas for accuracy, cost, and behavior.

Report unconditional metrics first. Conditional metrics diagnose the pipeline
but must not hide unavailable or failed queries.

Use paired confidence intervals or a paired bootstrap over episodes for policy
differences. Predeclare the primary comparison, test split, and random seeds
before the final run.

## Execution phases and gates

### Phase 0 — Contract stabilization

Deliverables:

- the five-action surface implemented in the offline simulator;
- Mapillary-only capabilities;
- consistent 1 km reward, availability, summaries, fixtures, and docs;
- legacy actions isolated from this experiment; and
- a green automated test suite.

Gate: identical supported actions and limits are advertised and enforced by
both backends, and public traces contain no private fields.

### Phase 1 — Audited feasibility pilot

Deliverables:

- frozen Pinpoint retrieval episode inputs plus pinned OSV-5M manifest and
  indexes;
- leakage-safe pilot split;
- 20–50-query feasibility report; and
- matcher and relaxed-budget acquisition diagnostics.

Gate: independent 1 km evidence and matcher recall are sufficient to make
policy choice consequential. Otherwise narrow or stop the experiment.

### Phase 2 — Multi-corpus candidate acquisition

Deliverables:

- location-only proposal adapters for MP16-Pro and OSV-5M;
- matched-budget candidate-location recall curves;
- source allocation and geographic-deduplication diagnostics; and
- public proposal traces that contain no retrieved asset identifiers.

Gate: the combined top-50 proposal set makes independent evidence reachable for
at least 30% of available queries and improves on MP16-Pro by at least 10
percentage points.

### Phase 3 — Simulator calibration

Deliverables:

- query-general disk-backed simulator;
- simulator/live comparison report;
- baseline-only, fixed, and adaptive-scripted benchmark results; and
- budget-response curves.

Gate: mismatch is quantified, scripted acquisition affects outcomes, and the
best search baseline improves accuracy or cost on held-out snapshot episodes.

### Phase 4 — Supervised controller

Deliverables:

- replayable trajectory format;
- model-policy adapter with strict action parsing;
- prompt-isolation tests; and
- supervised held-out results.

Gate: the model reliably completes episodes, rarely emits invalid actions, and
matches a meaningful scripted baseline.

### Phase 5 — Offline RL

Deliverables:

- reproducible RL training runs;
- learning curves and seed variance;
- held-out snapshot comparisons at matched budgets; and
- policy-behavior analysis.

Gate: RL improves on supervised imitation or clearly establishes that it adds
no value for this action space.

### Phase 6 — Frozen live transfer

Freeze policy selection before examining final live-test outcomes. Do not tune
on the live test set.

Deliverables:

- bounded live evaluation on predeclared locations;
- matched-budget comparison of all feasible policies;
- simulator-to-live transfer analysis; and
- provider failure and harmful-refinement audit.

Gate: decide which conclusion the evidence supports—successful transfer,
partial transfer with a narrowed claim, or failure caused by availability,
matching, simulation mismatch, or policy learning.

## Stop and revision conditions

Stop or revise the claimed experiment if any of the following occurs:

- too few queries have independent eligible references within 1 km;
- the frozen matcher cannot recognize those references at useful precision;
- fixed search cannot acquire useful references under a relaxed budget;
- live Mapillary access cannot support the declared action contract;
- simulator/live behavior differs enough to make training interactions
  misleading;
- gains depend on same-sequence, duplicate, or private-location leakage;
- learned policies improve only conditional metrics while harming all-query
  accuracy; or
- live harmful refinements erase the accuracy benefit.

For a negative result, attribute failure to the earliest failed pipeline stage:
availability, acquisition, recognition, selection, or transfer.

## Required experiment artifacts

Each released run should identify:

- experiment and contract version;
- code revision;
- OSV-5M manifest and split versions;
- Pinpoint retrieval, SigLIP2, location-index, and matcher versions;
- simulator calibration version;
- policy checkpoint and training configuration;
- budget and reward configuration;
- public episode traces;
- separately protected private scores; and
- aggregate results with uncertainty intervals.

Raw dataset paths, credentials, provider URLs, private coordinates, and private
diagnostics must not appear in policy observations or public traces.

## Definition of completion

The baseline experiment is complete when every policy has been evaluated on
the same held-out snapshot episodes and the frozen selected policies have been
evaluated on the predeclared live Mapillary set at matched budgets. The final
report must state:

1. whether geographic search improves frozen Pinpoint rank-1 retrieval;
2. whether adaptive control improves fixed search;
3. whether supervised learning improves scripted control;
4. whether RL improves supervised imitation;
5. whether the learned behavior transfers live; and
6. which pipeline stage limits performance when it does not.
