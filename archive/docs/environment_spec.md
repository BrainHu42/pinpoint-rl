# Mapillary Baseline Environment Specification

## Boundary

The environment evaluates search control around coordinates supplied by the
frozen Pinpoint submission contrastive retriever. Pinpoint's rank-1 result is
the baseline answer and its remaining ranked results are starting anchors. The
environment does not perform initial geolocation and never uses private query
coordinates during rollout.

Historical training uses an audited OSV-5M catalog. Controlled transfer
evaluation uses live Mapillary. Both implement the same `ToolBackend` contract.

## Episode input

A public episode contains an opaque query-image ID, frozen Pinpoint retrieval
candidates, its rank-1 baseline candidate, an acquisition budget, and optional
upstream provenance. Query coordinates and evaluation metadata are separate
private records loaded only after all public rollouts finish.

## Actions

```text
inspect_coverage(anchor_id, radius_m)
search_near(anchor_id, radius_m, cursor)
open_results(asset_ids)
finish(candidate_id)
```

An anchor is an initial candidate or a returned Mapillary camera card.
`inspect_coverage` returns a bounded summary of nearby image and sequence
coverage, capture range, and panorama fraction without discovering terminal
candidates. Snapshot counts are exact for the audited catalog; live counts are
approximate aggregates of public vector tiles.

`search_near` returns bounded public cards and an opaque cursor.
`open_results` resolves image bytes and computes one frozen query/reference
match score. It is the only action that makes a reference a terminal candidate.
`finish` selects an initial candidate or opened reference.

There is no provider selector: every reference source is Mapillary. Text
search, global semantic retrieval, verification, and sequence browsing are not
part of the baseline contract.

## Capabilities and costs

Every backend advertises exactly the four actions, its maximum radius, page
size, open batch size, evidence types, and contract version. The episode budget
defines coverage, search, and per-new-result opening costs. Reopening an
acquired result costs zero. Invalid actions consume a decision attempt but
reveal no evidence. Provider failures may charge the attempted acquisition
cost.

When no acquisition remains affordable, only `finish` is accepted. Failure to
produce a valid final candidate resolves to the designated baseline and is
marked as fallback.

## Evidence visibility

Search cards expose an opaque asset ID, thumbnail handle, published camera
coordinate, coordinate provenance and uncertainty, distance from the public
search anchor, and available capture, creator, and sequence fields. Opening
exposes the reference record and frozen
match score. Local paths, raw provider URLs, credentials, and private labels
never enter an observation or public trace.

Coverage summaries expose image and sequence counts, oldest/newest capture
times, panorama fraction when present, number of tiles queried, and whether the
values are approximate. They do not expose raw tile features or provider IDs.

## Simulator

`OSVSimulatorTools` consumes a bounded `CorpusStore`. The production store may
be disk-backed and must answer arbitrary supported geographic searches without
reading private labels. Cursors are bound to corpus version and request. The
simulator is calibrated to measured Mapillary pagination, density, failure,
and latency behavior; a historical archive is not described as live inventory.

`OSV5MDataset` is the lower-level asset source for that store. It joins raw
OSV-5M images with row-aligned cached SigLIP embeddings through read-only
memory maps. Dataset loading does not assign experimental splits; audited
geographic and sequence-safe split indices remain a separate layer.

The cached SigLIP vectors are inputs to recognition, not its final comparison
space. The frozen matcher projects a query through the Pinpoint `mp16` image
adapter and shared tower, projects OSV references through the `osv5m` adapter
and shared tower, L2-normalizes the outputs, and compares them by cosine.

## Live backend

`MapillaryApiClient` sends allowlisted bounding-box requests, then filters
returned computed camera coordinates to the requested radius.
Coverage inspection reads authenticated `mly1_public` vector tiles at a frozen
zoom, filters decoded geometry to the requested radius, deduplicates feature
IDs across tiles, and rejects windows above a fixed tile cap.
`LiveMapillaryTools` hashes provider IDs and pagination state into opaque
handles. Opened images are held in an ephemeral allowlisted cache for matching.
Time-stamped public probes support simulator calibration.

## Scoring

The private scorer evaluates only completed traces. The default reward is:

```text
1[final error <= 1 km]
  + progress_weight * (log1p(baseline error) - log1p(final error))
  - cost_per_credit * credits used
  - failure_penalty * 1[fallback]
```

Headline summaries report final and baseline 1 km accuracy, accuracy delta,
mean acquisition cost, fallback rate, and harmful refinement. A harmful
refinement changes a baseline answer that was correct within 1 km into a final
answer outside 1 km. Availability, acquisition coverage, and 25/100 m accuracy
are diagnostics.

## Reproducibility

Runs record contract, corpus, matcher, policy, budget, reward, and seed
versions. Public traces and private metrics are separate files. Final live
policy selection must be frozen before examining live-test outcomes.
