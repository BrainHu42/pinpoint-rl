"""Deterministic snapshot backend for tests and smoke runs."""

from __future__ import annotations

from typing import Sequence

from ..core.backend import ToolBackendError
from ..core.contracts import (
    Action,
    ActionCapability,
    ActionKind,
    BackendCapabilities,
    CoverageSummary,
    MatchScore,
    PublicEpisode,
    SearchResultCard,
    ToolResponse,
)
from ..core.geography import SnapshotWorld, distance_m
from ..models.matching import Matcher


class LocalSnapshotTools:
    """Small deterministic backend for contract tests and fixture smoke runs."""

    def __init__(
        self,
        world: SnapshotWorld,
        matcher: Matcher,
        *,
        page_size: int = 8,
        max_radius_m: float = 5_000.0,
    ) -> None:
        if type(page_size) is not int or not 1 <= page_size <= 8:
            raise ValueError("page_size must be between one and eight")
        if not 0 < max_radius_m <= 25_000:
            raise ValueError("max_radius_m must be in (0, 25000]")
        self.world = world
        self.matcher = matcher
        self._page_size = page_size
        self._max_radius_m = float(max_radius_m)
        self._references = {item.asset_id: item for item in world.references}

    def supports_episode(self, episode_id: str) -> bool:
        return isinstance(episode_id, str) and bool(episode_id.strip())

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            (
                ActionCapability(ActionKind.SEARCH_NEAR, max_radius_m=self._max_radius_m, page_size=self._page_size),
                ActionCapability(ActionKind.INSPECT_COVERAGE, max_radius_m=self._max_radius_m),
                ActionCapability(ActionKind.OPEN_RESULTS, max_batch_size=2),
                ActionCapability(ActionKind.FINISH),
            ),
            ("coverage_summary", "mapillary_camera_metadata", "match_score"),
        )

    def validate_action(self, action: Action) -> StructuredError | None:
        return None

    def _anchor(self, episode: PublicEpisode, anchor_id: str):
        initial = {item.candidate_id: item.coordinate for item in episode.initial_candidates}
        if anchor_id in initial:
            return initial[anchor_id]
        reference = self._references.get(anchor_id)
        if reference is None:
            raise ToolBackendError("invalid_anchor", "unknown reference anchor", charge_attempt=False)
        return reference.published_coordinate

    def search_near(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float, cursor: str | None
    ) -> ToolResponse:
        anchor = self._anchor(episode, anchor_id)
        try:
            offset = 0 if cursor is None else int(cursor.removeprefix("fixture:"))
        except ValueError as error:
            raise ToolBackendError("invalid_cursor", "invalid fixture cursor") from error
        ranked = sorted(
            (distance_m(anchor, item.published_coordinate), item.asset_id, item)
            for item in self.world.references
            if item.asset_id != anchor_id and distance_m(anchor, item.published_coordinate) <= radius_m
        )
        page = ranked[offset : offset + self._page_size]
        next_offset = offset + len(page)
        next_cursor = f"fixture:{next_offset}" if next_offset < len(ranked) else None
        return ToolResponse(
            ActionKind.SEARCH_NEAR,
            search_results=tuple(SearchResultCard.from_reference(item, distance) for distance, _, item in page),
            next_cursor=next_cursor,
        )

    def open_results(self, episode: PublicEpisode, asset_ids: Sequence[str]) -> ToolResponse:
        try:
            references = tuple(self._references[asset_id] for asset_id in asset_ids)
        except KeyError as error:
            raise ToolBackendError("unknown_asset", "snapshot asset no longer exists") from error
        return ToolResponse(
            ActionKind.OPEN_RESULTS,
            opened_assets=references,
            match_scores=tuple(MatchScore(item.asset_id, self.matcher.score(episode.query_asset_id, item)) for item in references),
        )

    def inspect_coverage(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float
    ) -> ToolResponse:
        anchor = self._anchor(episode, anchor_id)
        nearby = tuple(
            item
            for item in self.world.references
            if distance_m(anchor, item.published_coordinate) <= radius_m
        )
        captures = sorted(item.captured_at for item in nearby if item.captured_at is not None)
        sequences = {item.sequence_id for item in nearby if item.sequence_id is not None}
        return ToolResponse(
            ActionKind.INSPECT_COVERAGE,
            coverage_summary=CoverageSummary(
                anchor_id,
                radius_m,
                len(nearby),
                len(sequences),
                captures[0] if captures else None,
                captures[-1] if captures else None,
                None,
                0,
                False,
            ),
        )
