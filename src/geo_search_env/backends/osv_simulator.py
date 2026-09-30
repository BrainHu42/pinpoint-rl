"""API-shaped geographic search over an audited OSV-5M catalog."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from ..core.backend import ToolBackendError
from ..core.contracts import (
    Action,
    ActionCapability,
    ActionKind,
    BackendCapabilities,
    MatchScore,
    PublicEpisode,
    ReferenceAsset,
    SearchResultCard,
    StructuredError,
    ToolResponse,
)
from ..core.geography import SnapshotWorld
from ..data.corpus import CorpusPage, CorpusStore, InMemoryCorpusStore
from ..models.matching import Matcher


class OSVSimulatorTools:
    """Bounded Mapillary-like search over historical OSV-5M camera records."""

    def __init__(
        self,
        corpus: CorpusStore | SnapshotWorld,
        matcher: Matcher,
        *,
        page_size: int = 8,
        max_radius_m: float = 5_000.0,
    ) -> None:
        if isinstance(corpus, SnapshotWorld):
            corpus = InMemoryCorpusStore(corpus)
        if not isinstance(corpus, CorpusStore):
            raise TypeError("corpus must implement CorpusStore")
        if type(page_size) is not int or not 1 <= page_size <= 8:
            raise ValueError("page_size must be between one and eight")
        if not 0 < max_radius_m <= 25_000:
            raise ValueError("max_radius_m must be in (0, 25000]")
        self.corpus = corpus
        self.matcher = matcher
        self._page_size = page_size
        self._max_radius_m = float(max_radius_m)

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

    def _digest(self, request: dict) -> str:
        value = {"corpus_version": self.corpus.corpus_version, "request": request}
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _offset(self, cursor: str | None, request: dict) -> int:
        if cursor is None:
            return 0
        parts = cursor.split(":")
        if len(parts) != 3 or parts[:2] != ["osv-v1", self._digest(request)]:
            raise ToolBackendError("invalid_cursor", "cursor does not belong to this search")
        try:
            return int(parts[2])
        except ValueError as error:
            raise ToolBackendError("invalid_cursor", "cursor offset is invalid") from error

    @staticmethod
    def _anchor(episode: PublicEpisode, corpus: CorpusStore, anchor_id: str):
        initial = {item.candidate_id: item.coordinate for item in episode.initial_candidates}
        if anchor_id in initial:
            return initial[anchor_id]
        reference = corpus.get_reference(anchor_id)
        if reference is None:
            raise ToolBackendError("invalid_anchor", "anchor has no public camera coordinate", charge_attempt=False)
        return reference.published_coordinate

    def _response(self, page: CorpusPage, request: dict) -> ToolResponse:
        seen: set[str] = set()
        cards = []
        for hit in page.hits:
            if hit.reference.asset_id in seen:
                raise ToolBackendError("invalid_corpus_page", "corpus returned duplicate assets")
            seen.add(hit.reference.asset_id)
            cards.append(SearchResultCard.from_reference(hit.reference, hit.distance_from_search_anchor_m))
        cursor = f"osv-v1:{self._digest(request)}:{page.next_offset}" if page.next_offset is not None else None
        return ToolResponse(ActionKind.SEARCH_NEAR, search_results=tuple(cards), next_cursor=cursor)

    def search_near(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float, cursor: str | None
    ) -> ToolResponse:
        anchor = self._anchor(episode, self.corpus, anchor_id)
        request = {"anchor_id": anchor_id, "anchor": anchor.to_dict(), "radius_m": radius_m}
        offset = self._offset(cursor, request)
        page = self.corpus.search_near(anchor, radius_m, offset=offset, limit=self._page_size, exclude_asset_id=anchor_id)
        if offset and offset >= page.total:
            raise ToolBackendError("invalid_cursor", "cursor offset is outside the result set")
        return self._response(page, request)

    def inspect_coverage(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float
    ) -> ToolResponse:
        anchor = self._anchor(episode, self.corpus, anchor_id)
        return ToolResponse(
            ActionKind.INSPECT_COVERAGE,
            coverage_summary=self.corpus.inspect_coverage(anchor_id, anchor, radius_m),
        )

    def _references(self, asset_ids: Sequence[str]) -> tuple[ReferenceAsset, ...]:
        result = []
        for asset_id in asset_ids:
            reference = self.corpus.get_reference(asset_id)
            if reference is None:
                raise ToolBackendError("unknown_asset", "corpus no longer contains the asset")
            result.append(reference)
        return tuple(result)

    def open_results(self, episode: PublicEpisode, asset_ids: Sequence[str]) -> ToolResponse:
        references = self._references(asset_ids)
        return ToolResponse(
            ActionKind.OPEN_RESULTS,
            opened_assets=references,
            match_scores=tuple(MatchScore(item.asset_id, self.matcher.score(episode.query_asset_id, item)) for item in references),
        )
