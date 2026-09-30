"""Scalable OSV-5M catalog and lazy image-resolution boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..core.contracts import Coordinate, CoverageSummary, ReferenceAsset
from ..core.geography import SnapshotWorld, distance_m


@dataclass(frozen=True, slots=True)
class CorpusHit:
    reference: ReferenceAsset
    distance_from_search_anchor_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ReferenceAsset):
            raise ValueError("corpus hit reference must be a ReferenceAsset")
        distance = self.distance_from_search_anchor_m
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError("corpus hit distance must be numeric")
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("corpus hit distance must be finite and nonnegative")
        object.__setattr__(self, "distance_from_search_anchor_m", float(distance))


@dataclass(frozen=True, slots=True)
class CorpusPage:
    hits: tuple[CorpusHit, ...]
    offset: int
    total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(self.total) is not int or self.total < 0:
            raise ValueError("total must be a nonnegative integer")
        if self.offset + len(self.hits) > self.total:
            raise ValueError("corpus page lies outside its total result count")

    @property
    def next_offset(self) -> int | None:
        value = self.offset + len(self.hits)
        return value if value < self.total else None


@runtime_checkable
class CorpusStore(Protocol):
    """Mapillary camera catalog; implementations must never read private labels."""

    @property
    def corpus_version(self) -> str: ...

    def get_reference(self, asset_id: str) -> ReferenceAsset | None: ...

    def search_near(
        self,
        anchor: Coordinate,
        radius_m: float,
        *,
        offset: int,
        limit: int,
        exclude_asset_id: str | None = None,
    ) -> CorpusPage: ...

    def inspect_coverage(
        self, anchor_id: str, coordinate: Coordinate, radius_m: float
    ) -> CoverageSummary: ...

    def eligible_references_near(
        self, coordinate: Coordinate, radius_m: float
    ) -> Sequence[ReferenceAsset]: ...


@runtime_checkable
class ImageResolver(Protocol):
    def resolve_query_image(self, asset_id: str) -> Path: ...

    def resolve_reference_image(self, asset_id: str) -> Path: ...


class MappingImageResolver:
    """Small-pilot resolver; local paths remain behind opaque asset IDs."""

    def __init__(self, query_images: Mapping[str, Path], reference_images: Mapping[str, Path]) -> None:
        self._queries = self._validated(query_images, "query")
        self._references = self._validated(reference_images, "reference")

    @staticmethod
    def _validated(values: Mapping[str, Path], role: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for asset_id, raw_path in values.items():
            if not isinstance(asset_id, str) or not asset_id.strip():
                raise ValueError(f"{role} IDs must be nonempty strings")
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise ValueError(f"{role} image is missing for asset {asset_id!r}")
            result[asset_id] = path
        return result

    def resolve_query_image(self, asset_id: str) -> Path:
        try:
            return self._queries[asset_id]
        except KeyError as error:
            raise ValueError("unknown query image asset ID") from error

    def resolve_reference_image(self, asset_id: str) -> Path:
        try:
            return self._references[asset_id]
        except KeyError as error:
            raise ValueError("unknown reference image asset ID") from error


class InMemoryCorpusStore:
    """Deterministic reference store for fixtures and audited small pilots."""

    def __init__(self, world: SnapshotWorld) -> None:
        if not isinstance(world, SnapshotWorld):
            raise TypeError("world must be a SnapshotWorld")
        self.world = world
        self._references = {item.asset_id: item for item in world.references}

    @property
    def corpus_version(self) -> str:
        return self.world.corpus_version

    def get_reference(self, asset_id: str) -> ReferenceAsset | None:
        return self._references.get(asset_id)

    def search_near(
        self,
        anchor: Coordinate,
        radius_m: float,
        *,
        offset: int,
        limit: int,
        exclude_asset_id: str | None = None,
    ) -> CorpusPage:
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("pagination requires offset >= 0 and limit >= 1")
        ranked = sorted(
            (
                distance_m(anchor, reference.published_coordinate),
                reference.asset_id,
                reference,
            )
            for reference in self.world.references
            if reference.asset_id != exclude_asset_id
            and distance_m(anchor, reference.published_coordinate) <= radius_m
        )
        hits = tuple(CorpusHit(reference, distance) for distance, _, reference in ranked)
        return CorpusPage(hits[offset : offset + limit], offset, len(hits))

    def eligible_references_near(
        self, coordinate: Coordinate, radius_m: float
    ) -> tuple[ReferenceAsset, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.world.references
                    if distance_m(coordinate, item.published_coordinate) <= radius_m
                ),
                key=lambda item: item.asset_id,
            )
        )

    def inspect_coverage(
        self, anchor_id: str, coordinate: Coordinate, radius_m: float
    ) -> CoverageSummary:
        nearby = self.eligible_references_near(coordinate, radius_m)
        captures = sorted(item.captured_at for item in nearby if item.captured_at is not None)
        sequences = {item.sequence_id for item in nearby if item.sequence_id is not None}
        return CoverageSummary(
            anchor_id=anchor_id,
            radius_m=radius_m,
            approximate_image_count=len(nearby),
            approximate_sequence_count=len(sequences),
            oldest_capture_at=captures[0] if captures else None,
            newest_capture_at=captures[-1] if captures else None,
            panorama_fraction=None,
            tiles_queried=0,
            is_approximate=False,
        )
