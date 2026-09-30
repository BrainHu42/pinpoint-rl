"""Versioned snapshot world and geographic distance utility."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .contracts import Coordinate, ReferenceAsset


def distance_m(first: Coordinate, second: Coordinate) -> float:
    """Great-circle distance between two published coordinates, in meters."""
    lat1, lat2 = math.radians(first.latitude), math.radians(second.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second.longitude - first.longitude)
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(math.sqrt(min(1.0, haversine)))


@dataclass(frozen=True, slots=True)
class SnapshotWorld:
    """One corpus shared by episodes, containing public data only."""

    corpus_version: str
    references: tuple[ReferenceAsset, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.corpus_version, str) or not self.corpus_version.strip():
            raise ValueError("corpus_version must be a non-empty string")
        references = tuple(self.references)
        if any(not isinstance(item, ReferenceAsset) for item in references):
            raise ValueError("references must be ReferenceAsset records")
        ids = [item.asset_id for item in references]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot reference IDs must be unique")
        object.__setattr__(self, "references", references)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "references": [item.to_dict() for item in self.references],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SnapshotWorld:
        if not isinstance(value, Mapping) or set(value) != {"corpus_version", "references"}:
            raise ValueError("snapshot world fields do not match the schema")
        if not isinstance(value["references"], list):
            raise ValueError("references must be an array")
        return cls(
            corpus_version=value["corpus_version"],
            references=tuple(ReferenceAsset.from_dict(item) for item in value["references"]),
        )
