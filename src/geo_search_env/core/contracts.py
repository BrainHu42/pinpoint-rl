"""Strict public and private contracts for the Mapillary baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, ClassVar, Mapping, Sequence

__all__ = [
    "Action",
    "ActionCapability",
    "ActionKind",
    "AssetIdsArgs",
    "BackendCapabilities",
    "BudgetConfig",
    "ContractError",
    "Coordinate",
    "CoverageSummary",
    "EpisodeStatus",
    "EpisodeTrace",
    "FinalSelection",
    "FinishArgs",
    "GroundTruthRecord",
    "InitialCandidate",
    "InspectCoverageArgs",
    "MatchScore",
    "Observation",
    "PublicEpisode",
    "ReferenceAsset",
    "SearchNearArgs",
    "SearchResultCard",
    "StructuredError",
    "ToolResponse",
    "TraceStep",
    "Transition",
]

JsonObject = dict[str, Any]


class ContractError(ValueError):
    """Raised when serialized data violates an experiment contract."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _keys(data: Mapping[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ContractError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be a finite number")
    return result


def _strings(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must be an array of strings")
    result = tuple(_text(item, f"{name} item") for item in value)
    if len(result) < minimum or (maximum is not None and len(result) > maximum):
        raise ContractError(f"{name} has an invalid number of items")
    if len(result) != len(set(result)):
        raise ContractError(f"{name} must not contain duplicates")
    return result


class ActionKind(str, Enum):
    SEARCH_NEAR = "search_near"
    INSPECT_COVERAGE = "inspect_coverage"
    OPEN_RESULTS = "open_results"
    FINISH = "finish"


class EpisodeStatus(str, Enum):
    ACTIVE = "active"
    FINAL_ANSWER_ONLY = "final_answer_only"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class ActionCapability:
    action: ActionKind
    max_radius_m: float | None = None
    max_batch_size: int | None = None
    page_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionKind):
            object.__setattr__(self, "action", ActionKind(self.action))
        if self.max_radius_m is not None:
            radius = _number(self.max_radius_m, "max_radius_m")
            if radius <= 0:
                raise ContractError("max_radius_m must be positive")
            object.__setattr__(self, "max_radius_m", radius)
        for name in ("max_batch_size", "page_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(value, name, minimum=1))

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"action": self.action.value}
        for name in ("max_radius_m", "max_batch_size", "page_size"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, value: Any) -> ActionCapability:
        data = _mapping(value, "action capability")
        _keys(data, {"action"}, {"max_radius_m", "max_batch_size", "page_size"})
        return cls(ActionKind(data["action"]), data.get("max_radius_m"), data.get("max_batch_size"), data.get("page_size"))


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    actions: tuple[ActionCapability, ...]
    evidence_types: tuple[str, ...]
    contract_version: str = "mapillary-baseline-v2"

    def __post_init__(self) -> None:
        _text(self.contract_version, "contract_version")
        actions = tuple(self.actions)
        expected = {
            ActionKind.SEARCH_NEAR,
            ActionKind.INSPECT_COVERAGE,
            ActionKind.OPEN_RESULTS,
            ActionKind.FINISH,
        }
        if {item.action for item in actions} != expected or len(actions) != 4:
            raise ContractError("backend must advertise exactly the experiment actions")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "evidence_types", _strings(self.evidence_types, "evidence_types"))

    @property
    def action_kinds(self) -> frozenset[ActionKind]:
        return frozenset(item.action for item in self.actions)

    def capability(self, kind: ActionKind) -> ActionCapability:
        return next(item for item in self.actions if item.action is kind)

    def to_dict(self) -> JsonObject:
        return {"contract_version": self.contract_version, "actions": [item.to_dict() for item in self.actions], "evidence_types": list(self.evidence_types)}

    @classmethod
    def from_dict(cls, value: Any) -> BackendCapabilities:
        data = _mapping(value, "backend capabilities")
        _keys(data, {"contract_version", "actions", "evidence_types"})
        return cls(tuple(ActionCapability.from_dict(item) for item in data["actions"]), _strings(data["evidence_types"], "evidence_types"), data["contract_version"])


@dataclass(frozen=True, slots=True)
class Coordinate:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        latitude = _number(self.latitude, "latitude")
        longitude = _number(self.longitude, "longitude")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ContractError("coordinate is outside valid bounds")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)

    def to_dict(self) -> JsonObject:
        return {"latitude": self.latitude, "longitude": self.longitude}

    @classmethod
    def from_dict(cls, value: Any) -> Coordinate:
        data = _mapping(value, "coordinate")
        _keys(data, {"latitude", "longitude"})
        return cls(data["latitude"], data["longitude"])


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    credits: int = 32
    max_nonterminal_actions: int = 12
    search_cost: int = 4
    coverage_cost: int = 1
    open_result_cost: int = 2

    def __post_init__(self) -> None:
        for name in ("credits", "max_nonterminal_actions", "search_cost", "coverage_cost", "open_result_cost"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))

    @property
    def minimum_acquisition_cost(self) -> int:
        return min(self.search_cost, self.coverage_cost, self.open_result_cost)

    def to_dict(self) -> JsonObject:
        return {name: getattr(self, name) for name in ("credits", "max_nonterminal_actions", "search_cost", "coverage_cost", "open_result_cost")}

    @classmethod
    def from_dict(cls, value: Any) -> BudgetConfig:
        data = _mapping(value, "budget")
        _keys(data, {"credits", "max_nonterminal_actions", "search_cost", "coverage_cost", "open_result_cost"})
        return cls(**data)


@dataclass(frozen=True, slots=True)
class InitialCandidate:
    candidate_id: str
    coordinate: Coordinate
    rank: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        _integer(self.rank, "rank", minimum=1)
        if self.confidence is not None:
            confidence = _number(self.confidence, "confidence")
            if not 0 <= confidence <= 1:
                raise ContractError("confidence must be in [0, 1]")
            object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"candidate_id": self.candidate_id, "coordinate": self.coordinate.to_dict(), "rank": self.rank}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result

    @classmethod
    def from_dict(cls, value: Any) -> InitialCandidate:
        data = _mapping(value, "initial candidate")
        _keys(data, {"candidate_id", "coordinate", "rank"}, {"confidence"})
        return cls(data["candidate_id"], Coordinate.from_dict(data["coordinate"]), data["rank"], data.get("confidence"))


@dataclass(frozen=True, slots=True)
class PublicEpisode:
    episode_id: str
    query_asset_id: str
    initial_candidates: tuple[InitialCandidate, ...]
    baseline_candidate_id: str
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    upstream_provenance: Mapping[str, str] | None = None

    FORBIDDEN_PUBLIC_KEYS: ClassVar[frozenset[str]] = frozenset({"ground_truth", "ground_truth_coordinate", "query_coordinate", "private_metadata"})

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id")
        _text(self.query_asset_id, "query_asset_id")
        candidates = tuple(self.initial_candidates)
        if not candidates:
            raise ContractError("initial_candidates must not be empty")
        ids = [item.candidate_id for item in candidates]
        ranks = [item.rank for item in candidates]
        if len(ids) != len(set(ids)) or len(ranks) != len(set(ranks)):
            raise ContractError("candidate IDs and ranks must be unique")
        if self.baseline_candidate_id not in ids or candidates[ids.index(self.baseline_candidate_id)].rank != 1:
            raise ContractError("baseline candidate must identify the rank-1 candidate")
        object.__setattr__(self, "initial_candidates", candidates)
        if self.upstream_provenance is not None:
            provenance = _mapping(self.upstream_provenance, "upstream_provenance")
            if self.FORBIDDEN_PUBLIC_KEYS.intersection(provenance):
                raise ContractError("private fields are forbidden in upstream_provenance")
            object.__setattr__(self, "upstream_provenance", {_text(k, "provenance key"): _text(v, "provenance value") for k, v in provenance.items()})

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"episode_id": self.episode_id, "query_asset_id": self.query_asset_id, "initial_candidates": [item.to_dict() for item in self.initial_candidates], "baseline_candidate_id": self.baseline_candidate_id, "budget": self.budget.to_dict()}
        if self.upstream_provenance is not None:
            result["upstream_provenance"] = dict(self.upstream_provenance)
        return result

    @classmethod
    def from_dict(cls, value: Any) -> PublicEpisode:
        data = _mapping(value, "public episode")
        if cls.FORBIDDEN_PUBLIC_KEYS.intersection(data):
            raise ContractError("private fields are forbidden in public episodes")
        _keys(data, {"episode_id", "query_asset_id", "initial_candidates", "baseline_candidate_id", "budget"}, {"upstream_provenance"})
        return cls(data["episode_id"], data["query_asset_id"], tuple(InitialCandidate.from_dict(item) for item in data["initial_candidates"]), data["baseline_candidate_id"], BudgetConfig.from_dict(data["budget"]), data.get("upstream_provenance"))


@dataclass(frozen=True, slots=True)
class ReferenceAsset:
    asset_id: str
    image_handles: tuple[str, ...]
    published_coordinate: Coordinate
    provenance: str
    coordinate_provenance: str
    coordinate_uncertainty_m: float | None = None
    captured_at: str | None = None
    creator: str | None = None
    sequence_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.asset_id, "asset_id")
        object.__setattr__(self, "image_handles", _strings(self.image_handles, "image_handles", minimum=1))
        if not isinstance(self.published_coordinate, Coordinate):
            raise ContractError("published_coordinate must be a coordinate")
        _text(self.provenance, "provenance")
        _text(self.coordinate_provenance, "coordinate_provenance")
        if self.coordinate_uncertainty_m is not None:
            uncertainty = _number(self.coordinate_uncertainty_m, "coordinate_uncertainty_m")
            if uncertainty < 0:
                raise ContractError("coordinate_uncertainty_m must be nonnegative")
            object.__setattr__(self, "coordinate_uncertainty_m", uncertainty)
        for name in ("captured_at", "creator", "sequence_id"):
            _optional_text(getattr(self, name), name)

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"asset_id": self.asset_id, "image_handles": list(self.image_handles), "published_coordinate": self.published_coordinate.to_dict(), "provenance": self.provenance, "coordinate_provenance": self.coordinate_provenance}
        if self.coordinate_uncertainty_m is not None:
            result["coordinate_uncertainty_m"] = self.coordinate_uncertainty_m
        for name in ("captured_at", "creator", "sequence_id"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        return result

    @classmethod
    def from_dict(cls, value: Any) -> ReferenceAsset:
        data = _mapping(value, "reference asset")
        _keys(data, {"asset_id", "image_handles", "published_coordinate", "provenance", "coordinate_provenance"}, {"coordinate_uncertainty_m", "captured_at", "creator", "sequence_id"})
        return cls(data["asset_id"], _strings(data["image_handles"], "image_handles", minimum=1), Coordinate.from_dict(data["published_coordinate"]), data["provenance"], data["coordinate_provenance"], data.get("coordinate_uncertainty_m"), data.get("captured_at"), data.get("creator"), data.get("sequence_id"))


@dataclass(frozen=True, slots=True)
class GroundTruthRecord:
    episode_id: str
    query_coordinate: Coordinate
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id")
        object.__setattr__(self, "audit_metadata", dict(_mapping(self.audit_metadata, "audit_metadata")))

    def to_dict(self) -> JsonObject:
        return {"episode_id": self.episode_id, "query_coordinate": self.query_coordinate.to_dict(), "audit_metadata": dict(self.audit_metadata)}

    @classmethod
    def from_dict(cls, value: Any) -> GroundTruthRecord:
        data = _mapping(value, "ground truth")
        _keys(data, {"episode_id", "query_coordinate", "audit_metadata"})
        return cls(data["episode_id"], Coordinate.from_dict(data["query_coordinate"]), data["audit_metadata"])


@dataclass(frozen=True, slots=True)
class SearchNearArgs:
    anchor_id: str
    radius_m: float
    cursor: str | None = None

    def __post_init__(self) -> None:
        _text(self.anchor_id, "anchor_id")
        radius = _number(self.radius_m, "radius_m")
        if radius <= 0:
            raise ContractError("radius_m must be positive")
        object.__setattr__(self, "radius_m", radius)
        _optional_text(self.cursor, "cursor")

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"anchor_id": self.anchor_id, "radius_m": self.radius_m}
        if self.cursor is not None:
            result["cursor"] = self.cursor
        return result


@dataclass(frozen=True, slots=True)
class InspectCoverageArgs:
    anchor_id: str
    radius_m: float

    def __post_init__(self) -> None:
        _text(self.anchor_id, "anchor_id")
        radius = _number(self.radius_m, "radius_m")
        if radius <= 0:
            raise ContractError("radius_m must be positive")
        object.__setattr__(self, "radius_m", radius)

    def to_dict(self) -> JsonObject:
        return {"anchor_id": self.anchor_id, "radius_m": self.radius_m}


@dataclass(frozen=True, slots=True)
class AssetIdsArgs:
    asset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_ids", _strings(self.asset_ids, "asset_ids", minimum=1, maximum=2))

    def to_dict(self) -> JsonObject:
        return {"asset_ids": list(self.asset_ids)}


@dataclass(frozen=True, slots=True)
class FinishArgs:
    candidate_id: str

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")

    def to_dict(self) -> JsonObject:
        return {"candidate_id": self.candidate_id}


ActionArgs = SearchNearArgs | InspectCoverageArgs | AssetIdsArgs | FinishArgs


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    arguments: ActionArgs

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            object.__setattr__(self, "kind", ActionKind(self.kind))
        expected = {
            ActionKind.SEARCH_NEAR: SearchNearArgs,
            ActionKind.INSPECT_COVERAGE: InspectCoverageArgs,
            ActionKind.OPEN_RESULTS: AssetIdsArgs,
            ActionKind.FINISH: FinishArgs,
        }[self.kind]
        if not isinstance(self.arguments, expected):
            raise ContractError(f"{self.kind.value} has invalid arguments")

    @classmethod
    def search_near(cls, anchor_id: str, radius_m: float, cursor: str | None = None) -> Action:
        return cls(ActionKind.SEARCH_NEAR, SearchNearArgs(anchor_id, radius_m, cursor))

    @classmethod
    def inspect_coverage(cls, anchor_id: str, radius_m: float) -> Action:
        return cls(ActionKind.INSPECT_COVERAGE, InspectCoverageArgs(anchor_id, radius_m))

    @classmethod
    def open_results(cls, asset_ids: Sequence[str]) -> Action:
        return cls(ActionKind.OPEN_RESULTS, AssetIdsArgs(tuple(asset_ids)))

    @classmethod
    def finish(cls, candidate_id: str) -> Action:
        return cls(ActionKind.FINISH, FinishArgs(candidate_id))

    def to_dict(self) -> JsonObject:
        return {"tool": self.kind.value, "arguments": self.arguments.to_dict()}

    @classmethod
    def from_dict(cls, value: Any) -> Action:
        data = _mapping(value, "action")
        _keys(data, {"tool", "arguments"})
        kind = ActionKind(data["tool"])
        args = _mapping(data["arguments"], "action arguments")
        if kind is ActionKind.SEARCH_NEAR:
            _keys(args, {"anchor_id", "radius_m"}, {"cursor"})
            parsed: ActionArgs = SearchNearArgs(args["anchor_id"], args["radius_m"], args.get("cursor"))
        elif kind is ActionKind.INSPECT_COVERAGE:
            _keys(args, {"anchor_id", "radius_m"})
            parsed = InspectCoverageArgs(args["anchor_id"], args["radius_m"])
        elif kind is ActionKind.OPEN_RESULTS:
            _keys(args, {"asset_ids"})
            parsed = AssetIdsArgs(_strings(args["asset_ids"], "asset_ids", minimum=1, maximum=2))
        else:
            _keys(args, {"candidate_id"})
            parsed = FinishArgs(args["candidate_id"])
        return cls(kind, parsed)


@dataclass(frozen=True, slots=True)
class SearchResultCard:
    asset_id: str
    thumbnail_handle: str
    published_coordinate: Coordinate
    distance_from_search_anchor_m: float
    coordinate_provenance: str
    coordinate_uncertainty_m: float | None = None
    captured_at: str | None = None
    creator: str | None = None
    sequence_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.asset_id, "asset_id")
        _text(self.thumbnail_handle, "thumbnail_handle")
        distance = _number(self.distance_from_search_anchor_m, "distance_from_search_anchor_m")
        if distance < 0:
            raise ContractError("distance_from_search_anchor_m must be nonnegative")
        object.__setattr__(self, "distance_from_search_anchor_m", distance)
        _text(self.coordinate_provenance, "coordinate_provenance")
        if self.coordinate_uncertainty_m is not None:
            uncertainty = _number(self.coordinate_uncertainty_m, "coordinate_uncertainty_m")
            if uncertainty < 0:
                raise ContractError("coordinate_uncertainty_m must be nonnegative")
            object.__setattr__(self, "coordinate_uncertainty_m", uncertainty)

    @classmethod
    def from_reference(cls, reference: ReferenceAsset, distance_m: float) -> SearchResultCard:
        return cls(reference.asset_id, reference.image_handles[0], reference.published_coordinate, distance_m, reference.coordinate_provenance, reference.coordinate_uncertainty_m, reference.captured_at, reference.creator, reference.sequence_id)

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"asset_id": self.asset_id, "thumbnail_handle": self.thumbnail_handle, "published_coordinate": self.published_coordinate.to_dict(), "distance_from_search_anchor_m": self.distance_from_search_anchor_m, "coordinate_provenance": self.coordinate_provenance}
        if self.coordinate_uncertainty_m is not None:
            result["coordinate_uncertainty_m"] = self.coordinate_uncertainty_m
        for name in ("captured_at", "creator", "sequence_id"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        return result

    @classmethod
    def from_dict(cls, value: Any) -> SearchResultCard:
        data = _mapping(value, "search result")
        _keys(data, {"asset_id", "thumbnail_handle", "published_coordinate", "distance_from_search_anchor_m", "coordinate_provenance"}, {"coordinate_uncertainty_m", "captured_at", "creator", "sequence_id"})
        return cls(data["asset_id"], data["thumbnail_handle"], Coordinate.from_dict(data["published_coordinate"]), data["distance_from_search_anchor_m"], data["coordinate_provenance"], data.get("coordinate_uncertainty_m"), data.get("captured_at"), data.get("creator"), data.get("sequence_id"))


@dataclass(frozen=True, slots=True)
class MatchScore:
    asset_id: str
    score: float

    def __post_init__(self) -> None:
        _text(self.asset_id, "asset_id")
        score = _number(self.score, "score")
        if not 0 <= score <= 1:
            raise ContractError("match score must be in [0, 1]")
        object.__setattr__(self, "score", score)

    def to_dict(self) -> JsonObject:
        return {"asset_id": self.asset_id, "score": self.score}

    @classmethod
    def from_dict(cls, value: Any) -> MatchScore:
        data = _mapping(value, "match score")
        _keys(data, {"asset_id", "score"})
        return cls(data["asset_id"], data["score"])


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    anchor_id: str
    radius_m: float
    approximate_image_count: int
    approximate_sequence_count: int
    oldest_capture_at: str | None = None
    newest_capture_at: str | None = None
    panorama_fraction: float | None = None
    tiles_queried: int = 0
    is_approximate: bool = True

    def __post_init__(self) -> None:
        _text(self.anchor_id, "anchor_id")
        radius = _number(self.radius_m, "radius_m")
        if radius <= 0:
            raise ContractError("radius_m must be positive")
        object.__setattr__(self, "radius_m", radius)
        for name in ("approximate_image_count", "approximate_sequence_count", "tiles_queried"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("oldest_capture_at", "newest_capture_at"):
            _optional_text(getattr(self, name), name)
        if self.panorama_fraction is not None:
            fraction = _number(self.panorama_fraction, "panorama_fraction")
            if not 0 <= fraction <= 1:
                raise ContractError("panorama_fraction must be in [0, 1]")
            object.__setattr__(self, "panorama_fraction", fraction)
        if type(self.is_approximate) is not bool:
            raise ContractError("is_approximate must be boolean")

    def to_dict(self) -> JsonObject:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> CoverageSummary:
        data = _mapping(value, "coverage summary")
        _keys(data, set(cls.__dataclass_fields__))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ToolResponse:
    tool: ActionKind
    search_results: tuple[SearchResultCard, ...] = ()
    opened_assets: tuple[ReferenceAsset, ...] = ()
    match_scores: tuple[MatchScore, ...] = ()
    coverage_summary: CoverageSummary | None = None
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool, ActionKind):
            object.__setattr__(self, "tool", ActionKind(self.tool))
        object.__setattr__(self, "search_results", tuple(self.search_results))
        object.__setattr__(self, "opened_assets", tuple(self.opened_assets))
        object.__setattr__(self, "match_scores", tuple(self.match_scores))
        if self.coverage_summary is not None and not isinstance(self.coverage_summary, CoverageSummary):
            raise ContractError("coverage_summary must be a coverage summary")
        _optional_text(self.next_cursor, "next_cursor")

    def to_dict(self) -> JsonObject:
        return {"tool": self.tool.value, "search_results": [item.to_dict() for item in self.search_results], "opened_assets": [item.to_dict() for item in self.opened_assets], "match_scores": [item.to_dict() for item in self.match_scores], "coverage_summary": self.coverage_summary.to_dict() if self.coverage_summary else None, "next_cursor": self.next_cursor}

    @classmethod
    def from_dict(cls, value: Any) -> ToolResponse:
        data = _mapping(value, "tool response")
        _keys(data, {"tool", "search_results", "opened_assets", "match_scores", "coverage_summary", "next_cursor"})
        return cls(ActionKind(data["tool"]), tuple(SearchResultCard.from_dict(item) for item in data["search_results"]), tuple(ReferenceAsset.from_dict(item) for item in data["opened_assets"]), tuple(MatchScore.from_dict(item) for item in data["match_scores"]), CoverageSummary.from_dict(data["coverage_summary"]) if data["coverage_summary"] else None, data["next_cursor"])


@dataclass(frozen=True, slots=True)
class StructuredError:
    code: str
    message: str

    def __post_init__(self) -> None:
        _text(self.code, "error code")
        _text(self.message, "error message")

    def to_dict(self) -> JsonObject:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, value: Any) -> StructuredError:
        data = _mapping(value, "error")
        _keys(data, {"code", "message"})
        return cls(data["code"], data["message"])


@dataclass(frozen=True, slots=True)
class FinalSelection:
    candidate_id: str
    coordinate: Coordinate
    used_fallback: bool = False
    failure_code: str | None = None

    def to_dict(self) -> JsonObject:
        return {"candidate_id": self.candidate_id, "coordinate": self.coordinate.to_dict(), "used_fallback": self.used_fallback, "failure_code": self.failure_code}

    @classmethod
    def from_dict(cls, value: Any) -> FinalSelection:
        data = _mapping(value, "final selection")
        _keys(data, {"candidate_id", "coordinate", "used_fallback", "failure_code"})
        return cls(data["candidate_id"], Coordinate.from_dict(data["coordinate"]), data["used_fallback"], data["failure_code"])


@dataclass(frozen=True, slots=True)
class Observation:
    episode_id: str
    status: EpisodeStatus
    remaining_credits: int
    remaining_nonterminal_actions: int
    valid_candidate_ids: tuple[str, ...]
    capabilities: BackendCapabilities
    discovered_assets: tuple[SearchResultCard, ...] = ()
    opened_assets: tuple[ReferenceAsset, ...] = ()
    match_scores: tuple[MatchScore, ...] = ()
    latest_tool_response: ToolResponse | None = None

    def to_dict(self) -> JsonObject:
        return {"episode_id": self.episode_id, "status": self.status.value, "remaining_credits": self.remaining_credits, "remaining_nonterminal_actions": self.remaining_nonterminal_actions, "valid_candidate_ids": list(self.valid_candidate_ids), "capabilities": self.capabilities.to_dict(), "discovered_assets": [item.to_dict() for item in self.discovered_assets], "opened_assets": [item.to_dict() for item in self.opened_assets], "match_scores": [item.to_dict() for item in self.match_scores], "latest_tool_response": self.latest_tool_response.to_dict() if self.latest_tool_response else None}

    @classmethod
    def from_dict(cls, value: Any) -> Observation:
        data = _mapping(value, "observation")
        _keys(data, {"episode_id", "status", "remaining_credits", "remaining_nonterminal_actions", "valid_candidate_ids", "capabilities", "discovered_assets", "opened_assets", "match_scores", "latest_tool_response"})
        return cls(data["episode_id"], EpisodeStatus(data["status"]), data["remaining_credits"], data["remaining_nonterminal_actions"], _strings(data["valid_candidate_ids"], "valid_candidate_ids"), BackendCapabilities.from_dict(data["capabilities"]), tuple(SearchResultCard.from_dict(item) for item in data["discovered_assets"]), tuple(ReferenceAsset.from_dict(item) for item in data["opened_assets"]), tuple(MatchScore.from_dict(item) for item in data["match_scores"]), ToolResponse.from_dict(data["latest_tool_response"]) if data["latest_tool_response"] else None)


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    action_cost: int
    error: StructuredError | None
    termination_state: EpisodeStatus


@dataclass(frozen=True, slots=True)
class TraceStep:
    action: Action
    observation: Observation
    cost: int
    error: StructuredError | None = None

    def to_dict(self) -> JsonObject:
        return {"action": self.action.to_dict(), "observation": self.observation.to_dict(), "cost": self.cost, "error": self.error.to_dict() if self.error else None}

    @classmethod
    def from_dict(cls, value: Any) -> TraceStep:
        data = _mapping(value, "trace step")
        _keys(data, {"action", "observation", "cost", "error"})
        return cls(Action.from_dict(data["action"]), Observation.from_dict(data["observation"]), data["cost"], StructuredError.from_dict(data["error"]) if data["error"] else None)


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    public_episode: PublicEpisode
    initial_observation: Observation
    steps: tuple[TraceStep, ...]
    final_selection: FinalSelection | None

    @property
    def total_cost(self) -> int:
        return sum(item.cost for item in self.steps)

    def to_dict(self) -> JsonObject:
        return {"public_episode": self.public_episode.to_dict(), "initial_observation": self.initial_observation.to_dict(), "steps": [item.to_dict() for item in self.steps], "final_selection": self.final_selection.to_dict() if self.final_selection else None}

    @classmethod
    def from_dict(cls, value: Any) -> EpisodeTrace:
        data = _mapping(value, "episode trace")
        _keys(data, {"public_episode", "initial_observation", "steps", "final_selection"})
        return cls(PublicEpisode.from_dict(data["public_episode"]), Observation.from_dict(data["initial_observation"]), tuple(TraceStep.from_dict(item) for item in data["steps"]), FinalSelection.from_dict(data["final_selection"]) if data["final_selection"] else None)
