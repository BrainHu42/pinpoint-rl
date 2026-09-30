"""Bounded live Mapillary adapter for transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ..core.backend import ToolBackendError
from ..core.contracts import (
    Action,
    ActionCapability,
    ActionKind,
    BackendCapabilities,
    Coordinate,
    CoverageSummary,
    MatchScore,
    PublicEpisode,
    ReferenceAsset,
    SearchResultCard,
    StructuredError,
    ToolResponse,
)
from ..core.environment import SearchEnvironment
from ..core.geography import distance_m
from ..models.matching import Matcher

MAPILLARY_ENDPOINT = "https://graph.mapillary.com/images"
MAPILLARY_TILE_ENDPOINT = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
MAPILLARY_FIELDS = "id,computed_geometry,thumb_256_url,thumb_1024_url,captured_at,creator,sequence"


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


class HttpTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float],
        headers: Mapping[str, str],
        timeout_s: float,
        max_bytes: int,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    """GET-only transport with explicit response and redirect bounds."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float],
        headers: Mapping[str, str],
        timeout_s: float,
        max_bytes: int,
    ) -> HttpResponse:
        target = f"{url}?{urlencode(params)}" if params else url
        request = Request(target, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout_s) as response:
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise ValueError("HTTP response exceeds configured byte limit")
                return HttpResponse(response.status, body, dict(response.headers.items()))
        except HTTPError as error:
            body = error.read(max_bytes + 1)
            return HttpResponse(error.code, body[:max_bytes], dict(error.headers.items()))
        except URLError as error:
            raise OSError("HTTP transport failed") from error


class ReferenceAcquirer(Protocol):
    def acquire_reference(self, asset_id: str, source_url: str) -> Path: ...


class TemporaryImageResolver:
    """Ephemeral allowlisted cache for opened live images."""

    _ALLOWED_SUFFIXES = (".fbcdn.net", ".fbsbx.com", ".mapillary.com")

    def __init__(
        self,
        query_images: Mapping[str, Path],
        transport: HttpTransport,
        *,
        user_agent: str,
        timeout_s: float = 10.0,
        max_image_bytes: int = 12_000_000,
    ) -> None:
        _validate_user_agent(user_agent)
        self._queries = {key: Path(path).resolve() for key, path in query_images.items()}
        if any(not path.is_file() for path in self._queries.values()):
            raise ValueError("all live query images must exist")
        self._transport = transport
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._max_image_bytes = max_image_bytes
        self._temporary = tempfile.TemporaryDirectory(prefix="mapillary-live-")
        self._references: dict[str, Path] = {}

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> TemporaryImageResolver:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def resolve_query_image(self, asset_id: str) -> Path:
        try:
            return self._queries[asset_id]
        except KeyError as error:
            raise ValueError("unknown live query asset") from error

    def resolve_reference_image(self, asset_id: str) -> Path:
        try:
            return self._references[asset_id]
        except KeyError as error:
            raise ValueError("live reference has not been opened") from error

    def acquire_reference(self, asset_id: str, source_url: str) -> Path:
        if asset_id in self._references:
            return self._references[asset_id]
        parsed = urlparse(source_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(host.endswith(suffix) for suffix in self._ALLOWED_SUFFIXES):
            raise ToolBackendError("provider_asset_unavailable", "Mapillary image URL is not allowlisted")
        response = self._transport.get(
            source_url,
            params={},
            headers={"User-Agent": self._user_agent},
            timeout_s=self._timeout_s,
            max_bytes=self._max_image_bytes,
        )
        if response.status != 200 or not response.body:
            raise ToolBackendError("provider_asset_unavailable", "Mapillary image could not be acquired")
        path = Path(self._temporary.name) / f"{hashlib.sha256(asset_id.encode()).hexdigest()}.jpg"
        path.write_bytes(response.body)
        self._references[asset_id] = path
        return path


@dataclass(frozen=True, slots=True)
class _LiveAsset:
    source_id: str
    reference: ReferenceAsset
    image_url: str


@dataclass(frozen=True, slots=True)
class _ProviderPage:
    assets: tuple[_LiveAsset, ...]
    next_after: str | None


class _RateGate:
    def __init__(self, minimum_interval_s: float, *, clock: Callable[[], float], sleeper: Callable[[float], None]) -> None:
        self._interval = float(minimum_interval_s)
        self._clock = clock
        self._sleeper = sleeper
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self._interval - (now - self._last)
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
        self._last = now


def _validate_user_agent(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or ("@" not in value and "https://" not in value):
        raise ValueError("live user agent must include a contact email or HTTPS project URL")


def _provider_error(status: int) -> ToolBackendError:
    if status in (401, 403):
        return ToolBackendError("provider_authentication_failed", "Mapillary rejected authentication")
    if status == 429:
        return ToolBackendError("provider_rate_limited", "Mapillary rate limit was reached")
    if status == 404:
        return ToolBackendError("provider_asset_unavailable", "Mapillary resource is unavailable")
    return ToolBackendError("provider_http_error", f"Mapillary returned HTTP {status}")


def _json_response(response: HttpResponse) -> Mapping[str, Any]:
    if response.status != 200:
        raise _provider_error(response.status)
    try:
        value = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToolBackendError("provider_malformed_response", "Mapillary returned invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ToolBackendError("provider_malformed_response", "Mapillary response must be an object")
    return value


def _coordinate(latitude: Any, longitude: Any) -> Coordinate | None:
    try:
        return Coordinate(latitude, longitude)
    except (TypeError, ValueError):
        return None


def _optional_api_text(value: Any, *, limit: int = 512) -> str | None:
    return value[:limit] if isinstance(value, str) and value.strip() else None


def _radius_bbox(anchor: Coordinate, radius_m: float) -> tuple[float, float, float, float]:
    latitude_delta = math.degrees(radius_m / 6_371_000.0)
    longitude_delta = latitude_delta / max(0.01, math.cos(math.radians(anchor.latitude)))
    west, east = anchor.longitude - longitude_delta, anchor.longitude + longitude_delta
    if west < -180 or east > 180:
        raise ToolBackendError("unsupported_geographic_window", "antimeridian searches are not supported", charge_attempt=False)
    return west, max(-90.0, anchor.latitude - latitude_delta), east, min(90.0, anchor.latitude + latitude_delta)


def _tile_x(longitude: float, zoom: int) -> int:
    count = 1 << zoom
    return min(count - 1, max(0, int(math.floor((longitude + 180.0) / 360.0 * count))))


def _tile_y(latitude: float, zoom: int) -> int:
    latitude = min(85.05112878, max(-85.05112878, latitude))
    radians = math.radians(latitude)
    count = 1 << zoom
    value = (1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * count
    return min(count - 1, max(0, int(math.floor(value))))


def _coverage_tiles(anchor: Coordinate, radius_m: float, zoom: int) -> tuple[tuple[int, int], ...]:
    west, south, east, north = _radius_bbox(anchor, radius_m)
    return tuple(
        (x, y)
        for x in range(_tile_x(west, zoom), _tile_x(east, zoom) + 1)
        for y in range(_tile_y(north, zoom), _tile_y(south, zoom) + 1)
    )


def _map_tile_coordinate(value: Any, *, zoom: int, tile_x: int, tile_y: int, extent: int) -> Any:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value[:2])
    ):
        count = 1 << zoom
        world_x = tile_x + float(value[0]) / extent
        world_y = tile_y + float(value[1]) / extent
        longitude = world_x / count * 360.0 - 180.0
        latitude = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * world_y / count))))
        return [longitude, latitude, *value[2:]]
    if isinstance(value, (list, tuple)):
        return [_map_tile_coordinate(item, zoom=zoom, tile_x=tile_x, tile_y=tile_y, extent=extent) for item in value]
    return value


def _decode_mapillary_tile(body: bytes, zoom: int, tile_x: int, tile_y: int) -> Mapping[str, Any]:
    try:
        import mapbox_vector_tile
    except ImportError as error:
        raise RuntimeError("mapbox-vector-tile dependency is unavailable") from error
    decoded = mapbox_vector_tile.decode(body, default_options={"y_coord_down": True})
    if not isinstance(decoded, Mapping):
        raise ValueError("decoded vector tile is not an object")
    result: dict[str, Any] = {}
    for name, layer in decoded.items():
        if not isinstance(layer, Mapping):
            continue
        extent = layer.get("extent", 4096)
        if type(extent) is not int or extent < 1:
            raise ValueError("decoded vector tile has an invalid extent")
        features = []
        for feature in layer.get("features", ()):
            if not isinstance(feature, Mapping):
                continue
            copied = dict(feature)
            geometry = copied.get("geometry")
            if isinstance(geometry, Mapping):
                copied_geometry = dict(geometry)
                copied_geometry["coordinates"] = _map_tile_coordinate(
                    geometry.get("coordinates"), zoom=zoom, tile_x=tile_x, tile_y=tile_y, extent=extent
                )
                copied["geometry"] = copied_geometry
            features.append(copied)
        result[str(name)] = {"features": features}
    return result


def _points(value: Any):
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value[:2])
    ):
        yield Coordinate(value[1], value[0])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _points(item)


def _layer_features(tile: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], ...]:
    layer = tile.get(name)
    features = layer.get("features") if isinstance(layer, Mapping) else None
    if not isinstance(features, (list, tuple)):
        return ()
    return tuple(item for item in features if isinstance(item, Mapping))


def _feature_id(feature: Mapping[str, Any], fallback: str) -> str:
    properties = feature.get("properties")
    value = properties.get("id") if isinstance(properties, Mapping) else None
    if value is None:
        value = feature.get("id")
    text = str(value).strip() if value is not None else ""
    return text or fallback


class MapillaryApiClient:
    def __init__(
        self,
        transport: HttpTransport,
        access_token: str,
        *,
        user_agent: str,
        timeout_s: float = 10.0,
        minimum_interval_s: float = 0.25,
        rate_clock: Callable[[], float] = time.monotonic,
        rate_sleeper: Callable[[float], None] = time.sleep,
        tile_decoder: Callable[[bytes, int, int, int], Mapping[str, Any]] = _decode_mapillary_tile,
        coverage_zoom: int = 14,
        max_coverage_tiles: int = 64,
    ) -> None:
        _validate_user_agent(user_agent)
        if not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("Mapillary access token must be nonempty")
        if type(coverage_zoom) is not int or not 6 <= coverage_zoom <= 16:
            raise ValueError("coverage_zoom must be between 6 and 16")
        if type(max_coverage_tiles) is not int or not 1 <= max_coverage_tiles <= 256:
            raise ValueError("max_coverage_tiles must be between 1 and 256")
        self._transport = transport
        self._access_token = access_token
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._rate = _RateGate(minimum_interval_s, clock=rate_clock, sleeper=rate_sleeper)
        self._tile_decoder = tile_decoder
        self._coverage_zoom = coverage_zoom
        self._max_coverage_tiles = max_coverage_tiles

    def __repr__(self) -> str:
        return "MapillaryApiClient(access_token=<redacted>)"

    def search_near(self, anchor: Coordinate, radius_m: float, *, after: str | None, limit: int) -> _ProviderPage:
        parameters: dict[str, str | int | float] = {
            "bbox": ",".join(f"{value:.8f}" for value in _radius_bbox(anchor, radius_m)),
            "fields": MAPILLARY_FIELDS,
            "limit": limit,
        }
        if after is not None:
            parameters["after"] = after
        self._rate.wait()
        try:
            response = self._transport.get(
                MAPILLARY_ENDPOINT,
                params=parameters,
                headers={"Authorization": f"OAuth {self._access_token}", "User-Agent": self._user_agent},
                timeout_s=self._timeout_s,
                max_bytes=2_000_000,
            )
        except (OSError, TimeoutError, ValueError) as error:
            raise ToolBackendError("provider_transport_error", "Mapillary request failed within its bounds") from error
        value = _json_response(response)
        if not isinstance(value.get("data"), list):
            raise ToolBackendError("provider_malformed_response", "Mapillary response has no image page")
        assets = tuple(
            asset
            for item in value["data"]
            if isinstance(item, Mapping)
            for asset in (self._asset(item),)
            if asset is not None and distance_m(anchor, asset.reference.published_coordinate) <= radius_m
        )
        paging = value.get("paging", {})
        if not isinstance(paging, Mapping):
            raise ToolBackendError("provider_malformed_response", "Mapillary pagination is malformed")
        next_after = None
        if paging.get("next") is not None:
            cursors = paging.get("cursors")
            if not isinstance(cursors, Mapping) or not isinstance(cursors.get("after"), str):
                raise ToolBackendError("provider_malformed_response", "Mapillary next page has no safe cursor")
            next_after = cursors["after"]
        return _ProviderPage(assets, next_after)

    def inspect_coverage(
        self, anchor_id: str, anchor: Coordinate, radius_m: float
    ) -> CoverageSummary:
        tiles = _coverage_tiles(anchor, radius_m, self._coverage_zoom)
        if len(tiles) > self._max_coverage_tiles:
            raise ToolBackendError(
                "coverage_window_too_large",
                "coverage window exceeds the configured vector-tile bound",
                charge_attempt=False,
            )
        images: dict[str, tuple[str | None, bool | None]] = {}
        sequences: set[str] = set()
        for tile_x, tile_y in tiles:
            self._rate.wait()
            try:
                response = self._transport.get(
                    MAPILLARY_TILE_ENDPOINT.format(z=self._coverage_zoom, x=tile_x, y=tile_y),
                    params={"access_token": self._access_token},
                    headers={"User-Agent": self._user_agent},
                    timeout_s=self._timeout_s,
                    max_bytes=5_000_000,
                )
            except (OSError, TimeoutError, ValueError) as error:
                raise ToolBackendError("provider_transport_error", "Mapillary coverage request failed within its bounds") from error
            if response.status != 200:
                raise _provider_error(response.status)
            try:
                decoded = self._tile_decoder(response.body, self._coverage_zoom, tile_x, tile_y)
            except (RuntimeError, TypeError, ValueError) as error:
                raise ToolBackendError("provider_malformed_response", "Mapillary returned an invalid coverage tile") from error
            if not isinstance(decoded, Mapping):
                raise ToolBackendError("provider_malformed_response", "Mapillary coverage tile is not an object")

            tile_label = f"{self._coverage_zoom}/{tile_x}/{tile_y}"
            for index, feature in enumerate(_layer_features(decoded, "image")):
                geometry = feature.get("geometry")
                coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
                if not any(distance_m(anchor, point) <= radius_m for point in _points(coordinates)):
                    continue
                properties = feature.get("properties")
                properties = properties if isinstance(properties, Mapping) else {}
                image_id = _feature_id(feature, f"anonymous-image:{tile_label}:{index}")
                captured = properties.get("captured_at")
                captured_at = str(captured) if isinstance(captured, (int, str)) else None
                is_pano_value = properties.get("is_pano")
                is_pano = bool(is_pano_value) if isinstance(is_pano_value, (bool, int)) else None
                images.setdefault(image_id, (captured_at, is_pano))
                sequence_value = properties.get("sequence_id", properties.get("sequence"))
                if sequence_value is not None and str(sequence_value).strip():
                    sequences.add(str(sequence_value))

            for index, feature in enumerate(_layer_features(decoded, "sequence")):
                geometry = feature.get("geometry")
                coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
                if any(distance_m(anchor, point) <= radius_m for point in _points(coordinates)):
                    sequences.add(_feature_id(feature, f"anonymous-sequence:{tile_label}:{index}"))

        captures = sorted(value[0] for value in images.values() if value[0] is not None)
        panorama_values = [value[1] for value in images.values() if value[1] is not None]
        return CoverageSummary(
            anchor_id=anchor_id,
            radius_m=radius_m,
            approximate_image_count=len(images),
            approximate_sequence_count=len(sequences),
            oldest_capture_at=captures[0] if captures else None,
            newest_capture_at=captures[-1] if captures else None,
            panorama_fraction=(sum(panorama_values) / len(panorama_values)) if panorama_values else None,
            tiles_queried=len(tiles),
            is_approximate=True,
        )

    @staticmethod
    def _asset(item: Mapping[str, Any]) -> _LiveAsset | None:
        source_id = str(item.get("id", "")).strip()
        geometry = item.get("computed_geometry")
        coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return None
        coordinate = _coordinate(coordinates[1], coordinates[0])
        thumbnail = _optional_api_text(item.get("thumb_256_url"), limit=4096)
        image_url = _optional_api_text(item.get("thumb_1024_url"), limit=4096) or thumbnail
        if not source_id or coordinate is None or thumbnail is None or image_url is None:
            return None
        creator_value = item.get("creator")
        creator = _optional_api_text(creator_value.get("username") or creator_value.get("name")) if isinstance(creator_value, Mapping) else None
        sequence_value = item.get("sequence")
        sequence_id = (
            _optional_api_text(sequence_value.get("id"))
            if isinstance(sequence_value, Mapping)
            else _optional_api_text(sequence_value)
        )
        captured = item.get("captured_at")
        captured_at = str(captured) if isinstance(captured, (int, str)) else None
        asset_id = "mapillary:" + hashlib.sha256(source_id.encode()).hexdigest()[:24]
        reference = ReferenceAsset(
            asset_id,
            (f"live-thumbnail:{asset_id}", f"live-image:{asset_id}"),
            coordinate,
            "live-mapillary-api-v4",
            "provider-computed-camera-geometry",
            None,
            captured_at,
            creator,
            sequence_id,
        )
        return _LiveAsset(source_id, reference, image_url)


class _CursorRegistry:
    def __init__(self) -> None:
        self._values: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _signature(request: Mapping[str, Any]) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def issue(self, request: Mapping[str, Any], after: str) -> str:
        signature = self._signature(request)
        cursor = "mapillary-v1:" + hashlib.sha256(f"{signature}\0{after}".encode()).hexdigest()[:24]
        self._values[cursor] = (signature, after)
        return cursor

    def resolve(self, cursor: str | None, request: Mapping[str, Any]) -> str | None:
        if cursor is None:
            return None
        value = self._values.get(cursor)
        if value is None or value[0] != self._signature(request):
            raise ToolBackendError("invalid_cursor", "cursor does not belong to this Mapillary search")
        return value[1]


class LiveMapillaryTools:
    """The baseline ToolBackend implemented by bounded live Mapillary calls."""

    def __init__(
        self,
        client: MapillaryApiClient,
        matcher: Matcher,
        acquirer: ReferenceAcquirer,
        *,
        page_size: int = 8,
        max_radius_m: float = 5_000.0,
    ) -> None:
        if type(page_size) is not int or not 1 <= page_size <= 8:
            raise ValueError("page_size must be between one and eight")
        if not 0 < max_radius_m <= 25_000:
            raise ValueError("max_radius_m must be in (0, 25000]")
        self.client = client
        self.matcher = matcher
        self.acquirer = acquirer
        self._page_size = page_size
        self._max_radius_m = float(max_radius_m)
        self._assets: dict[str, _LiveAsset] = {}
        self._cursors = _CursorRegistry()

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

    def _anchor(self, episode: PublicEpisode, anchor_id: str) -> Coordinate:
        initial = {item.candidate_id: item.coordinate for item in episode.initial_candidates}
        if anchor_id in initial:
            return initial[anchor_id]
        asset = self._assets.get(anchor_id)
        if asset is None:
            raise ToolBackendError("invalid_anchor", "anchor has no public camera coordinate", charge_attempt=False)
        return asset.reference.published_coordinate

    def search_near(self, episode: PublicEpisode, anchor_id: str, radius_m: float, cursor: str | None) -> ToolResponse:
        anchor = self._anchor(episode, anchor_id)
        request = {"anchor_id": anchor_id, "anchor": anchor.to_dict(), "radius_m": radius_m}
        page = self.client.search_near(anchor, radius_m, after=self._cursors.resolve(cursor, request), limit=self._page_size)
        for asset in page.assets:
            self._assets[asset.reference.asset_id] = asset
        next_cursor = self._cursors.issue(request, page.next_after) if page.next_after else None
        return ToolResponse(
            ActionKind.SEARCH_NEAR,
            search_results=tuple(SearchResultCard.from_reference(asset.reference, distance_m(anchor, asset.reference.published_coordinate)) for asset in page.assets),
            next_cursor=next_cursor,
        )

    def inspect_coverage(
        self, episode: PublicEpisode, anchor_id: str, radius_m: float
    ) -> ToolResponse:
        anchor = self._anchor(episode, anchor_id)
        return ToolResponse(
            ActionKind.INSPECT_COVERAGE,
            coverage_summary=self.client.inspect_coverage(anchor_id, anchor, radius_m),
        )

    def open_results(self, episode: PublicEpisode, asset_ids: Sequence[str]) -> ToolResponse:
        try:
            assets = tuple(self._assets[asset_id] for asset_id in asset_ids)
        except KeyError as error:
            raise ToolBackendError("provider_asset_unavailable", "Mapillary asset is unavailable") from error
        scores = []
        for asset in assets:
            self.acquirer.acquire_reference(asset.reference.asset_id, asset.image_url)
            scores.append(MatchScore(asset.reference.asset_id, self.matcher.score(episode.query_asset_id, asset.reference)))
        return ToolResponse(ActionKind.OPEN_RESULTS, opened_assets=tuple(asset.reference for asset in assets), match_scores=tuple(scores))


@dataclass(frozen=True, slots=True)
class LiveProbe:
    captured_at: str
    episode: Mapping[str, Any]
    steps: tuple[Mapping[str, Any], ...]
    api_contract_version: str = "mapillary-baseline-v2"

    def to_dict(self) -> dict[str, Any]:
        return {"api_contract_version": self.api_contract_version, "captured_at": self.captured_at, "episode": dict(self.episode), "steps": [dict(item) for item in self.steps]}


def capture_live_probe(
    episode: PublicEpisode,
    actions: Sequence[Action],
    tools: LiveMapillaryTools,
    *,
    captured_at: str | None = None,
) -> LiveProbe:
    """Execute a bounded, predeclared public probe without private labels."""
    if len(actions) > episode.budget.max_nonterminal_actions + 1:
        raise ValueError("probe exceeds the episode decision bound")
    environment = SearchEnvironment(tools)
    environment.reset(episode)
    records = []
    for action in actions:
        transition = environment.step(action)
        response = transition.observation.latest_tool_response
        records.append({"action": action.to_dict(), "cost": transition.action_cost, "error": transition.error.to_dict() if transition.error else None, "response": response.to_dict() if response else None, "termination_state": transition.termination_state.value})
        if transition.termination_state is not None and transition.termination_state.value == "finished":
            break
    return LiveProbe(captured_at or datetime.now(timezone.utc).isoformat(), episode.to_dict(), tuple(records))


def write_live_probe(path: Path, probe: LiveProbe) -> None:
    if path.exists():
        raise FileExistsError("live probe output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(probe.to_dict(), sort_keys=True, indent=2) + "\n")
