"""Frozen matcher boundary and explicitly synthetic fixture implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..core.contracts import ReferenceAsset
from ..data.corpus import ImageResolver


class Matcher(Protocol):
    def score(self, query_asset: str, reference_asset: ReferenceAsset) -> float: ...


class OpenCVSIFTMatcher:
    """Frozen local-feature matcher over acquired local photos, never coordinates.

    Features are calculated lazily: construction and searches do not inspect
    reference image pixels. ``score`` is called only after an open action and
    The score is computed only after an ``open_results`` action.
    """

    def __init__(
        self,
        query_images: Mapping[str, Path] | None = None,
        reference_images: Mapping[str, Path] | None = None,
        *,
        image_resolver: ImageResolver | None = None,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError("real matching requires pip install '.[real]'") from error
        self._cv2 = cv2
        self._np = np
        if image_resolver is not None:
            if query_images is not None or reference_images is not None:
                raise ValueError("use either image mappings or image_resolver, not both")
            if not isinstance(image_resolver, ImageResolver):
                raise TypeError("image_resolver must implement ImageResolver")
            self._queries: dict[str, Path] = {}
            self._references: dict[str, Path] = {}
        else:
            if query_images is None or reference_images is None:
                raise ValueError("query_images and reference_images are both required")
            self._queries = {asset_id: Path(path) for asset_id, path in query_images.items()}
            self._references = {
                asset_id: Path(path) for asset_id, path in reference_images.items()
            }
        self._image_resolver = image_resolver
        self.version = f"sift-l2-ransac-v1-opencv-{cv2.__version__}"
        self._sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._feature_cache: dict[Path, tuple[Any, Any]] = {}
        self._pair_cache: dict[tuple[str, str], tuple[float, int, float]] = {}

    @classmethod
    def from_resolver(cls, image_resolver: ImageResolver) -> OpenCVSIFTMatcher:
        """Construct a matcher that resolves opaque IDs only when acquired."""
        return cls(image_resolver=image_resolver)

    def _query_path(self, asset_id: str) -> Path:
        if self._image_resolver is not None:
            return Path(self._image_resolver.resolve_query_image(asset_id))
        try:
            return self._queries[asset_id]
        except KeyError as error:
            raise ValueError("unknown real query image asset ID") from error

    def _reference_path(self, asset_id: str) -> Path:
        if self._image_resolver is not None:
            return Path(self._image_resolver.resolve_reference_image(asset_id))
        try:
            return self._references[asset_id]
        except KeyError as error:
            raise ValueError("unknown real reference image asset ID") from error

    def _features(self, path: Path):
        if path in self._feature_cache:
            return self._feature_cache[path]
        gray = self._cv2.imread(str(path), self._cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot decode image: {path}")
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        self._feature_cache[path] = (keypoints, descriptors)
        return self._feature_cache[path]

    def _pair_evidence(self, query_asset: str, reference_id: str) -> tuple[float, int, float]:
        key = (query_asset, reference_id)
        if key in self._pair_cache:
            return self._pair_cache[key]
        query_path = self._query_path(query_asset)
        reference_path = self._reference_path(reference_id)
        query_points, query_descriptors = self._features(query_path)
        reference_points, reference_descriptors = self._features(reference_path)
        if query_descriptors is None or reference_descriptors is None:
            self._pair_cache[key] = (0.0, 0, 0.0)
            return self._pair_cache[key]
        forward = self._directional_evidence(query_points, query_descriptors, reference_points, reference_descriptors)
        reverse = self._directional_evidence(reference_points, reference_descriptors, query_points, query_descriptors)
        self._pair_cache[key] = max(forward, reverse)
        return self._pair_cache[key]

    def _directional_evidence(self, source_points, source_descriptors, destination_points, destination_descriptors) -> tuple[float, int, float]:
        if len(destination_descriptors) < 2:
            return 0.0, 0, 0.0
        pairs = self._matcher.knnMatch(source_descriptors, destination_descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good) < 4:
            return 0.0, 0, 0.0
        source = self._np.float32([source_points[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        destination = self._np.float32([destination_points[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        self._cv2.setRNGSeed(0)
        try:
            _, mask = self._cv2.findHomography(source, destination, self._cv2.RANSAC, 4.0)
        except self._cv2.error:
            return 0.0, 0, 0.0
        inliers = int(mask.sum()) if mask is not None else 0
        ratio = inliers / len(good)
        score = min(1.0, inliers / 12.0) * min(1.0, ratio / 0.2)
        return float(score), inliers, float(ratio)

    def score(self, query_asset: str, reference_asset: ReferenceAsset) -> float:
        return self._pair_evidence(query_asset, reference_asset.asset_id)[0]

class SyntheticFixtureMatcher:
    """Fixed query/reference outputs; never derives scores from coordinates."""

    def __init__(
        self,
        scores: Mapping[tuple[str, str], float],
        *,
        version: str = "synthetic-fixture-v1",
        default_score: float | None = None,
    ) -> None:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("matcher version must be a non-empty string")
        self.version = version
        self._scores = dict(scores)
        self._default_score = self._normalized_score(default_score) if default_score is not None else None
        for key, score in self._scores.items():
            self._validate_key(key)
            self._scores[key] = self._normalized_score(score)

    @staticmethod
    def _validate_key(key: Any) -> None:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(item, str) or not item.strip() for item in key)
        ):
            raise ValueError("matcher keys must be (query_asset_id, reference_asset_id)")

    @staticmethod
    def _normalized_score(value: Any) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("synthetic matcher scores must be finite and in [0, 1]")
        return float(value)

    def score(self, query_asset: str, reference_asset: ReferenceAsset) -> float:
        value = self._scores.get((query_asset, reference_asset.asset_id), self._default_score)
        if value is None:
            raise ValueError("missing synthetic match score")
        return value

    @classmethod
    def from_dict(cls, value: Any) -> SyntheticFixtureMatcher:
        """Load fixed query/reference match scores."""
        if not isinstance(value, Mapping) or set(value) != {
            "matcher_version", "default_score", "pairs"
        }:
            raise ValueError("synthetic matcher fields do not match the schema")
        if not isinstance(value["pairs"], list):
            raise ValueError("pairs must be an array")
        scores: dict[tuple[str, str], float] = {}
        for pair in value["pairs"]:
            if not isinstance(pair, Mapping) or set(pair) != {
                "query_asset_id", "reference_asset_id", "score"
            }:
                raise ValueError("invalid synthetic matcher pair")
            key = (pair["query_asset_id"], pair["reference_asset_id"])
            if key in scores:
                raise ValueError("duplicate synthetic matcher pair")
            scores[key] = pair["score"]
        return cls(
            scores,
            version=value["matcher_version"],
            default_score=value["default_score"],
        )
