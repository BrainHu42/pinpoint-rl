from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGeoModel(ABC):
    @abstractmethod
    def predict(self, image_embedding: Any) -> tuple[float, float]:
        """Return predicted (latitude, longitude) from one image embedding."""
