"""OSV-5M dataset and reference-corpus access."""

from .corpus import (
    CorpusHit,
    CorpusPage,
    CorpusStore,
    ImageResolver,
    InMemoryCorpusStore,
    MappingImageResolver,
)
from .osv5m import OSV5MDataset, OSV5MSample

__all__ = [name for name in globals() if not name.startswith("_")]
