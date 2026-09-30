"""Public API for the Mapillary-first geolocation search experiment."""

from .core.contracts import (
    Action,
    ActionCapability,
    ActionKind,
    BackendCapabilities,
    BudgetConfig,
    Coordinate,
    CoverageSummary,
    EpisodeStatus,
    EpisodeTrace,
    GroundTruthRecord,
    InitialCandidate,
    MatchScore,
    Observation,
    PublicEpisode,
    ReferenceAsset,
    Transition,
)
from .core.environment import SearchEnvironment
from .core.geography import SnapshotWorld
from .data.corpus import CorpusHit, CorpusPage, CorpusStore, ImageResolver, InMemoryCorpusStore, MappingImageResolver
from .data.osv5m import OSV5MDataset, OSV5MSample
from .backends.mapillary import (
    HttpResponse,
    HttpTransport,
    LiveMapillaryTools,
    LiveProbe,
    MapillaryApiClient,
    ReferenceAcquirer,
    TemporaryImageResolver,
    UrllibHttpTransport,
    capture_live_probe,
    write_live_probe,
)
from .models.matching import Matcher, OpenCVSIFTMatcher, SyntheticFixtureMatcher
from .models.pinpoint import (
    PinpointImageEmbedder,
    PinpointRetrievalBaseline,
    RetrievalCandidate,
)
from .experiment.policies import AdaptiveSearchPolicy, BaselineOnlyPolicy, RoundRobinSearchPolicy
from .experiment.scoring import RewardConfig, ScoreResult, score_episode
from .backends.osv_simulator import OSVSimulatorTools
from .backends.fixture import LocalSnapshotTools

__all__ = [name for name in globals() if not name.startswith("_")]
