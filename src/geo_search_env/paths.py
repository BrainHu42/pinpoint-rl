"""Project-local storage defaults shared by data loaders and experiments."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/pinpoint"
HF_HOME = Path(os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/hf"))).expanduser()
HF_HUB_CACHE = Path(os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))).expanduser()
MP16_ROOT = HF_HOME / "datasets/MP16-Pro"
OSV_ROOT = HF_HOME / "datasets/osv5m"
PINPOINT_ROOT = Path(os.environ.get("PINPOINT_ROOT", str(DATA_ROOT / "pinpoint-submission"))).expanduser()
