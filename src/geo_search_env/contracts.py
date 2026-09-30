"""Compatibility imports for contracts now organized under :mod:`geo_search_env.core`."""

from .core.contracts import *
from .core.contracts import ContractError

__all__ = [name for name in globals() if not name.startswith("_")]
