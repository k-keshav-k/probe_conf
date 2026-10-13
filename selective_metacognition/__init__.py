"""Selective MLP retrieval for metacognitive confidence (toy experiments).

Implements the E0/E1/E2 stages of
``report/SELECTIVE_MLP_METACOGNITION_PROPOSAL.md`` on a controlled toy world
where relevant and irrelevant relation components are known by construction.
"""

from .config import CONFIDENCE_GRID, CONDITIONS, REPORT_CONDITIONS, ToyConfig
from .model import Selector
from .world import World

__all__ = [
    "CONFIDENCE_GRID",
    "CONDITIONS",
    "REPORT_CONDITIONS",
    "ToyConfig",
    "Selector",
    "World",
]
