"""
Feature trait modules — reusable feature bundles for leagues with specific structures.

Traits are opt-in: a league enables them via YAML config (e.g., extra_features.stats).
"""

from sportscore.features.traits.conference import (
    CONFERENCE_HANDLERS,
    CONFERENCE_STAT_DEFINITIONS,
)

__all__ = ["CONFERENCE_HANDLERS", "CONFERENCE_STAT_DEFINITIONS"]
