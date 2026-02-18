"""
Base Feature Registry - Framework for sport-specific feature definitions.

Provides the schema and base class that sport apps extend with their own
stat definitions. Basketball defines PER, ELO, pace, etc. Hockey defines
Corsi, Fenwick, xG, save%, etc. The framework is the same.

Sport apps subclass BaseFeatureRegistry and populate STAT_DEFINITIONS
with their sport-specific stats.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


class StatCategory(Enum):
    """Categories of stats based on computation method."""
    BASIC = "basic"           # Simple counting stats (goals, assists, points)
    RATE = "rate"             # Efficiency stats computed from ratios (save%, shooting%)
    NET = "net"               # Opponent-adjusted stats
    DERIVED = "derived"       # Computed from other stats (Corsi%, pace_interaction)
    SPECIAL = "special"       # Special handling required (ELO, injuries)


class CalcWeight(Enum):
    """Valid calculation weight methods."""
    RAW = "raw"                    # Raw aggregation
    AVG = "avg"                    # Per-game average
    WEIGHTED_MPG = "weighted_MPG"  # Weighted by minutes/ice-time played
    HARMONIC_MEAN = "harmonic_mean"
    DERIVED = "derived"


class Perspective(Enum):
    """Valid perspectives for feature computation."""
    DIFF = "diff"    # Home minus away differential
    HOME = "home"    # Absolute home team value
    AWAY = "away"    # Absolute away team value
    NONE = "none"    # Matchup-level (not split by team)


@dataclass
class StatDefinition:
    """Definition of a single statistic with all its metadata."""
    name: str                          # Canonical stat name (e.g., "goals", "corsi")
    category: StatCategory             # Computation category
    db_field: Optional[str] = None     # MongoDB field name (if different from name)
    description: str = ""              # Human-readable description
    supports_side_split: bool = False  # Can be computed for home/away sides separately
    supports_net: bool = False         # Has opponent-adjusted version (*_net)
    valid_calc_weights: Set[str] = field(default_factory=lambda: {"raw", "avg"})
    valid_time_periods: Set[str] = field(default_factory=set)  # Empty = all periods valid
    valid_perspectives: Set[str] = field(default_factory=set)  # Empty = all perspectives valid
    requires_aggregation: bool = False  # For rate stats: aggregate before computing

    # YAML-driven computation fields
    formula: Optional[str] = None              # e.g. "team.points - opponent.points"
    db_fields: Optional[List[str]] = None      # e.g. ["points"] for simple field lookup
    numerator: Optional[List[str]] = None      # e.g. ["pYards"] for rate stats
    denominator: Optional[List[str]] = None    # e.g. ["pAttempts"] for rate stats
    adjustments: Optional[List[Dict]] = None   # e.g. [{"field": "passTD", "weight": 20}]
    precomputed_field: Optional[str] = None    # e.g. "qbRTG" for pre-computed rate fields
    stat_type: Optional[str] = None            # "custom" for handler-dispatched stats
    handler: Optional[str] = None              # Python handler function name


class BaseFeatureRegistry:
    """
    Base feature registry. Sport apps subclass this and populate STAT_DEFINITIONS.

    Usage in a sport app:

        class BasketballFeatureRegistry(BaseFeatureRegistry):
            STAT_DEFINITIONS = {
                "points": StatDefinition(name="points", category=StatCategory.BASIC, ...),
                "efg": StatDefinition(name="efg", category=StatCategory.RATE, ...),
                ...
            }

        class HockeyFeatureRegistry(BaseFeatureRegistry):
            STAT_DEFINITIONS = {
                "goals": StatDefinition(name="goals", category=StatCategory.BASIC, ...),
                "corsi": StatDefinition(name="corsi", category=StatCategory.DERIVED, ...),
                ...
            }
    """

    # Subclasses MUST populate this
    STAT_DEFINITIONS: Dict[str, StatDefinition] = {}

    # --- Time periods (shared across sports) ---

    VALID_TIME_PERIODS = {"season", "last_5", "last_10", "last_15", "last_20"}

    # --- Separators ---

    SEPARATOR = "|"  # stat|period|weight|perspective

    @classmethod
    def get_all_stat_names(cls) -> List[str]:
        """Get all valid stat names."""
        return sorted(cls.STAT_DEFINITIONS.keys())

    @classmethod
    def get_stat_definition(cls, stat_name: str) -> Optional[StatDefinition]:
        """Get the definition for a stat, or None if not found."""
        return cls.STAT_DEFINITIONS.get(stat_name)

    @classmethod
    def get_db_field(cls, stat_name: str) -> Optional[str]:
        """Get the MongoDB field name for a stat."""
        stat_def = cls.STAT_DEFINITIONS.get(stat_name)
        if stat_def is None:
            return None
        return stat_def.db_field or stat_def.name

    @classmethod
    def get_stats_by_category(cls, category: StatCategory) -> List[str]:
        """Get all stat names in a given category."""
        return [
            name for name, stat_def in cls.STAT_DEFINITIONS.items()
            if stat_def.category == category
        ]

    @classmethod
    def get_side_splittable_stats(cls) -> List[str]:
        """Get stats that support home/away side splits."""
        return [
            name for name, stat_def in cls.STAT_DEFINITIONS.items()
            if stat_def.supports_side_split
        ]

    @classmethod
    def get_net_stats(cls) -> List[str]:
        """Get stats that support net (opponent-adjusted) versions."""
        return [
            name for name, stat_def in cls.STAT_DEFINITIONS.items()
            if stat_def.supports_net
        ]

    @classmethod
    def validate_feature(cls, feature_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a feature name string.

        Feature format: stat|period|weight|perspective
        Example: "goals|season|avg|diff"

        Returns:
            (is_valid, error_message_or_none)
        """
        parts = feature_name.split(cls.SEPARATOR)
        if len(parts) != 4:
            return False, f"Expected 4 parts (stat|period|weight|perspective), got {len(parts)}: '{feature_name}'"

        stat_name, period, weight, perspective = parts

        # Check stat name
        stat_def = cls.STAT_DEFINITIONS.get(stat_name)
        if stat_def is None:
            return False, f"Unknown stat: '{stat_name}'"

        # Check time period
        if stat_def.valid_time_periods:
            if period not in stat_def.valid_time_periods:
                return False, f"Invalid period '{period}' for stat '{stat_name}'"
        elif period not in cls.VALID_TIME_PERIODS:
            return False, f"Invalid period: '{period}'"

        # Check calc weight
        if stat_def.valid_calc_weights and weight not in stat_def.valid_calc_weights:
            return False, f"Invalid weight '{weight}' for stat '{stat_name}'"

        # Check perspective
        valid_perspectives = {"diff", "home", "away", "none"}
        if stat_def.valid_perspectives:
            valid_perspectives = stat_def.valid_perspectives
        if perspective not in valid_perspectives:
            return False, f"Invalid perspective '{perspective}' for stat '{stat_name}'"

        return True, None

    @classmethod
    def build_feature_name(cls, stat: str, period: str, weight: str, perspective: str) -> str:
        """Build a canonical feature name string."""
        return cls.SEPARATOR.join([stat, period, weight, perspective])
