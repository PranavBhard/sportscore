"""Typed service containers for the sportscore <-> sport app interface.

Sport plugins return a ``SportServices`` instance from ``get_web_services()``.
``shared_routes.py`` accesses fields via attribute lookup instead of dict ``.get()``,
giving typo detection at startup and IDE/mypy support for signatures.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class JobsService:
    """Background job lifecycle helpers (create / track / complete / fail)."""
    create: Optional[Callable] = None
    update: Optional[Callable] = None
    complete: Optional[Callable] = None
    fail: Optional[Callable] = None
    get_job: Optional[Callable] = None      # renamed from "get" to avoid shadowing builtin
    get_running: Optional[Callable] = None


@dataclass
class ModelServices:
    """Model configuration, training, and defaults."""
    config_manager: Optional[Any] = None
    training_service: Optional[Any] = None
    points_trainer: Optional[Any] = None
    default_model_types: Optional[List[str]] = None
    c_supported_models: Optional[List[str]] = None
    default_c_values: Optional[List[str]] = None


@dataclass
class MarketServices:
    """Kalshi market integration (account-level + sport-specific)."""
    dashboard_getter: Optional[Callable] = None
    fills_getter: Optional[Callable] = None
    settlements_getter: Optional[Callable] = None
    bins_getter: Optional[Callable] = None
    prices_getter: Optional[Callable] = None


@dataclass
class FeatureServices:
    """Feature sets, master training CSV, and regeneration helpers."""
    feature_sets: Optional[Dict[str, List[str]]] = None
    feature_set_descriptions: Optional[Dict[str, str]] = None
    available_features: Optional[Set[str]] = None
    master_training_csv: Optional[str] = None
    dependency_resolver: Optional[Callable] = None
    regenerator: Optional[Callable] = None
    possible_getter: Optional[Callable] = None
    column_adder: Optional[Callable] = None
    available_seasons_getter: Optional[Callable] = None
    season_regenerator: Optional[Callable] = None
    full_regenerator: Optional[Callable] = None


@dataclass
class EloServices:
    """Elo rating cache and league stats."""
    stats_getter: Optional[Callable] = None
    runner: Optional[Callable] = None
    clearer: Optional[Callable] = None
    cached_league_stats_getter: Optional[Callable] = None
    league_stats_cacher: Optional[Callable] = None


@dataclass
class DataServices:
    """External data source integrations (ESPN, etc.)."""
    espn_db_auditor: Optional[Callable] = None
    espn_db_puller: Optional[Callable] = None


@dataclass
class SportServices:
    """Top-level container set on ``g.services`` per request.

    Sport plugins populate the groups they support; unused groups
    default to empty sub-dataclasses (all fields None), and
    shared_routes returns 501 for those endpoints.
    """
    model: ModelServices = field(default_factory=ModelServices)
    market: MarketServices = field(default_factory=MarketServices)
    features: FeatureServices = field(default_factory=FeatureServices)
    elo: EloServices = field(default_factory=EloServices)
    data: DataServices = field(default_factory=DataServices)
    jobs: Optional[JobsService] = None
