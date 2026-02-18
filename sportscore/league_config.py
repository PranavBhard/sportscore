"""
League configuration loader.

Provides a base league config that sport-specific apps extend.
The app supports multiple leagues per sport. All league-specific
variables live in YAML files under `<sport_app>/leagues/<league_id>.yaml`.

Sport apps subclass BaseLeagueConfig and provide their own loader that
points to the correct leagues directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class LeagueConfigError(ValueError):
    pass


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise LeagueConfigError(f"League config must be a mapping at top-level: {path}")
        return data
    except FileNotFoundError as e:
        raise LeagueConfigError(f"League config not found: {path}") from e
    except yaml.YAMLError as e:
        raise LeagueConfigError(f"Failed to parse YAML: {path}: {e}") from e


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise LeagueConfigError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_str(x: Any, ctx: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise LeagueConfigError(f"Expected non-empty string for {ctx}")
    return x


def _as_dict(x: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise LeagueConfigError(f"Expected mapping for {ctx}")
    return x


def _normalize_path(path_str: str, repo_root: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(repo_root, path_str)


class _SafeFormatDict(dict):
    """
    Allows partial .format_map() substitution.
    Preserves unresolved placeholders like {YYYYMMDD} or {game_id}.
    """
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class BaseLeagueConfig:
    """
    Base league configuration. Sport-specific apps can use this directly
    or subclass it to add sport-specific properties.

    Provides access to:
    - meta (league_id, display_name, sport)
    - mongo collections
    - paths
    - season rules (exclude_game_types, cutover month, timezone)
    - elo configuration
    - training configuration
    - ESPN endpoint templates
    - pipeline configuration
    """
    raw: Dict[str, Any]
    config_path: str
    sport_config: Optional[Any] = field(default=None, repr=False)

    # --- Meta ---

    @property
    def league_id(self) -> str:
        return _as_str(_require(_require(self.raw, "meta", self.config_path), "league_id", self.config_path), "meta.league_id")

    @property
    def display_name(self) -> str:
        meta = _require(self.raw, "meta", self.config_path)
        return _as_str(_require(meta, "display_name", self.config_path), "meta.display_name")

    @property
    def sport(self) -> str:
        """Sport identifier (e.g., 'basketball', 'hockey', 'football')."""
        meta = _require(self.raw, "meta", self.config_path)
        return _as_str(meta.get("sport", "unknown"), "meta.sport")

    @property
    def logo_url(self) -> Optional[str]:
        meta = _require(self.raw, "meta", self.config_path)
        logo = meta.get("logo_url")
        if not logo:
            return None
        return _as_str(logo, "meta.logo_url")

    # --- Mongo collections ---

    @property
    def _required_collections(self) -> List[str]:
        """
        Override in subclass to require sport-specific collections.
        Base requires the common set shared across all sports.
        """
        return [
            "games",
            "player_stats",
            "players",
            "teams",
            "rosters",
            "venues",
            "model_config_classifier",
            "model_config_points",
            "master_training_metadata",
            "cached_league_stats",
            "experiment_runs",
            "jobs",
        ]

    @property
    def collections(self) -> Dict[str, str]:
        mongo = _as_dict(_require(self.raw, "mongo", self.config_path), "mongo")
        cols = _as_dict(_require(mongo, "collections", self.config_path), "mongo.collections")
        out: Dict[str, str] = {}
        for k in self._required_collections:
            out[k] = _as_str(_require(cols, k, f"{self.config_path} mongo.collections"), f"mongo.collections.{k}")
        # Preserve any additional collections
        for k, v in cols.items():
            if k not in out and isinstance(v, str):
                out[k] = v
        return out

    # --- Paths ---

    @property
    def _repo_root(self) -> str:
        """Derive repo root from config_path. Override if needed."""
        # config_path is typically <repo>/<app>/leagues/<league>.yaml
        # Go up from leagues/ -> app/ -> repo/
        leagues_dir = os.path.dirname(self.config_path)
        app_dir = os.path.dirname(leagues_dir)
        return os.path.dirname(app_dir)

    @property
    def master_training_csv(self) -> str:
        paths = _as_dict(_require(self.raw, "paths", self.config_path), "paths")
        p = _as_str(_require(paths, "master_training_csv", self.config_path), "paths.master_training_csv")
        return _normalize_path(p, self._repo_root)

    # --- Teams ---

    @property
    def team_id_map_file(self) -> Optional[str]:
        teams = self.raw.get("teams") or {}
        if not isinstance(teams, dict):
            return None
        f = teams.get("team_id_map_file")
        if not f or not isinstance(f, str):
            return None
        return _normalize_path(f, self._repo_root)

    @property
    def team_primary_identifier(self) -> str:
        teams = self.raw.get("teams") or {}
        if not isinstance(teams, dict):
            return "name"
        identifier = teams.get("primary_identifier", "name")
        if identifier not in ("name", "id"):
            raise LeagueConfigError(
                f"teams.primary_identifier must be 'name' or 'id', got '{identifier}' in {self.config_path}"
            )
        return identifier

    @property
    def include_team_id(self) -> bool:
        teams = self.raw.get("teams") or {}
        if not isinstance(teams, dict):
            return False
        return bool(teams.get("include_team_id", False))

    # --- Season rules ---

    @property
    def season_rules(self) -> Dict[str, Any]:
        season = self.raw.get("season") or {}
        if not isinstance(season, dict):
            raise LeagueConfigError(f"Expected mapping for season in {self.config_path}")
        return season

    @property
    def timezone(self) -> str:
        return _as_str(self.season_rules.get("timezone", "America/New_York"), "season.timezone")

    @property
    def season_cutover_month(self) -> int:
        m = self.season_rules.get("season_cutover_month", 8)
        try:
            m_int = int(m)
        except Exception as e:
            raise LeagueConfigError(f"Expected integer for season.season_cutover_month in {self.config_path}") from e
        if m_int < 1 or m_int > 12:
            raise LeagueConfigError(f"season.season_cutover_month out of range (1-12) in {self.config_path}")
        return m_int

    @property
    def exclude_game_types(self) -> list:
        v = self.season_rules.get("exclude_game_types", [])
        if v is None:
            return []
        if not isinstance(v, list):
            raise LeagueConfigError(f"Expected list for season.exclude_game_types in {self.config_path}")
        return v

    @property
    def exclude_game_types_effective(self) -> list:
        """League YAML -> sport config fallback -> hardcoded last resort."""
        league_val = self.season_rules.get("exclude_game_types")
        if league_val is not None:
            return league_val if isinstance(league_val, list) else []
        if self.sport_config is not None:
            return self.sport_config.default_exclude_game_types
        return ['preseason', 'allstar']

    # --- Season overrides ---

    @property
    def season_overrides(self) -> Dict[str, Dict]:
        """Per-season exceptions (e.g., COVID delays). Keyed by season string."""
        return self.season_rules.get("season_overrides") or {}

    def get_season_start_month(self, season: str = None) -> int:
        """Season start month, with per-season override support."""
        if season and season in self.season_overrides:
            override = self.season_overrides[season]
            if "start_month" in override:
                return int(override["start_month"])
        return self.season_start_month

    def get_season_end_month(self, season: str = None) -> int:
        """Season end month, with per-season override support."""
        if season and season in self.season_overrides:
            override = self.season_overrides[season]
            if "end_month" in override:
                return int(override["end_month"])
        return self.season_end_month

    # --- Elo configuration ---

    @property
    def elo_config(self) -> Dict[str, Any]:
        """Full elo configuration dict from league YAML."""
        return self.raw.get("elo") or {}

    @property
    def elo_starting_rating(self) -> float:
        return float(self.elo_config.get("starting_rating", 1500))

    @property
    def elo_k_factor(self) -> float:
        return float(self.elo_config.get("k_factor", 20))

    @property
    def elo_home_advantage(self) -> float:
        return float(self.elo_config.get("home_advantage", 100))

    @property
    def elo_strategy(self) -> str:
        return self.elo_config.get("strategy", "static")

    @property
    def elo_k_schedule(self) -> list:
        """For dynamic K strategy (future). List of {max_games, k} dicts."""
        return self.elo_config.get("k_schedule") or []

    @property
    def elo_home_advantage_overrides(self) -> Dict[str, float]:
        """Per game-type home advantage overrides (future). e.g., {tournament: 75}"""
        return self.elo_config.get("home_advantage_overrides") or {}

    # --- Training config ---

    @property
    def training_config(self) -> Dict[str, Any]:
        return self.raw.get("training") or {}

    @property
    def training_min_games_played(self) -> int:
        return int(self.training_config.get("min_games_played", 15))

    @property
    def training_team_string_columns(self) -> list:
        v = self.training_config.get("team_string_columns", ["Home", "Away"])
        return list(v)

    # --- ESPN endpoint templates ---

    @property
    def espn(self) -> Dict[str, Any]:
        return _as_dict(_require(self.raw, "espn", self.config_path), "espn")

    def espn_endpoint(self, key: str) -> str:
        espn = self.espn
        endpoints = _as_dict(_require(espn, "endpoints", self.config_path), "espn.endpoints")
        template = _as_str(_require(endpoints, key, f"{self.config_path} espn.endpoints"), f"espn.endpoints.{key}")

        base_url_web = _as_str(_require(espn, "base_url_web", self.config_path), "espn.base_url_web")
        base_url_site = _as_str(_require(espn, "base_url_site", self.config_path), "espn.base_url_site")
        league_slug = _as_str(_require(espn, "league_slug", self.config_path), "espn.league_slug")
        sport_path = _as_str(_require(espn, "sport_path", self.config_path), "espn.sport_path")

        return template.format_map(_SafeFormatDict(
            base_url_web=base_url_web,
            base_url_site=base_url_site,
            league_slug=league_slug,
            sport_path=sport_path,
        ))

    def espn_page(self, key: str) -> str:
        espn = self.espn
        pages = _as_dict(_require(espn, "pages", self.config_path), "espn.pages")
        template = _as_str(_require(pages, key, f"{self.config_path} espn.pages"), f"espn.pages.{key}")

        base_url_public = espn.get("base_url_public", "https://www.espn.com")
        base_url_public = _as_str(base_url_public, "espn.base_url_public")
        sport_path = _as_str(_require(espn, "sport_path", self.config_path), "espn.sport_path")

        return template.format_map(_SafeFormatDict(
            base_url_public=base_url_public,
            sport_path=sport_path,
        ))

    # --- Pipeline configuration ---

    @property
    def pipeline(self) -> Dict[str, Any]:
        return self.raw.get("pipeline") or {}

    @property
    def min_season(self) -> str:
        return self.pipeline.get("min_season", "2007-2008")

    @property
    def season_start_month(self) -> int:
        return int(self.pipeline.get("season_start_month", 10))

    @property
    def season_start_day(self) -> int:
        return int(self.pipeline.get("season_start_day", 1))

    @property
    def season_end_month(self) -> int:
        return int(self.pipeline.get("season_end_month", 6))

    @property
    def season_end_day(self) -> int:
        return int(self.pipeline.get("season_end_day", 30))

    @property
    def start_year(self) -> int:
        return int(self.min_season.split("-")[0])

    @property
    def pipelines(self) -> Dict[str, Any]:
        return self.raw.get("pipelines") or {}

    @property
    def extra_feature_stats(self) -> List[str]:
        """Sport-specific extra stats used in feature calculations (e.g., PER)."""
        return self.raw.get("extra_feature_stats") or []


# --- Loader infrastructure ---

_CACHE: Dict[str, BaseLeagueConfig] = {}


def get_available_leagues(leagues_dir: str) -> List[str]:
    """Return list of available league IDs from YAML files in leagues/ dir."""
    if not os.path.isdir(leagues_dir):
        return []
    return [f[:-5] for f in os.listdir(leagues_dir) if f.endswith(".yaml")]


def load_league_config(
    league_id: str,
    leagues_dir: str,
    config_class: type = BaseLeagueConfig,
    *,
    use_cache: bool = True,
    required_endpoints: Optional[List[str]] = None,
    sport_config: Optional[Any] = None,
) -> BaseLeagueConfig:
    """
    Load a league config from YAML.

    Args:
        league_id: League identifier (e.g., 'nba', 'nhl')
        leagues_dir: Absolute path to the leagues/ directory
        config_class: LeagueConfig class to instantiate (default: BaseLeagueConfig)
        use_cache: Whether to cache loaded configs
        required_endpoints: ESPN endpoints to validate on load
        sport_config: Optional BaseSportConfig for sport-level defaults

    Returns:
        BaseLeagueConfig (or subclass) instance
    """
    league_id = (league_id or "").strip().lower()
    cache_key = f"{leagues_dir}:{league_id}"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = os.path.join(leagues_dir, f"{league_id}.yaml")
    raw = _load_yaml(path)
    cfg = config_class(raw=raw, config_path=path, sport_config=sport_config)

    # Trigger validation
    _ = cfg.league_id
    _ = cfg.display_name
    _ = cfg.collections
    _ = cfg.master_training_csv

    if required_endpoints:
        for ep in required_endpoints:
            _ = cfg.espn_endpoint(ep)

    if use_cache:
        _CACHE[cache_key] = cfg
    return cfg
