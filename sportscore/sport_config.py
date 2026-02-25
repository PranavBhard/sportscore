"""
Sport configuration loader.

Provides a base sport config that defines sport-level parameters shared
across all leagues of a sport. League YAMLs override sport-level defaults.

Sport configs live in ``sportscore/sports/<sport_id>.yaml`` and are
shipped as package data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


class SportConfigError(ValueError):
    pass


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise SportConfigError(f"Sport config must be a mapping at top-level: {path}")
        return data
    except FileNotFoundError as e:
        raise SportConfigError(f"Sport config not found: {path}") from e
    except yaml.YAMLError as e:
        raise SportConfigError(f"Failed to parse YAML: {path}: {e}") from e


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise SportConfigError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_str(x: Any, ctx: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise SportConfigError(f"Expected non-empty string for {ctx}")
    return x


def _as_dict(x: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise SportConfigError(f"Expected mapping for {ctx}")
    return x


_SPORTS_DIR = os.path.join(os.path.dirname(__file__), "sports")


@dataclass(frozen=True)
class BaseSportConfig:
    """
    Base sport configuration loaded from a sport YAML file.

    Provides sport-level defaults that league configs can override.
    """
    raw: Dict[str, Any]
    config_path: str

    # --- Meta ---

    @property
    def sport_id(self) -> str:
        meta = _as_dict(_require(self.raw, "meta", self.config_path), "meta")
        return _as_str(_require(meta, "sport_id", self.config_path), "meta.sport_id")

    @property
    def display_name(self) -> str:
        meta = _as_dict(_require(self.raw, "meta", self.config_path), "meta")
        return _as_str(_require(meta, "display_name", self.config_path), "meta.display_name")

    @property
    def outcome_type(self) -> str:
        """Sport-level outcome type: ``"binary"`` or ``"3way"``."""
        meta = self.raw.get("meta", {})
        return meta.get("outcome_type", "binary")

    # --- Season defaults ---

    @property
    def default_exclude_game_types(self) -> List[str]:
        """Sport-level fallback for excluded game types."""
        season = self.raw.get("season") or {}
        v = season.get("default_exclude_game_types", [])
        if v is None:
            return []
        if not isinstance(v, list):
            raise SportConfigError(f"Expected list for season.default_exclude_game_types in {self.config_path}")
        return v


# --- Loader infrastructure ---

_CACHE: Dict[str, BaseSportConfig] = {}


def load_sport_config(
    sport_id: str,
    sports_dir: str = None,
    config_class: type = BaseSportConfig,
    *,
    use_cache: bool = True,
) -> BaseSportConfig:
    """
    Load a sport config from YAML.

    Args:
        sport_id: Sport identifier (e.g., 'basketball', 'hockey')
        sports_dir: Directory containing sport YAML files (default: built-in)
        config_class: Config class to instantiate (default: BaseSportConfig)
        use_cache: Whether to cache loaded configs

    Returns:
        BaseSportConfig (or subclass) instance
    """
    sport_id = (sport_id or "").strip().lower()
    if not sport_id:
        raise SportConfigError("sport_id must be a non-empty string")

    if sports_dir is None:
        sports_dir = _SPORTS_DIR

    cache_key = f"{sports_dir}:{sport_id}"

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = os.path.join(sports_dir, f"{sport_id}.yaml")
    raw = _load_yaml(path)
    cfg = config_class(raw=raw, config_path=path)

    # Trigger validation
    _ = cfg.sport_id
    _ = cfg.display_name

    if use_cache:
        _CACHE[cache_key] = cfg
    return cfg


def get_builtin_sport_config(sport_id: str) -> BaseSportConfig:
    """
    Convenience function to load a built-in sport config.

    Args:
        sport_id: Sport identifier (e.g., 'basketball')

    Returns:
        BaseSportConfig instance from the built-in sports/ directory
    """
    return load_sport_config(sport_id, sports_dir=_SPORTS_DIR)
