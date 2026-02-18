"""
Stat Loader - YAML parser for declarative stat definitions.

Reads a sport-specific stats.yaml file and converts it into
a Dict[str, StatDefinition] for use by the registry and engine.
"""

import yaml
from typing import Any, Dict

from sportscore.features.base_registry import StatCategory, StatDefinition


# Map YAML category strings to StatCategory enum values
_CATEGORY_MAP = {
    "basic": StatCategory.BASIC,
    "rate": StatCategory.RATE,
    "net": StatCategory.NET,
    "derived": StatCategory.DERIVED,
    "special": StatCategory.SPECIAL,
}


def load_stat_definitions(yaml_path: str) -> Dict[str, StatDefinition]:
    """
    Load a stats.yaml file and return Dict[str, StatDefinition].

    YAML schema example:
        stats:
          margin:
            category: basic
            description: "Point differential"
            supports_side_split: true
            formula: "team.points - opponent.points"
          pyper:
            category: rate
            numerator: [pYards]
            denominator: [pAttempts]
          elo:
            category: special
            type: custom
            handler: compute_elo
            valid_calc_weights: [raw]
            valid_time_periods: [none]
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    stats_raw = data.get("stats", {})
    definitions: Dict[str, StatDefinition] = {}

    for stat_name, cfg in stats_raw.items():
        category_str = cfg.get("category", "basic")
        category = _CATEGORY_MAP.get(category_str, StatCategory.BASIC)

        definitions[stat_name] = StatDefinition(
            name=stat_name,
            category=category,
            db_field=cfg.get("db_field"),
            description=cfg.get("description", ""),
            supports_side_split=cfg.get("supports_side_split", False),
            supports_net=cfg.get("supports_net", False),
            valid_calc_weights=set(cfg["valid_calc_weights"]) if "valid_calc_weights" in cfg else {"raw", "avg"},
            valid_time_periods=set(cfg["valid_time_periods"]) if "valid_time_periods" in cfg else set(),
            valid_perspectives=set(cfg["valid_perspectives"]) if "valid_perspectives" in cfg else set(),
            requires_aggregation=cfg.get("requires_aggregation", False),
            # YAML-driven computation fields
            formula=cfg.get("formula"),
            db_fields=cfg.get("db_fields"),
            numerator=cfg.get("numerator"),
            denominator=cfg.get("denominator"),
            adjustments=cfg.get("adjustments"),
            precomputed_field=cfg.get("precomputed_field"),
            stat_type=cfg.get("type"),
            handler=cfg.get("handler"),
        )

    return definitions


def load_stat_meta(yaml_path: str) -> Dict[str, Any]:
    """
    Load the meta section from a stats.yaml file.

    Returns dict with keys like valid_time_periods, recency_alpha, etc.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("meta", {})
