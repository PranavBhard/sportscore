"""
League Database Proxy - Maps collection attribute names to league-configured names.

Provides transparent access to MongoDB collections using either legacy sport-specific
attribute names or normalized names, resolving them via the active league's YAML configuration.

Usage:
    from sportscore.db.league_db_proxy import LeagueDbProxy
    from sportscore.db.mongo import Mongo
    from sportscore.league_config import load_league_config

    league = load_league_config('nba', '/path/to/leagues')
    db = LeagueDbProxy(Mongo().db, league)

    # Attribute access resolves through league config:
    db.games          # -> db['stats_nba'] (from league.collections['games'])
    db.player_stats   # -> db['nba_player_stats'] (from league.collections['player_stats'])
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from sportscore.league_config import BaseLeagueConfig


class LeagueDbProxy:
    """
    Database proxy mapping collection attribute access to the active league's
    configured collection names.

    Sport-specific apps can subclass this and extend _ATTR_TO_KEY with
    legacy attribute mappings.
    """

    # Base normalized attribute -> collection key mappings.
    # Sport apps can extend this in a subclass.
    _ATTR_TO_KEY: Dict[str, str] = {
        "games": "games",
        "player_stats": "player_stats",
        "players": "players",
        "teams": "teams",
        "venues": "venues",
        "rosters": "rosters",
        "model_config_classifier": "model_config_classifier",
        "model_config_points": "model_config_points",
        "master_training_metadata": "master_training_metadata",
        "cached_league_stats": "cached_league_stats",
        "elo_cache": "elo_cache",
        "experiment_runs": "experiment_runs",
        "jobs": "jobs",
    }

    def __init__(self, db, league: "BaseLeagueConfig"):
        self._db = db
        self._league = league

    def __getitem__(self, name: str):
        return self._db[name]

    def __getattr__(self, name: str):
        key = self._ATTR_TO_KEY.get(name)
        if key is not None:
            coll_name = self._league.collections.get(key)
            if coll_name:
                return self._db[coll_name]
        return getattr(self._db, name)
