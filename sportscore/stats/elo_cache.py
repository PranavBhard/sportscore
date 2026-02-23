"""
Elo Rating Cache Module

Manages cached Elo ratings in MongoDB for fast prediction-time lookups.
Elo ratings are computed from historical game data and stored per (team, date, season).

Document schema:
{
    "team": "Team Name",
    "league": "epl",           # present when league scoping is active
    "game_date": "2024-01-15",
    "season": "2023-2024",
    "elo": 1623.5,
    "created_at": ISODate("2024-01-16T10:30:00Z")
}

Indexes:
- (team, game_date, season) - unique, for fast lookups (legacy)
- (league, team, game_date, season) - unique, for league-scoped lookups
- (season) - for cache management by season
- (game_date) - for finding latest ratings
"""

import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

if TYPE_CHECKING:
    from sportscore.league_config import BaseLeagueConfig

# Default collection name
DEFAULT_ELO_CACHE_COLLECTION = 'cached_elo_ratings'

# Default Elo parameters
DEFAULT_STARTING_ELO = 1500
DEFAULT_K_FACTOR = 20
DEFAULT_HOME_ADVANTAGE = 100


class EloCache:
    """
    Manages Elo rating computation and caching in MongoDB.

    Usage:
        from sportscore.stats.elo_cache import EloCache
        from sportscore.db.mongo import Mongo

        mongo = Mongo()
        elo_cache = EloCache(mongo.db)

        # Compute and cache elo for all games
        stats = elo_cache.compute_and_cache_all()

        # Get elo for a specific game
        elo = elo_cache.get_elo_for_game("Team Name", "2024-01-15", "2023-2024")
    """

    def __init__(
        self,
        db: Database,
        league: Optional["BaseLeagueConfig"] = None,
        games_repo=None,
        collection_name: Optional[str] = None,
        starting_elo=None,
        k_factor=None,
        home_advantage=None
    ):
        """
        Initialize EloCache.

        Args:
            db: MongoDB database instance
            league: Optional league config for collection names and ELO params
            games_repo: Optional games repository instance for data access.
                        If not provided, uses raw db queries.
            collection_name: Optional explicit collection name override
            starting_elo: Initial Elo rating for new teams. Falls back to league config, then 1500.
            k_factor: K-factor for Elo updates. Falls back to league config, then 20.
            home_advantage: Home court/ice/field advantage in Elo points. Falls back to league config, then 100.
        """
        self.db = db
        self.league = league

        # League scoping — filter all queries/writes by league when set
        self._league_id = league.league_id if league else None

        # Collection resolution
        effective = collection_name
        if league is not None:
            effective = effective or league.collections.get("elo_cache")
        self.collection = db[effective or DEFAULT_ELO_CACHE_COLLECTION]

        # Resolve ELO params: explicit kwarg > league config > hardcoded default
        if league is not None:
            self.starting_elo = starting_elo if starting_elo is not None else league.elo_starting_rating
            self.k_factor = k_factor if k_factor is not None else league.elo_k_factor
            self.home_advantage = home_advantage if home_advantage is not None else league.elo_home_advantage
        else:
            self.starting_elo = starting_elo if starting_elo is not None else DEFAULT_STARTING_ELO
            self.k_factor = k_factor if k_factor is not None else DEFAULT_K_FACTOR
            self.home_advantage = home_advantage if home_advantage is not None else DEFAULT_HOME_ADVANTAGE

        # Derived config from league (or sensible defaults when no league)
        self._strategy = league.elo_strategy if league else "static"
        self._k_schedule = league.elo_k_schedule if league else []
        self._carryover_enabled = league.elo_carryover_enabled if league else False
        self._carryover_alpha = league.elo_carryover_alpha if league else 1.0
        self._carryover_mean = league.elo_carryover_mean_rating if league else self.starting_elo
        self._neutral_site_ha = league.elo_neutral_site_home_advantage if league else 0
        self._ha_overrides = league.elo_home_advantage_overrides if league else {}

        # Margin adjustment config
        self._margin_adj_enabled = league.elo_margin_adjustment_enabled if league else False
        self._margin_adj_method = league.elo_margin_adjustment_method if league else ""

        # Games repo for data access (sport-specific, injected by caller)
        self._games_repo = games_repo

        # In-memory cache for fast lookups (populated by preload())
        self._memory_cache = None  # {(team, game_date, season): elo}
        self._team_latest_cache = None  # {team: [(game_date, elo), ...]} sorted by date desc
        self._preloaded = False

        # Ensure indexes exist
        self._ensure_indexes()

    @property
    def _exclude_game_types(self) -> list:
        """Get excluded game types from league config, with fallback."""
        return self.league.exclude_game_types if self.league else ['preseason', 'allstar']

    def _league_query(self, query: dict = None) -> dict:
        """Return a query dict with league filter applied when league scoping is active."""
        q = dict(query) if query else {}
        if self._league_id:
            q["league"] = self._league_id
        return q

    def _get_k_for_games_played(self, games_played: int) -> float:
        """
        Look up K factor from k_schedule based on pre-game count.

        games_played is the number of games a team has ALREADY played this season
        (before the current game). max_games is an exclusive upper bound:
        max_games=10 covers pre-game counts 0..9 (the team's first 10 games).
        """
        if self._strategy != "dynamic" or not self._k_schedule:
            return self.k_factor
        for entry in self._k_schedule:
            if "max_games" in entry:
                if "k" not in entry:
                    raise ValueError(f"Malformed k_schedule entry (missing 'k'): {entry}")
                if games_played < entry["max_games"]:
                    return float(entry["k"])
        # Fall through to default entry
        for entry in self._k_schedule:
            if "default" in entry:
                return float(entry["default"])
        raise ValueError(
            f"k_schedule has no matching tier for games_played={games_played} "
            f"and no default entry: {self._k_schedule}"
        )

    def _get_score_diff(self, game: dict) -> int:
        """Extract absolute score differential from game doc (sport-agnostic)."""
        home = game.get('homeTeam', {})
        away = game.get('awayTeam', {})
        h = home.get('score') or home.get('runs') or home.get('goals') or 0
        a = away.get('score') or away.get('runs') or away.get('goals') or 0
        return abs(int(h) - int(a))

    def _ensure_indexes(self):
        """Create required indexes if they don't exist.

        Tolerates name conflicts (IndexOptionsConflict, code 85) which arise
        when another code path already created the same key pattern under a
        different name (e.g. MongoDB auto-generated names from full_pipeline).
        """
        from pymongo.errors import OperationFailure

        def _safe_create(keys, **kwargs):
            try:
                self.collection.create_index(keys, **kwargs)
            except OperationFailure as e:
                if e.code != 85:  # IndexOptionsConflict
                    raise

        # Legacy indexes (non-league-scoped)
        _safe_create(
            [("team", ASCENDING), ("game_date", ASCENDING), ("season", ASCENDING)],
            unique=True, name="team_date_season_unique",
        )
        _safe_create(
            [("season", ASCENDING)],
            name="season_idx",
        )
        _safe_create(
            [("game_date", DESCENDING)],
            name="game_date_desc_idx",
        )
        _safe_create(
            [("team", ASCENDING), ("game_date", DESCENDING)],
            name="team_date_desc_idx",
        )

        # League-scoped indexes
        if self._league_id:
            _safe_create(
                [("league", ASCENDING), ("team", ASCENDING),
                 ("game_date", ASCENDING), ("season", ASCENDING)],
                unique=True, name="league_team_date_season_unique",
            )
            _safe_create(
                [("league", ASCENDING), ("season", ASCENDING)],
                name="league_season_idx",
            )
            _safe_create(
                [("league", ASCENDING), ("team", ASCENDING),
                 ("game_date", DESCENDING)],
                name="league_team_date_desc_idx",
            )

    def preload(self, seasons: List[str] = None):
        """
        Preload all elo ratings into memory for fast batch processing.

        Args:
            seasons: Optional list of seasons to preload. If None, loads all.
        """
        query = self._league_query()
        if seasons:
            query['season'] = {'$in': seasons}

        print(f"  Preloading elo cache into memory...")
        docs = list(self.collection.find(query, {'team': 1, 'game_date': 1, 'season': 1, 'elo': 1, '_id': 0}))

        self._memory_cache = {}
        for doc in docs:
            key = (doc['team'], doc['game_date'], doc['season'])
            self._memory_cache[key] = doc['elo']

        team_records = defaultdict(list)
        for doc in docs:
            team_records[doc['team']].append((doc['game_date'], doc['elo']))

        self._team_latest_cache = {}
        for team, records in team_records.items():
            self._team_latest_cache[team] = sorted(records, key=lambda x: x[0], reverse=True)

        self._preloaded = True
        print(f"  Loaded {len(docs):,} elo records for {len(self._team_latest_cache)} teams")

    def compute_elo_ratings(
        self,
        games: List[dict],
        progress_callback: callable = None
    ) -> Tuple[Dict[Tuple[str, str, str], float], Dict[str, float]]:
        """
        Compute Elo ratings from a list of games.

        Supports season carryover regression, dynamic K schedules,
        neutral site handling, per-game_type home advantage overrides,
        and margin-of-victory adjustment.

        Args:
            games: List of game documents (multi-season, chronological).
                   Each game must have: homeTeam.name, awayTeam.name, date, season, homeWon
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Tuple of (elo_history, current_ratings)
            - elo_history: Dict mapping (team, game_date, season) -> pre-game elo
            - current_ratings: Dict mapping team -> current elo rating
        """
        elo = defaultdict(lambda: self.starting_elo)
        elo_history = {}
        games_played = defaultdict(int)  # team -> games played this season (pre-game count)
        current_season = None

        valid_games = [
            g for g in games
            if all([
                g.get('season'),
                g.get('homeTeam', {}).get('name'),
                g.get('awayTeam', {}).get('name'),
                g.get('date'),
                'homeWon' in g
            ])
        ]

        if len(valid_games) < len(games):
            print(f"  Filtered out {len(games) - len(valid_games)} games with missing fields")

        # Sort by (season, date) to guarantee season boundaries are clean
        sorted_games = sorted(valid_games, key=lambda g: (g.get('season', ''), g.get('date', '')))
        total = len(sorted_games)

        for idx, game in enumerate(sorted_games):
            season = game['season']

            # Season carryover regression at boundary
            if season != current_season:
                if current_season is not None and self._carryover_enabled:
                    for team in list(elo.keys()):
                        elo[team] = self._carryover_mean + self._carryover_alpha * (elo[team] - self._carryover_mean)
                games_played.clear()
                current_season = season

            home = game['homeTeam']['name']
            away = game['awayTeam']['name']
            game_date = game['date']

            # Store pre-game elo
            elo_history[(home, game_date, season)] = elo[home]
            elo_history[(away, game_date, season)] = elo[away]

            # Resolve home advantage: neutral > game_type override > default
            is_neutral = game.get('neutralSite', False)
            game_type = game.get('game_type', '')
            if is_neutral:
                ha = self._neutral_site_ha
            elif game_type in self._ha_overrides:
                ha = self._ha_overrides[game_type]
            else:
                ha = self.home_advantage

            # Expected outcome
            home_elo_adj = elo[home] + ha
            expected_home = 1 / (1 + 10 ** ((elo[away] - home_elo_adj) / 400))

            # Dynamic K: average of both teams' schedule-K (preserves zero-sum)
            k = (self._get_k_for_games_played(games_played[home])
                 + self._get_k_for_games_played(games_played[away])) / 2

            # Update ratings
            actual_home = 1 if game['homeWon'] else 0
            elo_change = k * (actual_home - expected_home)

            # Margin-of-victory adjustment
            if self._margin_adj_enabled and self._margin_adj_method == 'log_run_diff':
                score_diff = self._get_score_diff(game)
                if score_diff > 0:
                    elo_change *= math.log(1 + score_diff)

            elo[home] += elo_change
            elo[away] -= elo_change

            # Track games played (increment AFTER update — count is pre-game)
            games_played[home] += 1
            games_played[away] += 1

            if progress_callback and (idx + 1) % 500 == 0:
                progress_callback(idx + 1, total)

        if progress_callback:
            progress_callback(total, total)

        return elo_history, dict(elo)

    def cache_elo_ratings(
        self,
        elo_history: Dict[Tuple[str, str, str], float],
        batch_size: int = 1000,
        progress_callback: callable = None
    ) -> int:
        """
        Store computed Elo ratings in MongoDB cache.

        Args:
            elo_history: Dict mapping (team, game_date, season) -> elo rating
            batch_size: Number of documents to insert per batch
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Number of documents upserted
        """
        from pymongo import UpdateOne

        total = len(elo_history)
        upserted = 0
        operations = []
        now = datetime.utcnow()

        for idx, ((team, game_date, season), elo) in enumerate(elo_history.items()):
            # Match on the natural key only (team, game_date, season) — NOT
            # league — so that legacy docs (without a league field) are matched
            # and updated rather than causing a duplicate-key conflict with the
            # legacy team_date_season_unique index.
            filter_doc = {"team": team, "game_date": game_date, "season": season}
            set_doc = {
                "team": team,
                "game_date": game_date,
                "season": season,
                "elo": elo,
                "updated_at": now
            }
            if self._league_id:
                set_doc["league"] = self._league_id

            operations.append(
                UpdateOne(filter_doc, {"$set": set_doc}, upsert=True)
            )

            if len(operations) >= batch_size:
                result = self.collection.bulk_write(operations)
                upserted += result.upserted_count + result.modified_count
                operations = []

                if progress_callback:
                    progress_callback(idx + 1, total)

        if operations:
            result = self.collection.bulk_write(operations)
            upserted += result.upserted_count + result.modified_count

        if progress_callback:
            progress_callback(total, total)

        return upserted

    def compute_and_cache_all(
        self,
        seasons: List[str] = None,
        progress_callback: callable = None
    ) -> dict:
        """
        Compute and cache Elo ratings for all games (or specific seasons).

        Requires a games_repo to be set (injected at construction time).

        Args:
            seasons: Optional list of seasons to process
            progress_callback: Optional callback(stage, current, total, message)

        Returns:
            Dict with cache statistics
        """
        if self._games_repo is None:
            raise RuntimeError("games_repo must be provided to use compute_and_cache_all()")

        query = {'game_type': {'$nin': self._exclude_game_types}}
        if self._league_id:
            query['league'] = self._league_id
        if seasons:
            query['season'] = {'$in': seasons}

        if progress_callback:
            progress_callback('fetch', 0, 1, 'Fetching games from database...')

        games = self._games_repo.find(query)
        games.sort(key=lambda g: g.get('date', ''))
        game_count = len(games)

        if progress_callback:
            progress_callback('fetch', 1, 1, f'Fetched {game_count} games')

        if game_count == 0:
            return {
                'games_processed': 0,
                'ratings_cached': 0,
                'teams': 0,
                'seasons': [],
                'date_range': None
            }

        if progress_callback:
            progress_callback('compute', 0, game_count, 'Computing Elo ratings...')

        def compute_progress(current, total):
            if progress_callback:
                progress_callback('compute', current, total, f'Computing Elo: {current}/{total} games')

        elo_history, current_ratings = self.compute_elo_ratings(games, compute_progress)

        if progress_callback:
            progress_callback('cache', 0, len(elo_history), 'Caching Elo ratings to MongoDB...')

        def cache_progress(current, total):
            if progress_callback:
                progress_callback('cache', current, total, f'Caching: {current}/{total} ratings')

        ratings_cached = self.cache_elo_ratings(elo_history, progress_callback=cache_progress)

        seasons_in_data = sorted(set(g['season'] for g in games if g.get('season')))
        dates = [g['date'] for g in games if g.get('date')]

        return {
            'games_processed': game_count,
            'ratings_cached': ratings_cached,
            'teams': len(current_ratings),
            'seasons': seasons_in_data,
            'date_range': {
                'min': min(dates) if dates else None,
                'max': max(dates) if dates else None
            },
            'current_ratings': current_ratings
        }

    def get_elo_for_game(self, team: str, game_date: str, season: str) -> Optional[float]:
        """
        Get cached pre-game Elo rating for a team.

        Args:
            team: Team name
            game_date: Game date string (YYYY-MM-DD)
            season: Season string (e.g., "2023-2024")

        Returns:
            Elo rating if cached, None otherwise
        """
        if self._preloaded and self._memory_cache is not None:
            return self._memory_cache.get((team, game_date, season))

        doc = self.collection.find_one(self._league_query({
            "team": team,
            "game_date": game_date,
            "season": season
        }))
        return doc['elo'] if doc else None

    def get_elo_for_game_with_fallback(
        self,
        team: str,
        game_date: str,
        season: str
    ) -> float:
        """
        Get cached Elo rating with fallback to most recent or default.

        Args:
            team: Team name
            game_date: Game date string (YYYY-MM-DD)
            season: Season string

        Returns:
            Elo rating (cached, most recent for team, or default)
        """
        elo = self.get_elo_for_game(team, game_date, season)
        if elo is not None:
            return elo

        if self._preloaded and self._team_latest_cache is not None:
            team_records = self._team_latest_cache.get(team)
            if team_records:
                for rec_date, rec_elo in team_records:
                    if rec_date < game_date:
                        return rec_elo
            return self.starting_elo

        doc = self.collection.find_one(
            self._league_query({"team": team, "game_date": {"$lt": game_date}}),
            sort=[("game_date", DESCENDING)]
        )
        if doc:
            return doc['elo']

        return self.starting_elo

    def get_current_elo(self, team: str) -> float:
        """
        Get the most recent Elo rating for a team.

        Args:
            team: Team name

        Returns:
            Most recent Elo rating, or default if not found
        """
        doc = self.collection.find_one(
            self._league_query({"team": team}),
            sort=[("game_date", DESCENDING)]
        )
        return doc['elo'] if doc else self.starting_elo

    def get_all_current_elos(self) -> Dict[str, float]:
        """
        Get the most recent Elo rating for all teams.

        Returns:
            Dict mapping team name -> current elo rating
        """
        pipeline = []
        if self._league_id:
            pipeline.append({"$match": {"league": self._league_id}})
        pipeline.extend([
            {"$sort": {"game_date": -1}},
            {"$group": {
                "_id": "$team",
                "elo": {"$first": "$elo"},
                "game_date": {"$first": "$game_date"}
            }},
            {"$sort": {"elo": -1}}
        ])

        results = list(self.collection.aggregate(pipeline))
        return {doc['_id']: doc['elo'] for doc in results}

    def get_cache_stats(self) -> dict:
        """Get statistics about the Elo cache."""
        base_query = self._league_query()
        total_docs = self.collection.count_documents(base_query)

        if total_docs == 0:
            return {
                'total_ratings': 0,
                'teams': 0,
                'seasons': [],
                'date_range': None,
                'last_updated': None
            }

        teams = self.collection.distinct('team', base_query)
        seasons = sorted(self.collection.distinct('season', base_query))

        oldest = self.collection.find_one(base_query, sort=[("game_date", ASCENDING)])
        newest = self.collection.find_one(base_query, sort=[("game_date", DESCENDING)])

        last_updated_doc = self.collection.find_one(base_query, sort=[("updated_at", DESCENDING)])
        last_updated = last_updated_doc.get('updated_at') if last_updated_doc else None

        return {
            'total_ratings': total_docs,
            'teams': len(teams),
            'team_names': sorted(teams),
            'seasons': seasons,
            'date_range': {
                'min': oldest['game_date'] if oldest else None,
                'max': newest['game_date'] if newest else None
            },
            'last_updated': last_updated
        }

    def get_team_elo_history(
        self,
        team: str,
        season: str = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Get Elo rating history for a team.

        Args:
            team: Team name
            season: Optional season filter
            limit: Maximum number of records to return

        Returns:
            List of {game_date, season, elo} dicts, sorted by date descending
        """
        query = self._league_query({"team": team})
        if season:
            query["season"] = season

        cursor = self.collection.find(
            query,
            {"_id": 0, "game_date": 1, "season": 1, "elo": 1}
        ).sort("game_date", DESCENDING).limit(limit)

        return list(cursor)

    def clear_cache(self, seasons: List[str] = None) -> int:
        """
        Clear Elo cache.

        Args:
            seasons: Optional list of seasons to clear. If None, clears all.

        Returns:
            Number of documents deleted
        """
        query = self._league_query()
        if seasons:
            query["season"] = {"$in": seasons}

        result = self.collection.delete_many(query)
        return result.deleted_count
