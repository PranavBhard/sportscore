"""
Base Prediction Service - Sport-agnostic prediction infrastructure.

Provides the framework for prediction context preloading and
prediction orchestration. Sport-specific apps subclass these to
plug in their own feature generators, stat handlers, and models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from threading import Lock


@dataclass
class BasePredictionContext:
    """
    Preloaded data context for predictions. Caches game data,
    team stats, and other expensive-to-compute values.

    Sport-specific apps extend this with their own cached data
    (e.g., basketball adds PER cache, hockey adds Corsi cache).
    """
    league_id: str = ""
    season: str = ""
    games_home: Dict[str, Any] = field(default_factory=dict)
    games_away: Dict[str, Any] = field(default_factory=dict)
    team_records: Dict[str, Any] = field(default_factory=dict)
    venue_cache: Dict[str, Any] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self.games_home.clear()
            self.games_away.clear()
            self.team_records.clear()
            self.venue_cache.clear()


class BasePredictionService(ABC):
    """
    Abstract prediction service. Sport-specific apps implement
    the abstract methods to provide their own prediction logic.

    The flow is:
    1. build_context() - preload data for a date/season
    2. predict_game() - generate prediction for a single game
    3. predict_date() - batch predict all games on a date
    """

    @abstractmethod
    def build_context(self, league_id: str, target_date: str, **kwargs) -> BasePredictionContext:
        """
        Build a prediction context by preloading all data needed
        for predictions on the target date.
        """
        ...

    @abstractmethod
    def predict_game(self, game: Dict[str, Any], context: BasePredictionContext, **kwargs) -> Dict[str, Any]:
        """
        Generate a prediction for a single game.

        Returns a dict with at minimum:
        - home_win_prob: float (0-1)
        - away_win_prob: float (0-1)
        - draw_prob: float (0-1, optional — provided for 3-way sports like soccer)
        - model_id: str
        """
        ...

    def predict_date(self, games: List[Dict[str, Any]], context: BasePredictionContext, **kwargs) -> List[Dict[str, Any]]:
        """
        Batch predict all games. Default implementation calls
        predict_game() for each. Override for parallelism.
        """
        return [self.predict_game(game, context, **kwargs) for game in games]
