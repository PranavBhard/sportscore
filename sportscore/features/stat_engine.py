"""
Stat Engine - Sport-agnostic feature computation engine.

Implements the 4-layer computation model:
  Layer 1: Raw stat extraction    (per-game, per-team from game docs)
  Layer 2: Time period windowing  (season, last_N, career, none)
  Layer 3: Calc weight            (raw=aggregate, avg=per-game average)
  Layer 4: Perspective            (diff, home, away, none)

Complex stats use type=custom + Python handler functions dispatched by name.
"""

import re
from math import exp
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sportscore.features.base_registry import StatDefinition


class StatEngine:
    """
    Sport-agnostic computation engine for declarative stat definitions.

    Game documents are expected to have:
      - homeTeam / awayTeam sub-dicts
      - Each sub-dict has a 'name' field for team identification
      - Stat fields live inside the team sub-dicts
    """

    def __init__(
        self,
        stat_definitions: Dict[str, StatDefinition],
        custom_handlers: Optional[Dict[str, Callable]] = None,
        recency_alpha: float = 0.0,
        valid_time_periods: Optional[Set[str]] = None,
    ):
        self.stat_definitions = stat_definitions
        self.custom_handlers = custom_handlers or {}
        self.recency_alpha = recency_alpha
        self.valid_time_periods = valid_time_periods or {
            "season", "last_3", "last_5", "last_8", "career", "none",
        }

    # =========================================================================
    # Public API
    # =========================================================================

    def compute_feature(
        self,
        feature_name: str,
        home_team: str,
        away_team: str,
        home_games: List[Dict],
        away_games: List[Dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Compute a single feature value from a pipe-delimited feature name."""
        parts = feature_name.split("|")
        if len(parts) != 4:
            return None

        stat_name, time_period, calc_weight, perspective = parts
        stat_def = self.stat_definitions.get(stat_name)
        if stat_def is None:
            return None

        ctx = context or {}

        # Custom handler dispatch
        if stat_def.stat_type == "custom" and stat_def.handler:
            handler_fn = self.custom_handlers.get(stat_def.handler)
            if handler_fn is None:
                return None
            return handler_fn(
                stat_name, time_period, calc_weight, perspective,
                home_team, away_team, home_games, away_games,
                **ctx,
            )

        # Standard computation via 4-layer model
        year = ctx.get("year", 0)
        week = ctx.get("week", 0)

        # Layer 2: window games
        home_windowed = self._window_games(home_games, time_period)
        away_windowed = self._window_games(away_games, time_period)

        # Layer 1+3: extract and aggregate
        home_val = self._aggregate_stat(home_windowed, home_team, stat_def, calc_weight, year, week)
        away_val = self._aggregate_stat(away_windowed, away_team, stat_def, calc_weight, year, week)

        # Layer 4: perspective
        return self._apply_perspective(home_val, away_val, perspective)

    def compute_features(
        self,
        feature_names: List[str],
        home_team: str,
        away_team: str,
        home_games: List[Dict],
        away_games: List[Dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[float]]:
        """Compute multiple features at once."""
        return {
            name: self.compute_feature(name, home_team, away_team, home_games, away_games, context)
            for name in feature_names
        }

    # =========================================================================
    # Layer 1: Raw stat extraction
    # =========================================================================

    def _get_team_data(self, game: Dict, team_name: str) -> Tuple[Dict, Dict, bool]:
        """Return (team_data, opponent_data, is_home) for a game."""
        is_home = game.get("homeTeam", {}).get("name") == team_name
        if is_home:
            return game["homeTeam"], game["awayTeam"], True
        return game["awayTeam"], game["homeTeam"], False

    def _extract_basic_value(self, game: Dict, team_name: str, stat_def: StatDefinition) -> Optional[float]:
        """Layer 1: extract a single value from a game document."""
        team_data, opp_data, is_home = self._get_team_data(game, team_name)

        # Formula-based (e.g. "team.points - opponent.points")
        if stat_def.formula:
            return self._eval_formula(stat_def.formula, team_data, opp_data)

        # db_fields list (e.g. ["won"] for wins → converts bool to 1/0)
        if stat_def.db_fields:
            total = 0
            for field_name in stat_def.db_fields:
                val = team_data.get(field_name)
                if val is None:
                    return None
                if isinstance(val, bool):
                    val = 1 if val else 0
                total += val
            return float(total)

        # Single db_field
        if stat_def.db_field:
            val = team_data.get(stat_def.db_field)
            if val is None:
                return None
            if isinstance(val, bool):
                val = 1 if val else 0
            return float(val)

        return None

    def _extract_rate_components(
        self, game: Dict, team_name: str, stat_def: StatDefinition
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract numerator and denominator values from a game for rate stats."""
        team_data, _, _ = self._get_team_data(game, team_name)

        if not stat_def.numerator or not stat_def.denominator:
            return None, None

        num = sum(team_data.get(f, 0) or 0 for f in stat_def.numerator)
        denom = sum(team_data.get(f, 0) or 0 for f in stat_def.denominator)

        # Apply adjustments (e.g. +20*passTD, -45*ints for pyper_adj)
        if stat_def.adjustments:
            for adj in stat_def.adjustments:
                adj_val = team_data.get(adj["field"], 0) or 0
                num += adj["weight"] * adj_val

        return float(num), float(denom)

    # =========================================================================
    # Layer 2: Time period windowing
    # =========================================================================

    def _window_games(self, games: List[Dict], time_period: str) -> List[Dict]:
        """Slice games list according to the time period."""
        if time_period in ("season", "none", "career"):
            return games

        match = re.match(r"last_(\d+)", time_period)
        if match:
            n = int(match.group(1))
            return games[-n:] if len(games) > n else games

        return games

    # =========================================================================
    # Layer 3: Aggregation (calc_weight)
    # =========================================================================

    def _aggregate_stat(
        self,
        games: List[Dict],
        team_name: str,
        stat_def: StatDefinition,
        calc_weight: str,
        year: int,
        week: int,
    ) -> Optional[float]:
        """Route to the correct aggregation method based on stat definition."""
        if not games:
            return None

        # Rate stats with numerator/denominator
        if stat_def.numerator and stat_def.denominator:
            return self._compute_rate_aggregate(games, team_name, stat_def, calc_weight, year, week)

        # Pre-computed rate field (e.g. qbRTG)
        if stat_def.precomputed_field:
            return self._compute_precomputed_aggregate(games, team_name, stat_def, calc_weight, year, week)

        # Basic stats (formula, db_fields, db_field)
        return self._compute_basic_aggregate(games, team_name, stat_def, calc_weight, year, week)

    def _compute_basic_aggregate(
        self, games, team_name, stat_def, calc_weight, year, week
    ) -> Optional[float]:
        """Aggregate a basic stat: raw=weighted sum, avg=weighted average."""
        total = 0.0
        weight_sum = 0.0

        for game in games:
            val = self._extract_basic_value(game, team_name, stat_def)
            if val is None:
                continue

            w = self._get_recency_weight(week, game, year)
            total += val * w
            weight_sum += w

        if weight_sum == 0:
            return None

        if calc_weight == "raw":
            return total
        # avg
        return total / weight_sum

    def _compute_rate_aggregate(
        self, games, team_name, stat_def, calc_weight, year, week
    ) -> Optional[float]:
        """Aggregate a rate stat: raw=sum(num)/sum(denom), avg=avg of per-game rates."""
        if calc_weight == "raw":
            num_sum = 0.0
            denom_sum = 0.0

            for game in games:
                num, denom = self._extract_rate_components(game, team_name, stat_def)
                if num is None:
                    continue

                w = self._get_recency_weight(week, game, year)
                num_sum += num * w
                denom_sum += denom * w

            if denom_sum == 0:
                return None
            return num_sum / denom_sum

        else:  # avg
            rate_sum = 0.0
            weight_sum = 0.0

            for game in games:
                num, denom = self._extract_rate_components(game, team_name, stat_def)
                if num is None or denom == 0:
                    continue

                rate = num / denom
                w = self._get_recency_weight(week, game, year)
                rate_sum += rate * w
                weight_sum += w

            if weight_sum == 0:
                return None
            return rate_sum / weight_sum

    def _compute_precomputed_aggregate(
        self, games, team_name, stat_def, calc_weight, year, week
    ) -> Optional[float]:
        """Aggregate a pre-computed per-game rate field (always returns weighted avg)."""
        total = 0.0
        weight_sum = 0.0

        for game in games:
            team_data, _, _ = self._get_team_data(game, team_name)
            val = team_data.get(stat_def.precomputed_field)
            if val is None:
                continue

            w = self._get_recency_weight(week, game, year)
            total += float(val) * w
            weight_sum += w

        if weight_sum == 0:
            return None
        return total / weight_sum

    # =========================================================================
    # Layer 4: Perspective
    # =========================================================================

    def _apply_perspective(
        self, home_val: Optional[float], away_val: Optional[float], perspective: str
    ) -> Optional[float]:
        """Combine home/away values based on perspective."""
        if perspective == "diff":
            if home_val is not None and away_val is not None:
                return home_val - away_val
            return None
        elif perspective == "home":
            return home_val
        elif perspective == "away":
            return away_val
        elif perspective == "none":
            return home_val  # For matchup-level stats, home_val is the value
        return None

    # =========================================================================
    # Recency weighting
    # =========================================================================

    def _get_recency_weight(self, current_week: int, game: Dict, current_year: int) -> float:
        """Compute exponential recency weight for a game."""
        if self.recency_alpha <= 0:
            return 1.0

        game_week = game.get("_week", game.get("week", current_week))
        game_year = game.get("_year", game.get("year", current_year))

        if game_year < current_year:
            weeks_ago = (current_year - game_year) * 17 + (current_week - game_week)
        else:
            weeks_ago = max(current_week - game_week, 0)

        return exp(-self.recency_alpha * weeks_ago)

    # =========================================================================
    # Formula evaluation (safe, no eval())
    # =========================================================================

    def _eval_formula(self, formula: str, team_data: Dict, opp_data: Dict) -> Optional[float]:
        """
        Evaluate a simple formula like 'team.points - opponent.points'.

        Supports +, -, *, / operators and team.X / opponent.X field references.
        """
        # Tokenize: split on operators while keeping them
        tokens = re.split(r'\s*([+\-*/])\s*', formula.strip())
        if not tokens:
            return None

        def _resolve(token: str) -> Optional[float]:
            token = token.strip()
            if token.startswith("team."):
                field = token[5:]
                val = team_data.get(field)
            elif token.startswith("opponent."):
                field = token[9:]
                val = opp_data.get(field)
            else:
                try:
                    return float(token)
                except ValueError:
                    return None
            if val is None:
                return None
            if isinstance(val, bool):
                val = 1 if val else 0
            return float(val)

        # Evaluate left to right (no operator precedence needed for simple formulas)
        result = _resolve(tokens[0])
        if result is None:
            return None

        i = 1
        while i < len(tokens) - 1:
            op = tokens[i]
            right = _resolve(tokens[i + 1])
            if right is None:
                return None

            if op == "+":
                result += right
            elif op == "-":
                result -= right
            elif op == "*":
                result *= right
            elif op == "/":
                if right == 0:
                    return None
                result /= right
            i += 2

        return result
