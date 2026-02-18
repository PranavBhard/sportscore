"""
Stat Engine - Sport-agnostic feature computation engine.

Implements the 4-layer computation model:
  Layer 1: Raw stat extraction    (per-game, per-team from game docs)
  Layer 2: Time period windowing  (season, last_N, games_N, days_N, months_N, career, none)
  Layer 3: Calc weight            (raw=aggregate, avg=per-game average, std=standard deviation)
  Layer 4: Perspective            (diff, home, away, none)

Feature names: stat|period|weight|perspective[|side]
  - 4-part: standard feature
  - 5-part with "side" suffix: filter to games where team was home/away

Complex stats use type=custom + Python handler functions dispatched by name.
"""

import re
from math import exp
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sportscore.features.base_registry import StatDefinition


def _date_minus_days(date_str: str, n: int) -> str:
    """YYYY-MM-DD string minus N days."""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - timedelta(days=n)).strftime("%Y-%m-%d")


def _date_minus_months(date_str: str, n: int) -> str:
    """YYYY-MM-DD string minus N months (approximate: 30 days per month)."""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - timedelta(days=n * 30)).strftime("%Y-%m-%d")


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
        recency_mode: str = "week",
        valid_time_periods: Optional[Set[str]] = None,
    ):
        self.stat_definitions = stat_definitions
        self.custom_handlers = custom_handlers or {}
        self.recency_alpha = recency_alpha
        self.recency_mode = recency_mode  # "week" (football) or "date" (basketball)
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
        """Compute a single feature value from a pipe-delimited feature name.

        Supports 4-part (stat|period|weight|perspective) and 5-part
        (stat|period|weight|perspective|side) feature names.
        """
        parts = feature_name.split("|")
        if len(parts) == 5:
            stat_name, time_period, calc_weight, perspective, side_str = parts
            has_side = (side_str == "side")
        elif len(parts) == 4:
            stat_name, time_period, calc_weight, perspective = parts
            has_side = False
        else:
            return None

        stat_def = self.stat_definitions.get(stat_name)
        if stat_def is None:
            return None

        ctx = dict(context or {})
        ctx["has_side"] = has_side

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
        reference_date = ctx.get("reference_date")

        # Layer 2: window games
        home_windowed = self._window_games(home_games, time_period, reference_date)
        away_windowed = self._window_games(away_games, time_period, reference_date)

        # Side filter: keep only games where team was actually home/away
        if has_side:
            home_windowed = [g for g in home_windowed if g.get("homeTeam", {}).get("name") == home_team]
            away_windowed = [g for g in away_windowed if g.get("awayTeam", {}).get("name") == away_team]

        # Layer 1+3: extract and aggregate
        home_val = self._aggregate_stat(home_windowed, home_team, stat_def, calc_weight, year, week, ctx)
        away_val = self._aggregate_stat(away_windowed, away_team, stat_def, calc_weight, year, week, ctx)

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

    def _window_games(self, games: List[Dict], time_period: str,
                      reference_date: str = None) -> List[Dict]:
        """Slice games list according to the time period.

        Supports:
          - season, none, career: return all games
          - last_N: last N games (football-style)
          - games_N: last N games (basketball-style alias)
          - days_N: games within last N days (requires reference_date)
          - months_N: games within last N months (requires reference_date)
        """
        if time_period in ("season", "none", "career"):
            return games

        # games_N — last N games (basketball's primary windowing)
        match = re.match(r"games_(\d+)$", time_period)
        if match:
            n = int(match.group(1))
            return games[-n:] if len(games) > n else games

        # last_N — existing pattern (football)
        match = re.match(r"last_(\d+)$", time_period)
        if match:
            n = int(match.group(1))
            return games[-n:] if len(games) > n else games

        # days_N — games within last N days (requires reference_date)
        match = re.match(r"days_(\d+)$", time_period)
        if match and reference_date:
            n_days = int(match.group(1))
            cutoff = _date_minus_days(reference_date, n_days)
            return [g for g in games if g.get("date", "") >= cutoff]

        # months_N — games within last N months (requires reference_date)
        match = re.match(r"months_(\d+)$", time_period)
        if match and reference_date:
            n_months = int(match.group(1))
            cutoff = _date_minus_months(reference_date, n_months)
            return [g for g in games if g.get("date", "") >= cutoff]

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
        context: Optional[Dict] = None,
    ) -> Optional[float]:
        """Route to the correct aggregation method based on stat definition."""
        if not games:
            return None

        # Rate stats with numerator/denominator
        if stat_def.numerator and stat_def.denominator:
            return self._compute_rate_aggregate(games, team_name, stat_def, calc_weight, year, week, context)

        # Pre-computed rate field (e.g. qbRTG)
        if stat_def.precomputed_field:
            return self._compute_precomputed_aggregate(games, team_name, stat_def, calc_weight, year, week, context)

        # Basic stats (formula, db_fields, db_field)
        return self._compute_basic_aggregate(games, team_name, stat_def, calc_weight, year, week, context)

    def _compute_basic_aggregate(
        self, games, team_name, stat_def, calc_weight, year, week,
        context=None,
    ) -> Optional[float]:
        """Aggregate a basic stat: raw=weighted sum, avg=weighted average, std=standard deviation."""
        if calc_weight == "std":
            values = []
            for game in games:
                val = self._extract_basic_value(game, team_name, stat_def)
                if val is not None:
                    values.append(val)
            if len(values) < 2:
                return None
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            return variance ** 0.5

        total = 0.0
        weight_sum = 0.0

        for game in games:
            val = self._extract_basic_value(game, team_name, stat_def)
            if val is None:
                continue

            w = self._get_recency_weight(week, game, year, context)
            total += val * w
            weight_sum += w

        if weight_sum == 0:
            return None

        if calc_weight == "raw":
            return total
        # avg
        return total / weight_sum

    def _compute_rate_aggregate(
        self, games, team_name, stat_def, calc_weight, year, week,
        context=None,
    ) -> Optional[float]:
        """Aggregate a rate stat: raw=sum(num)/sum(denom), avg=avg of per-game rates, std=std of per-game rates."""
        if calc_weight == "std":
            values = []
            for game in games:
                num, denom = self._extract_rate_components(game, team_name, stat_def)
                if num is not None and denom and denom != 0:
                    values.append(num / denom)
            if len(values) < 2:
                return None
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            return variance ** 0.5

        if calc_weight == "raw":
            num_sum = 0.0
            denom_sum = 0.0

            for game in games:
                num, denom = self._extract_rate_components(game, team_name, stat_def)
                if num is None:
                    continue

                w = self._get_recency_weight(week, game, year, context)
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
                w = self._get_recency_weight(week, game, year, context)
                rate_sum += rate * w
                weight_sum += w

            if weight_sum == 0:
                return None
            return rate_sum / weight_sum

    def _compute_precomputed_aggregate(
        self, games, team_name, stat_def, calc_weight, year, week,
        context=None,
    ) -> Optional[float]:
        """Aggregate a pre-computed per-game rate field."""
        if calc_weight == "std":
            values = []
            for game in games:
                team_data, _, _ = self._get_team_data(game, team_name)
                val = team_data.get(stat_def.precomputed_field)
                if val is not None:
                    values.append(float(val))
            if len(values) < 2:
                return None
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            return variance ** 0.5

        total = 0.0
        weight_sum = 0.0

        for game in games:
            team_data, _, _ = self._get_team_data(game, team_name)
            val = team_data.get(stat_def.precomputed_field)
            if val is None:
                continue

            w = self._get_recency_weight(week, game, year, context)
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

    def _get_recency_weight(self, current_week: int, game: Dict, current_year: int,
                            context: Optional[Dict] = None) -> float:
        """Compute exponential recency weight for a game.

        Supports two modes:
          - "week" (default/football): weight by weeks_ago
          - "date" (basketball): weight by days_ago using game['date'] and context['reference_date']
        """
        if self.recency_alpha <= 0:
            return 1.0

        if self.recency_mode == "date":
            game_date = game.get("date", "")
            ref_date = (context or {}).get("reference_date", "")
            if game_date and ref_date:
                from datetime import datetime
                try:
                    d1 = datetime.strptime(game_date, "%Y-%m-%d")
                    d2 = datetime.strptime(ref_date, "%Y-%m-%d")
                    days_ago = max((d2 - d1).days, 0)
                    return exp(-self.recency_alpha * days_ago)
                except (ValueError, TypeError):
                    return 1.0
            return 1.0

        # Week-based mode (football default)
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
