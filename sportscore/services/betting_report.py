"""
Betting Report Service - Generate betting recommendations based on model vs market edge.

Compares model predictions against market odds and recommends stakes
using a Kelly Criterion-based approach with confidence adjustments.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from pytz import timezone

if TYPE_CHECKING:
    from sportscore.league_config import BaseLeagueConfig


@dataclass
class BetRecommendation:
    """A single betting recommendation."""
    game_id: str
    game_time: datetime  # UTC
    game_time_formatted: str  # "0700PM" format
    team: str  # Team abbreviation (predicted winner)
    home_team: str  # Home team abbreviation
    away_team: str  # Away team abbreviation
    market_prob: float  # 0.0-1.0
    market_odds: int  # American odds
    model_prob: float  # 0.0-1.0
    model_odds: int  # American odds
    edge: float  # p_model - p_market
    edge_kelly: float  # Kelly edge
    dog_variance_penalty: float  # Underdog probability coefficient
    stake_fraction: float  # Fraction of bankroll
    stake: float  # Actual stake amount
    adjusted_stake: float  # Adjusted stake with probability-based confidence
    market_status: str  # Market status: "active", "closed", "settled", etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = asdict(self)
        if isinstance(result['game_time'], datetime):
            result['game_time'] = result['game_time'].isoformat()
        return result


def prob_to_american_odds(prob: float) -> int:
    """
    Convert probability (0.0-1.0) to American odds.

    Examples:
        0.60 -> -150 (60% favorite)
        0.40 -> +150 (40% underdog)
        0.50 -> -100 (even)
    """
    if prob <= 0:
        return 0
    if prob >= 1:
        return -10000
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


def format_time_for_report(dt: datetime, tz_name: str = 'US/Eastern') -> str:
    """
    Format datetime as '0700PM' in the given timezone.

    Args:
        dt: datetime object (assumed UTC if naive)
        tz_name: Timezone name (default: 'US/Eastern')

    Returns:
        String like "0700PM" or "1030PM"
    """
    if dt is None:
        return "TBD"

    tz = timezone(tz_name)

    if dt.tzinfo is None:
        from pytz import utc
        dt = utc.localize(dt)

    local_time = dt.astimezone(tz)
    return local_time.strftime('%I%M%p').upper()


def calculate_stake(
    p_model: float,
    p_market: float,
    brier_score: float,
    bankroll: float
) -> Dict[str, float]:
    """
    Calculate recommended stake using Kelly Criterion with adjustments.

    Args:
        p_model: Model probability (0.0-1.0)
        p_market: Market probability (0.0-1.0)
        brier_score: Model's Brier score (lower is better)
        bankroll: Total bankroll amount

    Returns:
        Dict with edge_kelly, kelly_fraction, confidence, dog_variance_penalty,
        stake_fraction, stake, adjusted_confidence, adjusted_stake_fraction, adjusted_stake
    """
    market_odds_decimal = 1 / p_market if p_market > 0 else 100

    if market_odds_decimal <= 1:
        edge_kelly = 0
    else:
        edge_kelly = (p_model * market_odds_decimal - 1) / (market_odds_decimal - 1)

    kelly_fraction = 0.25

    raw_confidence = 1 - brier_score / 0.25
    confidence = max(0.3, min(1.0, raw_confidence))

    dog_variance_penalty = min(1.0, p_model / 0.30)

    stake_fraction = max(0, edge_kelly * kelly_fraction * confidence * dog_variance_penalty)
    stake = bankroll * stake_fraction

    confidence_base = max(0.3, min(1.0, raw_confidence))
    mult = max(0.5, min(1.2, p_model / 0.5))
    adjusted_confidence = max(0.3, min(1.0, confidence_base * mult))

    adjusted_stake_fraction = max(0, edge_kelly * kelly_fraction * adjusted_confidence * dog_variance_penalty)
    adjusted_stake = bankroll * adjusted_stake_fraction

    return {
        'edge_kelly': edge_kelly,
        'kelly_fraction': kelly_fraction,
        'confidence': confidence,
        'dog_variance_penalty': dog_variance_penalty,
        'stake_fraction': stake_fraction,
        'stake': stake,
        'adjusted_confidence': adjusted_confidence,
        'adjusted_stake_fraction': adjusted_stake_fraction,
        'adjusted_stake': adjusted_stake
    }
