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
    edge_kelly: float  # Kelly edge (computed from p_adj)
    bin_trust_weight: float  # Empirical bin trust (replaces dog_variance_penalty)
    stake_fraction: float  # Fraction of bankroll
    stake: float  # Actual stake amount
    adjusted_stake: float  # Primary stake (same as stake in new system)
    market_status: str  # Market status: "active", "closed", "settled", etc.
    # New fields from market-relative skill system
    p_adj: float = 0.0  # Shrunk probability (w*p_model + (1-w)*p_market)
    skill: float = 0.0  # Market-relative skill score
    shrinkage_w: float = 0.5  # Shrinkage weight (how much to trust model)
    edge_gate: float = 1.0  # Edge gate multiplier (0-1)

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
    bankroll: float,
    *,
    log_loss_score: Optional[float] = None,
    market_brier: Optional[float] = None,
    market_log_loss: Optional[float] = None,
    bin_trust_weights: Optional[List[dict]] = None,
) -> Dict[str, float]:
    """
    Calculate recommended stake using Kelly Criterion with market-relative
    skill assessment, probability shrinkage, and edge gating.

    Uses a blended Brier/log-loss skill score to shrink model probability
    toward market (reducing miscalibration), then applies Kelly with an
    edge gate that ramps out tiny/illusory edges.

    Args:
        p_model: Model probability (0.0-1.0)
        p_market: Market probability (0.0-1.0)
        brier_score: Model's Brier score (lower is better)
        bankroll: Total bankroll amount
        log_loss_score: Model's log-loss (optional; Brier-only if omitted)
        market_brier: Market baseline Brier score. Should come from the
            market_calibration collection (computed by compute_market_calibration
            CLI script). Falls back to 0.21 if not provided.
        market_log_loss: Market baseline log-loss. Should come from the
            market_calibration collection. Falls back to 0.60 if not provided.

    Returns:
        Dict with stake sizing details and diagnostic fields.
    """
    if market_brier is None:
        market_brier = 0.21
    if market_log_loss is None:
        market_log_loss = 0.60

    # --- 1. Market-relative skill (blended Brier + log-loss) ---
    skill_bs = 1.0 - (brier_score / market_brier) if market_brier > 0 else 0.0

    if log_loss_score is not None and market_log_loss > 0:
        skill_ll = 1.0 - (log_loss_score / market_log_loss)
        skill = 0.6 * skill_ll + 0.4 * skill_bs
    else:
        skill = skill_bs

    # --- 2. Shrink probability toward market ---
    shrinkage_w = max(0.10, min(0.90, 0.50 + 2.0 * skill))
    p_adj = shrinkage_w * p_model + (1.0 - shrinkage_w) * p_market

    # --- 3. Kelly edge using adjusted probability ---
    market_odds_decimal = 1.0 / p_market if p_market > 0 else 100.0

    if market_odds_decimal <= 1:
        edge_kelly = 0.0
    else:
        edge_kelly = (p_adj * market_odds_decimal - 1.0) / (market_odds_decimal - 1.0)

    kelly_fraction = 0.25

    # --- 4. Edge gating: ramp from 0 at <=1% edge to full at >=5% ---
    edge = p_adj - p_market
    edge_gate = max(0.0, min(1.0, (abs(edge) - 0.01) / 0.04))

    # --- 5. Bin trust weight (replaces dog variance penalty) ---
    if bin_trust_weights:
        from sportscore.services.bin_trust import lookup_trust
        bin_trust_weight = lookup_trust(bin_trust_weights, p_adj)
    else:
        bin_trust_weight = min(1.0, p_adj / 0.30)  # Legacy fallback

    # --- 6. Final stake ---
    stake_fraction = max(0.0, edge_kelly * kelly_fraction * edge_gate * bin_trust_weight)
    stake = bankroll * stake_fraction

    # Skill-based confidence multiplier (diagnostic / legacy field)
    conf_mult = max(0.1, min(1.2, 0.5 + 1.0 * skill))

    return {
        'edge_kelly': edge_kelly,
        'kelly_fraction': kelly_fraction,
        'confidence': conf_mult,
        'bin_trust_weight': bin_trust_weight,
        'stake_fraction': stake_fraction,
        'stake': stake,
        'adjusted_confidence': conf_mult,
        'adjusted_stake_fraction': stake_fraction,
        'adjusted_stake': stake,
        # New diagnostic fields
        'p_adj': p_adj,
        'skill': skill,
        'shrinkage_w': shrinkage_w,
        'edge_gate': edge_gate,
    }
