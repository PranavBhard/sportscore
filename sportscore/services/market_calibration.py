"""
Market calibration — compute Brier score and log-loss for market implied probabilities.

Pure computation module: no DB, no file I/O. Sport-agnostic.
"""

import numpy as np


def compute_market_metrics(y_true, market_probs_home, market_probs_away=None):
    """
    Compute market Brier score and log-loss from outcomes and implied probs.

    If market_probs_away is provided, performs vig removal:
        p_fair = p_home / (p_home + p_away)

    Args:
        y_true: array-like of 0/1 outcomes (1 = home win)
        market_probs_home: array-like of market implied home-win probabilities
        market_probs_away: optional array-like of market implied away-win probs
            (for vig removal). If None, market_probs_home is used as-is.

    Returns:
        dict with:
            'market_brier': float — market Brier score
            'market_log_loss': float — market log-loss
            'n_games': int — number of games used
            'avg_overround': float — mean(p_home + p_away) before normalization
                (1.0 if no away probs provided)
    """
    y = np.asarray(y_true, dtype=float)
    p_home = np.asarray(market_probs_home, dtype=float)

    if market_probs_away is not None:
        p_away = np.asarray(market_probs_away, dtype=float)
        overround = p_home + p_away
        avg_overround = float(np.mean(overround))
        # Vig removal: normalize to fair probabilities
        p_fair = p_home / overround
    else:
        avg_overround = 1.0
        p_fair = p_home

    # Clip to avoid log(0)
    eps = 1e-15
    p_fair = np.clip(p_fair, eps, 1.0 - eps)

    n = len(y)

    # Brier score
    brier = float(np.mean((p_fair - y) ** 2))

    # Log-loss
    log_loss = float(-np.mean(y * np.log(p_fair) + (1 - y) * np.log(1 - p_fair)))

    return {
        'market_brier': round(brier, 6),
        'market_log_loss': round(log_loss, 6),
        'n_games': n,
        'avg_overround': round(avg_overround, 4),
    }
