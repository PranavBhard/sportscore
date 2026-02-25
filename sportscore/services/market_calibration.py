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


def compute_market_metrics_3way(y_outcome, market_probs_home, market_probs_draw, market_probs_away):
    """
    3-way market calibration (home/draw/away).

    Args:
        y_outcome: array-like of integers — 2=home win, 1=draw, 0=away win
        market_probs_home: array-like of raw implied home-win probs (pre-vig)
        market_probs_draw: array-like of raw implied draw probs (pre-vig)
        market_probs_away: array-like of raw implied away-win probs (pre-vig)

    Returns:
        dict with 'market_brier', 'market_log_loss', 'n_games', 'avg_overround'
    """
    y = np.asarray(y_outcome, dtype=int)
    p_home = np.asarray(market_probs_home, dtype=float)
    p_draw = np.asarray(market_probs_draw, dtype=float)
    p_away = np.asarray(market_probs_away, dtype=float)

    n = len(y)

    # Vig removal: normalize all 3 to sum to 1
    total = p_home + p_draw + p_away
    avg_overround = float(np.mean(total))
    p_h = p_home / total
    p_d = p_draw / total
    p_a = p_away / total

    # Build (n, 3) probability matrix: columns = [away, draw, home]
    # Column index matches y_outcome encoding (0=away, 1=draw, 2=home)
    eps = 1e-15
    p_matrix = np.clip(np.column_stack([p_a, p_d, p_h]), eps, 1.0 - eps)

    # One-hot encode outcomes
    y_onehot = np.zeros((n, 3))
    y_onehot[np.arange(n), y] = 1.0

    # Multiclass Brier: mean per-game sum of squared errors
    brier = float(np.mean(np.sum((p_matrix - y_onehot) ** 2, axis=1)))

    # Multiclass log-loss: mean per-game cross-entropy
    log_loss = float(np.mean(-np.sum(y_onehot * np.log(p_matrix), axis=1)))

    return {
        'market_brier': round(brier, 6),
        'market_log_loss': round(log_loss, 6),
        'n_games': n,
        'avg_overround': round(avg_overround, 4),
    }
