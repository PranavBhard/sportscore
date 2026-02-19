"""
Bin Trust Weights — compute empirical trust from market P&L bins.

Pure computation module — no DB, no I/O. Takes the output of
compute_market_bins() and derives per-bin trust weights that replace
the hardcoded underdog variance penalty in calculate_stake().

Trust = "does our edge actually materialize in this probability range?"
"""
from datetime import datetime, timezone
from typing import List, Optional


def compute_bin_trust_weights(
    bins_result: dict,
    *,
    min_samples: int = 10,
    default_trust: float = 1.0,
    min_trust: float = 0.1,
    max_trust: float = 1.5,
    smooth_neighbors: int = 1,
) -> List[dict]:
    """
    Compute per-bin trust weights from historical P&L data.

    For each bin with enough data:
      raw_trust = actual_roi / expected_roi  (clamped to [min_trust, max_trust])

    Sample-size shrinkage: blend toward default_trust when count is low.
      effective_trust = (count / (count + min_samples)) * raw_trust
                      + (min_samples / (count + min_samples)) * default_trust

    Neighbor smoothing: weighted average with ±1 adjacent bins (weight=0.5).

    Args:
        bins_result: Output of compute_market_bins() — must have 'bins' key.
        min_samples: Below this count, trust shrinks heavily toward default.
        default_trust: Neutral trust (1.0 = no adjustment).
        min_trust: Floor clamp for final trust.
        max_trust: Ceiling clamp for final trust.
        smooth_neighbors: Number of adjacent bins to average with (0 = none).

    Returns:
        List of dicts with keys: prob_low, prob_high, trust, raw_trust,
        count, roi, shrunk.
    """
    bins = bins_result.get('bins', [])
    if not bins:
        return []

    # Step 1: Compute raw trust per bin
    # Trust = 1.0 + roi/100: direct mapping from actual P&L performance.
    #   ROI of  0%  → trust 1.0 (break even, neutral)
    #   ROI of +50% → trust 1.5 (boost stakes)
    #   ROI of -50% → trust 0.5 (cut stakes in half)
    #   ROI of -100% → trust 0.0, clamped to min_trust
    raw_entries = []
    for b in bins:
        count = b.get('count', 0)
        roi = b.get('roi', 0)  # percentage, e.g. 15.0 means +15%

        if count == 0:
            raw_entries.append({
                'prob_low': b['low'],
                'prob_high': b['high'],
                'raw_trust': default_trust,
                'count': count,
                'roi': roi,
            })
            continue

        raw_trust = 1.0 + roi / 100.0
        raw_trust = max(min_trust, min(max_trust, raw_trust))

        raw_entries.append({
            'prob_low': b['low'],
            'prob_high': b['high'],
            'raw_trust': raw_trust,
            'count': count,
            'roi': roi,
        })

    # Step 2: Sample-size shrinkage
    shrunk_entries = []
    for e in raw_entries:
        count = e['count']
        weight = count / (count + min_samples) if (count + min_samples) > 0 else 0
        shrunk_trust = weight * e['raw_trust'] + (1.0 - weight) * default_trust
        shrunk_entries.append({
            **e,
            'shrunk': round(shrunk_trust, 4),
        })

    # Step 3: Neighbor smoothing
    n = len(shrunk_entries)
    if smooth_neighbors > 0 and n > 1:
        smoothed = []
        for i in range(n):
            total_weight = 1.0
            weighted_sum = shrunk_entries[i]['shrunk']

            for offset in range(1, smooth_neighbors + 1):
                neighbor_weight = 0.5 ** offset
                if i - offset >= 0:
                    weighted_sum += neighbor_weight * shrunk_entries[i - offset]['shrunk']
                    total_weight += neighbor_weight
                if i + offset < n:
                    weighted_sum += neighbor_weight * shrunk_entries[i + offset]['shrunk']
                    total_weight += neighbor_weight

            smoothed_trust = weighted_sum / total_weight
            smoothed_trust = max(min_trust, min(max_trust, smoothed_trust))
            smoothed.append({
                'prob_low': shrunk_entries[i]['prob_low'],
                'prob_high': shrunk_entries[i]['prob_high'],
                'trust': round(smoothed_trust, 4),
                'raw_trust': round(shrunk_entries[i]['raw_trust'], 4),
                'count': shrunk_entries[i]['count'],
                'roi': shrunk_entries[i]['roi'],
                'shrunk': shrunk_entries[i]['shrunk'],
            })
        return smoothed

    # No smoothing — just clamp and return
    return [
        {
            'prob_low': e['prob_low'],
            'prob_high': e['prob_high'],
            'trust': max(min_trust, min(max_trust, e['shrunk'])),
            'raw_trust': round(e['raw_trust'], 4),
            'count': e['count'],
            'roi': e['roi'],
            'shrunk': e['shrunk'],
        }
        for e in shrunk_entries
    ]


def lookup_trust(
    trust_weights: List[dict],
    p_adj: float,
) -> float:
    """
    Look up trust weight for a given p_adj (0.0-1.0 scale).

    Finds the bin containing p_adj * 100, returns its trust value.
    Falls back to 1.0 if no matching bin.
    """
    if not trust_weights:
        return 1.0

    prob_pct = p_adj * 100.0

    for tw in trust_weights:
        low = tw.get('prob_low', 0)
        high = tw.get('prob_high', 100)
        if low <= prob_pct < high or (high >= 100 and prob_pct >= low):
            return tw.get('trust', 1.0)

    return 1.0


def trust_weights_to_doc(
    trust_weights: List[dict],
    league_id: str,
    ticker_prefixes: List[str],
    n_positions: int = 0,
) -> dict:
    """Format trust weights as a MongoDB document for storage."""
    return {
        'league_id': league_id,
        'bins': trust_weights,
        'ticker_prefixes': ticker_prefixes,
        'computed_at': datetime.now(timezone.utc),
        'n_positions': n_positions,
    }
