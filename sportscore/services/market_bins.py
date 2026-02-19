"""
Market P&L Bins Analysis — sport-agnostic pure computation.

Fetches all portfolio data via a MarketConnector, filters by ticker prefix,
groups fills into positions, joins with settlements, and buckets P&L by
implied-probability bins.
"""
from typing import List, Dict, Optional, Tuple, Any


# ── Fetch helpers (paginated) ───────────────────────────────────────────

def fetch_all_fills(connector) -> List[dict]:
    """Paginated fetch of ALL fills from portfolio API."""
    all_fills = []
    cursor = None
    while True:
        resp = connector.get_fills(limit=100, cursor=cursor)
        page = resp.get('fills', [])
        if not page:
            break
        all_fills.extend(page)
        cursor = resp.get('cursor')
        if not cursor:
            break
    return all_fills


def fetch_all_settlements(connector) -> List[dict]:
    """Paginated fetch of ALL settlements from portfolio API."""
    all_settlements = []
    cursor = None
    while True:
        resp = connector.get_settlements(limit=100, cursor=cursor)
        page = resp.get('settlements', [])
        if not page:
            break
        all_settlements.extend(page)
        cursor = resp.get('cursor')
        if not cursor:
            break
    return all_settlements


# ── Internal helpers ────────────────────────────────────────────────────

def _filter_by_prefix(items: List[dict], prefixes: List[str]) -> List[dict]:
    """Keep items whose ticker starts with any of the given prefixes."""
    if not prefixes:
        return items
    return [
        item for item in items
        if any(item.get('ticker', '').startswith(p) for p in prefixes)
    ]


def _compute_position_implied_prob(fills: List[dict]) -> Optional[float]:
    """
    Compute weighted-average implied probability from buy fills for a single ticker.

    YES buys: implied prob = weighted avg of yes_price (higher price = higher prob)
    NO buys:  implied prob = 100 - weighted avg of no_price (buying NO cheap = high prob for YES)

    Only buy fills define the entry point; sells are excluded.
    Returns implied probability in 0-100 scale, or None if no buy fills.
    """
    yes_cost = 0
    yes_count = 0
    no_cost = 0
    no_count = 0

    for f in fills:
        if f.get('action') != 'buy':
            continue
        side = f.get('side', '')
        count = f.get('count', 0)
        if side == 'yes':
            yes_cost += f.get('yes_price', 0) * count
            yes_count += count
        elif side == 'no':
            no_cost += f.get('no_price', 0) * count
            no_count += count

    if yes_count > 0 and no_count > 0:
        # Mixed position — use net-weighted implied prob
        yes_avg = yes_cost / yes_count  # cents, 0-100 scale
        no_avg = no_cost / no_count
        # YES side implied prob is yes_avg; NO side implies prob of (100-no_avg)
        # Weight by total contracts on each side
        total = yes_count + no_count
        return (yes_avg * yes_count + (100 - no_avg) * no_count) / total
    elif yes_count > 0:
        return yes_cost / yes_count
    elif no_count > 0:
        return 100 - (no_cost / no_count)
    return None


def _implied_prob_to_american(prob: float) -> str:
    """Convert implied probability (0-100) to American odds string."""
    if prob <= 0 or prob >= 100:
        return "N/A"
    if prob >= 50:
        odds = -(prob / (100 - prob)) * 100
        return f"{odds:+.0f}"
    else:
        odds = ((100 - prob) / prob) * 100
        return f"+{odds:.0f}"


def _format_bin_label(low: float, high: float, bin_mode: str) -> str:
    """Format a bin range as a human-readable label."""
    if bin_mode == "american_odds":
        lo_odds = _implied_prob_to_american(low if low > 0 else 0.5)
        hi_odds = _implied_prob_to_american(high if high < 100 else 99.5)
        return f"{hi_odds} to {lo_odds}"
    return f"{low:.0f}-{high:.0f}%"


# ── Core analysis ───────────────────────────────────────────────────────

DEFAULT_BINS = [(i, i + 10) for i in range(0, 100, 10)]


def compute_market_bins(
    fills: List[dict],
    settlements: List[dict],
    ticker_prefixes: List[str],
    *,
    bins: Optional[List[Tuple[float, float]]] = None,
    bin_mode: str = "implied_prob",
) -> Dict[str, Any]:
    """
    Core analysis function.

    Steps:
    1. Filter fills/settlements to those matching any ticker_prefix
    2. Group fills by ticker → compute weighted-avg entry price (implied prob)
    3. Join with settlements by ticker → get revenue, cost, P&L
    4. Assign each position to a bin based on implied prob
    5. Aggregate per-bin: count, won, win%, cost, revenue, pnl, roi

    Returns dict with keys: bins, totals, n_positions, n_fills, n_unsettled
    """
    if bins is None:
        bins = DEFAULT_BINS

    # 1. Filter by ticker prefix
    filtered_fills = _filter_by_prefix(fills, ticker_prefixes)
    filtered_settlements = _filter_by_prefix(settlements, ticker_prefixes)

    # 2. Group fills by ticker
    fills_by_ticker: Dict[str, List[dict]] = {}
    for f in filtered_fills:
        ticker = f.get('ticker', '')
        if ticker:
            fills_by_ticker.setdefault(ticker, []).append(f)

    # Build settlement lookup by ticker
    settlement_by_ticker: Dict[str, dict] = {}
    for s in filtered_settlements:
        ticker = s.get('ticker', '')
        if ticker:
            settlement_by_ticker[ticker] = s

    # 3. Build positions: implied prob + P&L from settlement
    positions = []
    n_unsettled = 0

    for ticker, ticker_fills in fills_by_ticker.items():
        implied_prob = _compute_position_implied_prob(ticker_fills)
        if implied_prob is None:
            continue

        settlement = settlement_by_ticker.get(ticker)
        if not settlement:
            n_unsettled += 1
            continue

        revenue = settlement.get('revenue', 0)
        yes_cost = settlement.get('yes_total_cost', 0)
        no_cost = settlement.get('no_total_cost', 0)
        cost = yes_cost + no_cost
        pnl = revenue - cost

        positions.append({
            'ticker': ticker,
            'implied_prob': implied_prob,
            'cost': cost,
            'revenue': revenue,
            'pnl': pnl,
            'won': pnl > 0,
        })

    # 4. Assign positions to bins
    bin_results = []
    for (low, high) in bins:
        bucket_positions = [
            p for p in positions
            if low <= p['implied_prob'] < high
            or (high == 100 and p['implied_prob'] == 100)
        ]

        count = len(bucket_positions)
        won = sum(1 for p in bucket_positions if p['won'])
        cost = sum(p['cost'] for p in bucket_positions)
        revenue = sum(p['revenue'] for p in bucket_positions)
        pnl = sum(p['pnl'] for p in bucket_positions)
        win_pct = (won / count * 100) if count > 0 else 0
        roi = (pnl / cost * 100) if cost > 0 else 0

        bin_results.append({
            'low': low,
            'high': high,
            'label': _format_bin_label(low, high, bin_mode),
            'count': count,
            'won': won,
            'win_pct': round(win_pct, 1),
            'cost': cost,
            'revenue': revenue,
            'pnl': pnl,
            'roi': round(roi, 1),
        })

    # 5. Totals
    total_count = len(positions)
    total_won = sum(1 for p in positions if p['won'])
    total_cost = sum(p['cost'] for p in positions)
    total_revenue = sum(p['revenue'] for p in positions)
    total_pnl = sum(p['pnl'] for p in positions)

    totals = {
        'count': total_count,
        'won': total_won,
        'win_pct': round((total_won / total_count * 100) if total_count > 0 else 0, 1),
        'cost': total_cost,
        'revenue': total_revenue,
        'pnl': total_pnl,
        'roi': round((total_pnl / total_cost * 100) if total_cost > 0 else 0, 1),
    }

    return {
        'bins': bin_results,
        'totals': totals,
        'n_positions': total_count,
        'n_fills': len(filtered_fills),
        'n_unsettled': n_unsettled,
    }
