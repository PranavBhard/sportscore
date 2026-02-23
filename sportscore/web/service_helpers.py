"""Shared service helpers for sport web plugins.

These factories eliminate boilerplate that every sport's ``build_services()``
would otherwise duplicate.  Sport-specific ``web/services.py`` files import
what they need and override only what differs.
"""

import json as json_mod
import logging
import os

logger = logging.getLogger(__name__)

# ── Process-level caches ─────────────────────────────────────────────────

_available_features_cache = {}  # league_id -> set[str]

from sportscore.market import SimpleCache

_market_portfolio_cache = SimpleCache(default_ttl=300)         # fills/settlements (5 min)
_market_dashboard_cache = SimpleCache(default_ttl=86400 * 365) # results (manual refresh)

_portfolio_db = None  # lazy-init Mongo connection for portfolio persistence

_DEFAULT_METADATA_COLS = frozenset({
    "Year", "Month", "Day", "Home", "Away", "HomeWon",
})


# ── Available features from master CSV ───────────────────────────────────

def get_available_features(league, extra_metadata=None):
    """Read column headers from the master training CSV and return feature names.

    Caches per ``league.league_id`` for process lifetime.

    Args:
        league: LeagueConfig with ``league_id`` and ``master_training_csv``.
        extra_metadata: Additional column names to exclude beyond the defaults
            (Year, Month, Day, Home, Away, HomeWon).
    """
    lid = league.league_id
    if lid in _available_features_cache:
        return _available_features_cache[lid]

    csv_path = getattr(league, "master_training_csv", None)
    if not csv_path or not os.path.exists(csv_path):
        _available_features_cache[lid] = set()
        return set()

    try:
        import pandas as pd
        cols = set(pd.read_csv(csv_path, nrows=0).columns)
        exclude = set(_DEFAULT_METADATA_COLS)
        if extra_metadata:
            exclude |= set(extra_metadata)
        feats = cols - exclude
        _available_features_cache[lid] = feats
        return feats
    except Exception as e:
        logger.warning("Failed to read master training CSV for %s: %s", lid, e)
        _available_features_cache[lid] = set()
        return set()


# ── Persistent portfolio store (MongoDB, per-fill granularity) ─────────────

def _get_portfolio_db():
    """Lazy-init a MongoDB connection for portfolio persistence."""
    global _portfolio_db
    if _portfolio_db is None:
        try:
            from sportscore.db.mongo import Mongo
            _portfolio_db = Mongo().db
        except Exception as e:
            logger.warning("Cannot connect to MongoDB for portfolio store: %s", e)
            return None
    return _portfolio_db


def _persist_portfolio(fills, settlements):
    """Upsert fills and settlements into MongoDB."""
    db = _get_portfolio_db()
    if db is None:
        return

    try:
        from pymongo import UpdateOne
        if fills:
            ops = [
                UpdateOne(
                    {"fill_id": f["fill_id"]},
                    {"$set": f},
                    upsert=True,
                )
                for f in fills if f.get("fill_id")
            ]
            if ops:
                db.kalshi_fills.bulk_write(ops, ordered=False)

        if settlements:
            ops = [
                UpdateOne(
                    {"ticker": s["ticker"]},
                    {"$set": s},
                    upsert=True,
                )
                for s in settlements if s.get("ticker")
            ]
            if ops:
                db.kalshi_settlements.bulk_write(ops, ordered=False)

        logger.debug("Persisted portfolio (%d fills, %d settlements)",
                      len(fills), len(settlements))
    except Exception as e:
        logger.warning("Failed to persist portfolio to DB: %s", e)


def _load_portfolio_from_db():
    """Load fills and settlements from MongoDB. Returns (fills, settlements) or (None, None)."""
    db = _get_portfolio_db()
    if db is None:
        return None, None

    try:
        fills = list(db.kalshi_fills.find({}, {"_id": 0}))
        settlements = list(db.kalshi_settlements.find({}, {"_id": 0}))
        if fills or settlements:
            logger.info("Loaded portfolio from DB (%d fills, %d settlements)",
                        len(fills), len(settlements))
            return fills, settlements
    except Exception as e:
        logger.warning("Failed to load portfolio from DB: %s", e)
    return None, None


def _fetch_portfolio(connector, refresh=False):
    """Fetch fills and settlements with layered caching.

    Priority: in-memory (5 min TTL) → API (persists to MongoDB) → DB fallback.
    Returns (fills, settlements).  Raises on total failure.
    """
    from sportscore.services.market_bins import fetch_all_fills, fetch_all_settlements

    if refresh:
        _market_portfolio_cache.clear()

    # 1. Try in-memory cache
    fills = _market_portfolio_cache.get('portfolio_fills')
    settlements = _market_portfolio_cache.get('portfolio_settlements')
    if fills is not None and settlements is not None:
        return fills, settlements

    # 2. Try API, persist to DB on success
    try:
        if fills is None:
            fills = fetch_all_fills(connector)
        if settlements is None:
            settlements = fetch_all_settlements(connector)
        _market_portfolio_cache.set('portfolio_fills', fills, ttl=300)
        _market_portfolio_cache.set('portfolio_settlements', settlements, ttl=300)
        _persist_portfolio(fills, settlements)
        return fills, settlements
    except Exception as api_err:
        logger.warning("API fetch failed, trying DB fallback: %s", api_err)

    # 3. Fall back to MongoDB
    fills, settlements = _load_portfolio_from_db()
    if fills is not None and settlements is not None:
        _market_portfolio_cache.set('portfolio_fills', fills, ttl=300)
        _market_portfolio_cache.set('portfolio_settlements', settlements, ttl=300)
        return fills, settlements

    raise RuntimeError("Could not fetch portfolio data from API or database")


# ── Jobs service dict ────────────────────────────────────────────────────

def make_jobs_service():
    """Return a ``JobsService`` dataclass for shared_routes."""
    from sportscore.services.jobs import (
        create_job, update_job_progress, complete_job, fail_job,
        get_job, get_running_job,
    )
    from sportscore.web.services import JobsService
    return JobsService(
        create=create_job,
        update=update_job_progress,
        complete=complete_job,
        fail=fail_job,
        get_job=get_job,
        get_running=get_running_job,
    )


# ── Market helpers (sport-agnostic) ──────────────────────────────────────

def make_market_connector():
    """Create a MarketConnector from environment variables, or None."""
    api_key = os.environ.get("KALSHI_API_KEY")
    private_key_dir = os.environ.get("KALSHI_PRIVATE_KEY_DIR")
    if not api_key or not private_key_dir:
        return None
    try:
        from sportscore.market.connector import MarketConnector
        return MarketConnector({
            "KALSHI_API_KEY": api_key,
            "KALSHI_PRIVATE_KEY_DIR": private_key_dir,
        })
    except Exception:
        return None


def market_dashboard_getter(refresh=False):
    """Account-level dashboard with computed stats, recent activity, and cumulative chart data."""
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    connector = make_market_connector()
    if not connector:
        return {"success": False, "error": "Kalshi credentials not configured (KALSHI_API_KEY, KALSHI_PRIVATE_KEY_DIR)"}
    try:
        # Balance (always fresh — cheap call)
        balance_resp = connector.get_balance()
        balance_cents = balance_resp.get("balance", 0)

        # All fills/settlements (layered cache: memory → API → DB)
        fills, settlements = _fetch_portfolio(connector, refresh=refresh)

        # ── Compute stats from settlements ────────────────────────────
        now = datetime.now(timezone.utc)
        total_return = 0
        return_24h = 0
        return_7d = 0
        return_30d = 0
        total_invested = 0
        winning = 0
        losing = 0
        daily_pnl = defaultdict(int)

        for s in settlements:
            revenue = s.get('revenue', 0)
            cost = s.get('yes_total_cost', 0) + s.get('no_total_cost', 0)
            pnl = revenue - cost
            total_return += pnl
            total_invested += cost

            if pnl > 0:
                winning += 1
            elif pnl < 0:
                losing += 1

            # Parse settled_time for time-window returns + chart
            ts = s.get('settled_time', '')
            if ts:
                ts_clean = ts.replace('Z', '+00:00')
                try:
                    dt = datetime.fromisoformat(ts_clean)
                    age = now - dt
                    if age <= timedelta(hours=24):
                        return_24h += pnl
                    if age <= timedelta(days=7):
                        return_7d += pnl
                    if age <= timedelta(days=30):
                        return_30d += pnl
                    daily_pnl[dt.strftime('%Y-%m-%d')] += pnl
                except (ValueError, TypeError):
                    pass

        total_trades = len(settlements)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        roi = (total_return / total_invested * 100) if total_invested > 0 else 0

        stats = {
            "balance_cents": balance_cents,
            "total_return_cents": total_return,
            "return_24h_cents": return_24h,
            "return_7d_cents": return_7d,
            "return_30d_cents": return_30d,
            "total_invested_cents": total_invested,
            "roi": roi,
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": win_rate,
        }

        # ── Cumulative returns over time ──────────────────────────────
        returns_over_time = []
        if daily_pnl:
            cumulative = 0
            for date in sorted(daily_pnl):
                cumulative += daily_pnl[date]
                returns_over_time.append({"date": date, "cumulative_return": cumulative})

        # ── Recent items for tables (most recent first) ───────────────
        recent_fills = sorted(fills, key=lambda f: f.get('created_time', ''), reverse=True)[:20]
        recent_settlements = sorted(settlements, key=lambda s: s.get('settled_time', ''), reverse=True)[:20]

        return {
            "success": True,
            "stats": stats,
            "recent_fills": recent_fills,
            "recent_settlements": recent_settlements,
            "returns_over_time": returns_over_time,
        }
    except Exception as e:
        logger.error("Market dashboard error: %s", e)
        return {"success": False, "error": str(e)}


def market_fills_getter(cursor=None, limit=100):
    """Account-level fills with pagination."""
    connector = make_market_connector()
    if not connector:
        return {"success": False, "error": "Kalshi credentials not configured"}
    try:
        result = connector.get_fills(limit=limit, cursor=cursor)
        return {
            "success": True,
            "fills": result.get("fills", []),
            "cursor": result.get("cursor"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def market_settlements_getter(cursor=None, limit=100):
    """Account-level settlements with pagination."""
    connector = make_market_connector()
    if not connector:
        return {"success": False, "error": "Kalshi credentials not configured"}
    try:
        result = connector.get_settlements(limit=limit, cursor=cursor)
        return {
            "success": True,
            "settlements": result.get("settlements", []),
            "cursor": result.get("cursor"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def market_bins_getter(args, league=None):
    """Market bin analysis — P&L bucketed by implied probability or time.

    Requires a league with ``market:`` config (series_ticker / spread_series_ticker)
    in its YAML.  Supports moneyline, spread, both, parlay, and all market types,
    custom bins, and time-based binning (day_of_week, month, hour).
    """
    from sportscore.services.market_bins import (
        compute_market_bins, compute_time_bins,
    )

    # ── League market config ──────────────────────────────────────────
    market_config = league and league.raw.get('market')
    if not market_config:
        return {"success": False, "error": "No market configuration for this league"}

    connector = make_market_connector()
    if not connector:
        return {"success": False, "error": "Kalshi credentials not configured (KALSHI_API_KEY, KALSHI_PRIVATE_KEY_DIR)"}

    # ── Parse query params ────────────────────────────────────────────
    market_type = args.get('market_type', 'moneyline')
    bin_mode = args.get('bin_mode', 'implied_prob')
    bin_type = args.get('bin_type', 'implied_prob')
    bins_param = args.get('bins')
    refresh = args.get('refresh') == 'true'

    # ── Ticker prefixes from league config ────────────────────────────
    ticker_prefixes = []
    league_series_prefixes = []
    series = market_config.get('series_ticker')
    spread_series = market_config.get('spread_series_ticker')
    if series:
        league_series_prefixes.append(series)
    if spread_series:
        league_series_prefixes.append(spread_series)

    include_parlays = market_type in ('parlay', 'all')

    if market_type in ('moneyline', 'both', 'all'):
        if series:
            ticker_prefixes.append(series)
    if market_type in ('spread', 'both', 'all'):
        if spread_series:
            ticker_prefixes.append(spread_series)

    if not ticker_prefixes and not include_parlays:
        return {"success": False, "error": "No series ticker configured for this market type"}

    # ── Parse custom bins ─────────────────────────────────────────────
    custom_bins = None
    if bins_param:
        try:
            custom_bins = [(b[0], b[1]) for b in json_mod.loads(bins_param)]
        except (json_mod.JSONDecodeError, TypeError, IndexError):
            return {"success": False, "error": "Invalid bins parameter — expected JSON array of [low, high] pairs"}

    # ── Cache key ─────────────────────────────────────────────────────
    league_key = league.league_id if league else 'default'
    bins_hash = hash(json_mod.dumps(custom_bins)) if custom_bins else 'default'
    cache_key = f"bins:{league_key}:{market_type}:{bin_mode}:{bin_type}:{bins_hash}"

    try:
        # On refresh, clear computed-bins cache (portfolio cache cleared by _fetch_portfolio)
        if refresh:
            _market_dashboard_cache.delete(cache_key)
        else:
            cached = _market_dashboard_cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch fills/settlements (layered cache: memory → API → DB)
        fills, settlements = _fetch_portfolio(connector, refresh=refresh)

        # ── Compute bins ──────────────────────────────────────────────
        if include_parlays and league_series_prefixes:
            def _is_league_parlay(item):
                t = item.get('ticker', '').upper()
                if 'MULTIGAME' not in t:
                    return False
                et = item.get('event_ticker', '').upper()
                return any(p.upper() in t or p.upper() in et for p in league_series_prefixes)

            parlay_fills = [f for f in fills if _is_league_parlay(f)]
            parlay_settlements = [s for s in settlements if _is_league_parlay(s)]

            if market_type == 'parlay':
                result = compute_market_bins(
                    parlay_fills, parlay_settlements, [],
                    bins=custom_bins, bin_mode=bin_mode,
                )
            else:
                # "all": combine regular prefix-filtered + parlay results
                regular = compute_market_bins(
                    fills, settlements, ticker_prefixes,
                    bins=custom_bins, bin_mode=bin_mode,
                )
                parlay = compute_market_bins(
                    parlay_fills, parlay_settlements, [],
                    bins=custom_bins, bin_mode=bin_mode,
                )
                # Merge bin-by-bin
                for i, rb in enumerate(regular['bins']):
                    pb = parlay['bins'][i]
                    rb['count'] += pb['count']
                    rb['won'] += pb['won']
                    rb['cost'] += pb['cost']
                    rb['revenue'] += pb['revenue']
                    rb['pnl'] += pb['pnl']
                    rb['win_pct'] = round((rb['won'] / rb['count'] * 100) if rb['count'] > 0 else 0, 1)
                    rb['roi'] = round((rb['pnl'] / rb['cost'] * 100) if rb['cost'] > 0 else 0, 1)
                # Merge totals
                for key in ('count', 'won', 'cost', 'revenue', 'pnl'):
                    regular['totals'][key] += parlay['totals'][key]
                t = regular['totals']
                t['win_pct'] = round((t['won'] / t['count'] * 100) if t['count'] > 0 else 0, 1)
                t['roi'] = round((t['pnl'] / t['cost'] * 100) if t['cost'] > 0 else 0, 1)
                regular['n_fills'] += parlay['n_fills']
                regular['n_unsettled'] += parlay['n_unsettled']
                regular['positions'] = regular.get('positions', []) + parlay.get('positions', [])
                result = regular
        else:
            result = compute_market_bins(
                fills, settlements, ticker_prefixes,
                bins=custom_bins, bin_mode=bin_mode,
            )

        # ── Time-based binning ────────────────────────────────────────
        all_positions = result.get('positions', [])
        if bin_type in ('day_of_week', 'month', 'hour'):
            result['bins'] = compute_time_bins(all_positions, bin_type)

        # Lightweight positions for cumulative P&L chart
        result['positions'] = [
            {'created_time': p['created_time'], 'pnl': p['pnl']}
            for p in all_positions if p.get('created_time')
        ]

        result['success'] = True
        _market_dashboard_cache.set(cache_key, result)
        return result

    except Exception as e:
        logger.error("Market bins error: %s", e)
        return {"success": False, "error": str(e)}


def make_market_helpers():
    """Return a ``MarketServices`` dataclass for ``build_services()``.

    All market helpers are sport-agnostic (Kalshi account-level operations).
    """
    from sportscore.web.services import MarketServices
    return MarketServices(
        dashboard_getter=market_dashboard_getter,
        fills_getter=market_fills_getter,
        settlements_getter=market_settlements_getter,
        bins_getter=market_bins_getter,
    )
