"""Sport-agnostic game-level Kalshi market functions.

Provides ticker building/parsing, market data fetching, and portfolio-to-game
matching for any sport. Sport apps configure via their league YAML::

    market:
      provider: kalshi
      series_ticker: KXEPLGAME
      spread_series_ticker: ""          # optional
      abbrev_length: 3                  # optional, default 3
      team_abbrev_map:                  # Kalshi abbrev -> internal DB abbrev
        GSW: GS

All functions accept a ``league`` object (anything with a ``.raw`` dict and
``.league_id`` attribute) — the same object available on ``g.league`` in the
web layer.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from sportscore.market.kalshi import KalshiPublicClient, SimpleCache

logger = logging.getLogger(__name__)

# Module-level cache for market data (60s default TTL)
_market_cache = SimpleCache(default_ttl=60)


# ---------------------------------------------------------------------------
# League config helpers
# ---------------------------------------------------------------------------

def _market_config(league) -> dict:
    """Extract ``market:`` block from league config, or empty dict."""
    if league is None:
        return {}
    raw = getattr(league, "raw", None) or {}
    return raw.get("market", {})


def get_team_abbrev_map(league) -> Dict[str, str]:
    """Kalshi abbreviation -> internal DB abbreviation.

    Only entries that differ are listed; unlisted teams use the same abbrev.
    """
    return _market_config(league).get("team_abbrev_map", {})


def get_reverse_abbrev_map(league) -> Dict[str, str]:
    """Internal DB abbreviation -> Kalshi abbreviation."""
    return {v: k for k, v in get_team_abbrev_map(league).items()}


def internal_to_kalshi_abbrev(internal_abbrev: str, league) -> str:
    """Convert internal DB abbreviation to Kalshi abbreviation."""
    return get_reverse_abbrev_map(league).get(internal_abbrev, internal_abbrev)


def kalshi_to_internal_abbrev(kalshi_abbrev: str, league) -> str:
    """Convert Kalshi abbreviation to internal DB abbreviation."""
    return get_team_abbrev_map(league).get(kalshi_abbrev, kalshi_abbrev)


# ---------------------------------------------------------------------------
# Ticker building / parsing
# ---------------------------------------------------------------------------

def build_event_ticker(
    game_date: date,
    away_team: str,
    home_team: str,
    league,
) -> str:
    """Build Kalshi event ticker from game info.

    Format: ``{series_ticker}-{YY}{MON}{DD}{AWAY}{HOME}``
    Example: ``KXEPLGAME-26FEB23ARSLIV``
    """
    kalshi_away = internal_to_kalshi_abbrev(away_team, league)
    kalshi_home = internal_to_kalshi_abbrev(home_team, league)

    year_2d = game_date.strftime("%y")
    month_abbrev = game_date.strftime("%b").upper()
    day_2d = game_date.strftime("%d")

    mc = _market_config(league)
    series_ticker = mc.get("series_ticker", "KXGAME")

    return f"{series_ticker}-{year_2d}{month_abbrev}{day_2d}{kalshi_away}{kalshi_home}"


def parse_event_ticker(
    event_ticker: str,
    league,
) -> Optional[Dict[str, Any]]:
    """Parse Kalshi event ticker to extract game info.

    Returns dict with *date*, *away_team*, *home_team* (Kalshi abbrevs),
    *away_team_internal*, *home_team_internal*, *series_ticker*
    — or ``None`` on failure.
    """
    try:
        parts = event_ticker.split("-")
        if len(parts) < 2:
            return None

        game_part = parts[-1]  # e.g. "26FEB23ARSLIV"
        if len(game_part) < 9:
            return None

        year_2d = game_part[:2]
        month_abbrev = game_part[2:5]
        day_2d = game_part[5:7]
        teams_part = game_part[7:]

        game_date = datetime.strptime(f"{month_abbrev} {day_2d} 20{year_2d}", "%b %d %Y").date()

        mc = _market_config(league)
        abbrev_len = mc.get("abbrev_length", 3)

        away_team = None
        home_team = None

        if abbrev_len and len(teams_part) >= abbrev_len * 2:
            away_team = teams_part[:abbrev_len]
            home_team = teams_part[abbrev_len : abbrev_len * 2]
        else:
            # Variable-length: try all splits, score by known abbreviations
            all_kalshi = set(get_team_abbrev_map(league).keys())
            all_internal = set(get_team_abbrev_map(league).values())
            known = all_kalshi | all_internal

            valid_splits = []
            for alen in range(2, min(5, len(teams_part) - 1)):
                ca = teams_part[:alen]
                ch = teams_part[alen:]
                if not (2 <= len(ch) <= 4):
                    continue
                a_known = ca in known
                h_known = ch in known
                if a_known and h_known:
                    valid_splits.append((2, ca, ch))
                elif a_known or h_known:
                    valid_splits.append((1, ca, ch))
                elif ca.isupper() and ch.isupper():
                    valid_splits.append((0, ca, ch))

            if valid_splits:
                valid_splits.sort(key=lambda x: (x[0], -abs(len(x[1]) - len(x[2]))), reverse=True)
                _, away_team, home_team = valid_splits[0]
            else:
                mid = len(teams_part) // 2
                away_team = teams_part[:mid]
                home_team = teams_part[mid:]

        if not away_team or not home_team:
            return None

        return {
            "date": game_date,
            "away_team": away_team,
            "home_team": home_team,
            "away_team_internal": kalshi_to_internal_abbrev(away_team, league),
            "home_team_internal": kalshi_to_internal_abbrev(home_team, league),
            "series_ticker": "-".join(parts[:-1]),
        }
    except Exception as e:
        logger.debug("Could not parse event ticker %s: %s", event_ticker, e)
        return None


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """Normalized market data for a single game."""
    event_ticker: str
    home_team: str
    away_team: str
    home_yes_price: float
    home_yes_bid: float
    home_yes_ask: float
    away_yes_price: float
    away_yes_bid: float
    away_yes_ask: float
    home_volume: int
    away_volume: int
    total_liquidity: float
    status: str
    last_updated: datetime
    # Draw fields (3-way markets, e.g. soccer) — defaults for backward compat
    draw_yes_price: float = 0.0
    draw_yes_bid: float = 0.0
    draw_yes_ask: float = 0.0
    draw_volume: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_ticker": self.event_ticker,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_yes_price": self.home_yes_price,
            "home_yes_bid": self.home_yes_bid,
            "home_yes_ask": self.home_yes_ask,
            "away_yes_price": self.away_yes_price,
            "away_yes_bid": self.away_yes_bid,
            "away_yes_ask": self.away_yes_ask,
            "home_volume": self.home_volume,
            "away_volume": self.away_volume,
            "draw_yes_price": self.draw_yes_price,
            "draw_yes_bid": self.draw_yes_bid,
            "draw_yes_ask": self.draw_yes_ask,
            "draw_volume": self.draw_volume,
            "total_liquidity": self.total_liquidity,
            "status": self.status,
            "last_updated": self.last_updated.isoformat(),
        }


def _extract_market_prices(market: Optional[Dict]) -> Tuple[float, float, float, int, str]:
    """Return ``(mid_price, bid, ask, volume, status)``."""
    if not market:
        return (0.0, 0.0, 0.0, 0, "unknown")
    yes_bid = market.get("yes_bid", 0) / 100.0
    yes_ask = market.get("yes_ask", 0) / 100.0
    if yes_bid > 0 and yes_ask > 0:
        mid = (yes_bid + yes_ask) / 2.0
    else:
        mid = market.get("last_price", 0) / 100.0
    return (mid, yes_bid, yes_ask, market.get("volume", 0), market.get("status", "unknown"))


def get_game_market_data(
    game_date: date,
    away_team: str,
    home_team: str,
    league,
    use_cache: bool = True,
    cache_ttl: int = 60,
) -> Optional[MarketData]:
    """Fetch Kalshi market data for a single game."""
    import requests

    event_ticker = build_event_ticker(game_date, away_team, home_team, league)
    cache_key = f"market:{event_ticker}"

    if use_cache:
        cached = _market_cache.get(cache_key)
        if cached is not None:
            return cached

    client = KalshiPublicClient()
    try:
        data = client.get_event(event_ticker)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.debug("No market found for %s", event_ticker)
            return None
        logger.error("API error fetching %s: %s", event_ticker, e)
        return None
    except Exception as e:
        logger.error("Error fetching market data for %s: %s", event_ticker, e)
        return None

    markets = data.get("markets", [])
    if not markets:
        return None

    kalshi_home = internal_to_kalshi_abbrev(home_team, league)
    kalshi_away = internal_to_kalshi_abbrev(away_team, league)

    home_market = away_market = draw_market = None
    for m in markets:
        ticker = m.get("ticker", "")
        if ticker.endswith(f"-{kalshi_home}"):
            home_market = m
        elif ticker.endswith(f"-{kalshi_away}"):
            away_market = m
        else:
            # Third market in a 3-way event = draw
            draw_market = m

    if not home_market and not away_market:
        logger.warning("Could not identify team markets for %s", event_ticker)
        return None

    hp, hb, ha, hv, hs = _extract_market_prices(home_market)
    ap, ab, aa, av, as_ = _extract_market_prices(away_market)
    dp, db_, da, dv, ds = _extract_market_prices(draw_market)

    home_liq = float(home_market.get("liquidity_dollars", "0")) if home_market else 0
    away_liq = float(away_market.get("liquidity_dollars", "0")) if away_market else 0
    draw_liq = float(draw_market.get("liquidity_dollars", "0")) if draw_market else 0

    result = MarketData(
        event_ticker=event_ticker,
        home_team=home_team,
        away_team=away_team,
        home_yes_price=hp,
        home_yes_bid=hb,
        home_yes_ask=ha,
        away_yes_price=ap,
        away_yes_bid=ab,
        away_yes_ask=aa,
        home_volume=hv,
        away_volume=av,
        total_liquidity=home_liq + away_liq + draw_liq,
        status=hs if home_market else as_,
        last_updated=datetime.utcnow(),
        draw_yes_price=dp,
        draw_yes_bid=db_,
        draw_yes_ask=da,
        draw_volume=dv,
    )

    if use_cache:
        _market_cache.set(cache_key, result, cache_ttl)

    return result


def get_kalshi_events_for_date(
    game_date: date,
    league,
    use_cache: bool = True,
    cache_ttl: int = 300,
) -> list:
    """Fetch all Kalshi events for a specific date and league."""
    lid = getattr(league, "league_id", "unknown")
    cache_key = f"events:{lid}:{game_date.isoformat()}"

    if use_cache:
        cached = _market_cache.get(cache_key)
        if cached is not None:
            return cached

    mc = _market_config(league)
    series_ticker = mc.get("series_ticker")
    if not series_ticker:
        return []

    client = KalshiPublicClient()
    matched = []
    try:
        data = client.get_events(series_ticker=series_ticker, limit=200)
        date_prefix = f"{game_date.strftime('%y')}{game_date.strftime('%b').upper()}{game_date.strftime('%d')}"
        for ev in data.get("events", []):
            if f"-{date_prefix}" in ev.get("event_ticker", ""):
                matched.append(ev)
    except Exception as e:
        logger.error("Error fetching Kalshi events for %s: %s", game_date, e)

    if use_cache and matched:
        _market_cache.set(cache_key, matched, cache_ttl)
    return matched


def clear_market_cache() -> None:
    """Clear the module-level market data cache."""
    _market_cache.clear()


# ---------------------------------------------------------------------------
# Portfolio <-> game matching
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMatch:
    """Result of matching portfolio items to games."""
    game_data: Dict[str, Dict[str, Any]]
    unmatched_fills: list
    debug_info: Optional[Dict[str, Any]] = None


def match_portfolio_to_games(
    games: list,
    positions: list,
    fills: list,
    orders: list,
    settlements: list,
    game_date: date,
    league,
) -> PortfolioMatch:
    """Match Kalshi portfolio items to games for a given date.

    Handles winner/moneyline and (optionally) spread markets.
    """
    mc = _market_config(league)
    series_ticker = mc.get("series_ticker", "KXGAME")
    spread_series = mc.get("spread_series_ticker", "")

    # Build event ticker → game_id lookups
    game_tickers: Dict[str, str] = {}
    spread_tickers: Dict[str, str] = {}

    for game in games:
        gid = game.get("game_id")
        ht = game.get("homeTeam", {}).get("name")
        at = game.get("awayTeam", {}).get("name")
        if not (gid and ht and at):
            continue
        evt = build_event_ticker(game_date, at, ht, league)
        game_tickers[evt] = gid
        if spread_series:
            spread_tickers[evt.replace(series_ticker, spread_series, 1)] = gid

    # Settlement lookup for P&L
    settlement_lookup = {}
    for s in settlements:
        ticker = s.get("ticker", "")
        rev = s.get("revenue", 0)
        settlement_lookup[ticker] = {
            "settled": True,
            "revenue": rev,
            "pnl": rev - s.get("cost_basis", 0) if "cost_basis" in s else rev,
        }

    # Init per-game buckets
    game_data: Dict[str, Dict[str, Any]] = {
        g.get("game_id"): {"positions": [], "fills": [], "orders": [], "parlay_fills": []}
        for g in games if g.get("game_id")
    }

    # Team → game_id (for parlay detection)
    reverse_map = get_reverse_abbrev_map(league)
    team_to_game: Dict[str, str] = {}
    for game in games:
        gid = game.get("game_id")
        for side in ("homeTeam", "awayTeam"):
            name = game.get(side, {}).get("name", "").upper()
            if name and gid:
                team_to_game[name] = gid
                kalshi_name = reverse_map.get(name, name)
                if kalshi_name != name:
                    team_to_game[kalshi_name] = gid

    # ── Match positions ──
    for pos in positions:
        ticker = pos.get("ticker", "")
        evt = pos.get("event_ticker", "")
        game_id = game_tickers.get(evt) or spread_tickers.get(evt)
        if not game_id or game_id not in game_data:
            continue

        is_spread = evt in spread_tickers
        team, spread_value = _parse_ticker_team(ticker, is_spread)
        count = pos.get("total_traded", 0)
        avg_price = pos.get("average_price_cents", 0) / 100.0
        market_value = pos.get("market_value_cents", 0) / 100.0
        cost_basis = count * avg_price

        sett = settlement_lookup.get(ticker, {})
        if sett.get("settled"):
            pnl = sett.get("pnl", 0) / 100.0
            status = "won" if pnl > 0 else "lost"
        else:
            status = "live"
            pnl = market_value - cost_basis

        label = f"{team} -{spread_value}" if is_spread and spread_value else team
        game_data[game_id]["positions"].append({
            "ticker": ticker, "team": team, "market_label": label,
            "side": pos.get("side", ""), "action": "buy",
            "count": count, "avg_price": avg_price,
            "current_value": market_value, "cost": cost_basis,
            "pnl": pnl, "status": status,
            "is_spread": is_spread, "spread_value": spread_value,
        })

    # ── Match fills ──
    unmatched_fills: list = []
    for fill in fills:
        ticker = fill.get("ticker", "")
        fill_evt = fill.get("event_ticker", "")
        game_id = None
        team = ""
        spread_value = None

        # Strategy 1: event_ticker field
        if fill_evt:
            game_id = game_tickers.get(fill_evt) or spread_tickers.get(fill_evt)
            if game_id:
                team = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""

        # Strategy 2: parse ticker as {event_ticker}-{team}
        if not game_id and "-" in ticker:
            evt_part, team_part = ticker.rsplit("-", 1)
            if evt_part in game_tickers:
                game_id = game_tickers[evt_part]
                team = team_part

        # Strategy 3: spread ticker parsing
        if not game_id and spread_series and ticker.startswith(spread_series):
            parts = ticker.split("-")
            if len(parts) >= 3:
                team, spread_value = _parse_spread_suffix(parts[-1])
                spread_evt = "-".join(parts[:-1])
                game_id = spread_tickers.get(spread_evt)

        if game_id and game_id in game_data:
            side = fill.get("side", "")
            price_cents = fill.get("yes_price", 0) if side == "yes" else fill.get("no_price", 0)
            count = fill.get("count", 0)
            sett = settlement_lookup.get(ticker, {})
            status = ("won" if sett.get("pnl", 0) > 0 else "lost") if sett.get("settled") else "live"
            label = f"{team} -{spread_value}" if spread_value else team

            game_data[game_id]["fills"].append({
                "ticker": ticker, "team": team, "market_label": label,
                "side": side, "action": fill.get("action", ""),
                "count": count, "price": price_cents / 100.0,
                "cost": count * (price_cents / 100.0),
                "time": fill.get("created_time", ""), "status": status,
                "is_spread": spread_value is not None, "spread_value": spread_value,
            })
        else:
            unmatched_fills.append(fill)

    # ── Parlay detection on unmatched fills ──
    abbrev_map = get_team_abbrev_map(league)
    for fill in list(unmatched_fills):
        ticker = fill.get("ticker", "")
        involved_ids: set = set()
        involved_teams: list = []
        for part in ticker.split("-"):
            up = part.upper()
            internal = abbrev_map.get(up, up)
            gid = team_to_game.get(internal) or team_to_game.get(up)
            if gid:
                involved_ids.add(gid)
                involved_teams.append(internal)

        if len(involved_ids) >= 2:
            side = fill.get("side", "")
            price_cents = fill.get("yes_price", 0) if side == "yes" else fill.get("no_price", 0)
            count = fill.get("count", 0)
            cost = count * (price_cents / 100.0)
            sett = settlement_lookup.get(ticker, {})
            if sett.get("settled"):
                pnl = sett.get("pnl", 0) / 100.0
                status = "won" if pnl > 0 else "lost"
                if status == "lost" and pnl == 0:
                    pnl = -cost
            else:
                status = "live"
                pnl = 0

            n = len(involved_ids)
            for gid in involved_ids:
                if gid in game_data:
                    game_data[gid]["parlay_fills"].append({
                        "ticker": ticker, "teams": involved_teams,
                        "all_game_ids": list(involved_ids),
                        "market_label": f"{n}-leg parlay",
                        "side": side, "action": fill.get("action", ""),
                        "count": count, "price": price_cents / 100.0,
                        "total_cost": cost, "fractional_cost": cost / n,
                        "total_pnl": pnl, "fractional_pnl": pnl / n,
                        "num_legs": n, "time": fill.get("created_time", ""),
                        "status": status, "is_parlay": True,
                    })

    # ── Match orders ──
    for order in orders:
        ticker = order.get("ticker", "")
        evt = order.get("event_ticker", "")
        game_id = game_tickers.get(evt) or spread_tickers.get(evt)
        if not game_id or game_id not in game_data:
            continue

        is_spread = evt in spread_tickers
        team, spread_value = _parse_ticker_team(ticker, is_spread)
        label = f"{team} -{spread_value}" if is_spread and spread_value else team
        count = order.get("remaining_count", 0)
        price = (order.get("yes_price", 0) / 100.0 if order.get("yes_price")
                 else order.get("no_price", 0) / 100.0)

        game_data[game_id]["orders"].append({
            "ticker": ticker, "team": team, "market_label": label,
            "side": order.get("side", ""), "action": order.get("action", ""),
            "count": count, "price": price, "cost": count * price,
            "status": "pending",
            "is_spread": is_spread, "spread_value": spread_value,
        })

    return PortfolioMatch(
        game_data=game_data,
        unmatched_fills=unmatched_fills,
        debug_info={
            "game_tickers": list(game_tickers.keys())[:10],
            "spread_tickers": list(spread_tickers.keys())[:10],
            "series_ticker": series_ticker,
            "spread_series_ticker": spread_series,
        },
    )


# ---------------------------------------------------------------------------
# Prices getter factory (for MarketServices.prices_getter)
# ---------------------------------------------------------------------------

def make_prices_getter():
    """Return a ``prices_getter(date_str, db, league)`` for MarketServices."""

    def prices_getter(date_str, db, league):
        mc = _market_config(league)
        if not mc.get("series_ticker"):
            return {"success": True, "markets": {}, "message": "No market config for this league"}

        try:
            game_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return {"success": False, "error": "Invalid date format"}

        games_coll = league.collections.get("games", "games")
        query = {"date": date_str}
        if hasattr(league, "league_id"):
            query["league"] = league.league_id
        game_list = list(db[games_coll].find(
            query,
            {"game_id": 1, "homeTeam.name": 1, "awayTeam.name": 1},
        ))
        if not game_list:
            return {"success": True, "markets": {}, "message": "No games for this date"}

        markets = {}
        for game in game_list:
            gid = game.get("game_id")
            ht = game.get("homeTeam", {}).get("name", "")
            at = game.get("awayTeam", {}).get("name", "")
            if not (ht and at):
                continue
            try:
                md = get_game_market_data(game_date, at, ht, league, use_cache=True, cache_ttl=30)
                if md:
                    markets[gid] = md.to_dict()
            except Exception:
                continue
        return {"success": True, "markets": markets}

    return prices_getter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ticker_team(ticker: str, is_spread: bool) -> Tuple[str, Optional[str]]:
    """Extract team abbreviation and optional spread value from last ticker segment."""
    parts = ticker.split("-")
    last = parts[-1] if parts else ""

    if is_spread:
        m = re.match(r"^([A-Z]+)([\d.]+)$", last)
        if m:
            return m.group(1), m.group(2)
        if len(parts) >= 4:
            return parts[-2], parts[-1]

    return last, None


def _parse_spread_suffix(suffix: str) -> Tuple[str, Optional[str]]:
    """Parse ``OKC7`` into ``('OKC', '7')``."""
    m = re.match(r"^([A-Z]+)([\d.]+)$", suffix)
    if m:
        return m.group(1), m.group(2)
    return suffix, None
