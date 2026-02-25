"""
Market calibration orchestration service.

Reads a master training CSV, computes per-season market Brier/log-loss
from historical vegas implied probabilities, and stores results in MongoDB.

Falls back to querying game documents directly when the CSV lacks odds
columns (or doesn't exist at all), using sport-agnostic YAML config
(``market_calibration.db_odds``) to locate odds fields and their format.

Sport-agnostic: operates on any league config with standard attributes
(master_training_csv, season_cutover_month, collections, league_id).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sportscore.league_config import BaseLeagueConfig


def _vectorized_season_labels(years, months, cutover_month: int) -> List[str]:
    """Vectorized season label derivation — avoids slow row-wise apply."""
    import numpy as np
    years = np.asarray(years, dtype=int)
    months = np.asarray(months, dtype=int)
    after_cutover = months >= cutover_month
    start_years = np.where(after_cutover, years, years - 1)
    end_years = start_years + 1
    return [f"{s}-{e}" for s, e in zip(start_years, end_years)]


def _dot_get(doc: dict, path: str):
    """Traverse nested dicts by dot-separated *path*. Returns None on miss."""
    cur = doc
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _odds_to_prob(val, fmt: str) -> Optional[float]:
    """Convert a single odds value to an implied probability.

    *fmt* is ``"decimal"`` (European, e.g. 2.15 → 1/2.15) or
    ``"american"`` (moneyline, e.g. -150 → 150/250).
    """
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None

    if fmt == "decimal":
        if val <= 0:
            return None
        return 1.0 / val
    elif fmt == "american":
        if val < 0:
            return (-val) / (-val + 100)
        elif val > 0:
            return 100.0 / (val + 100)
        else:
            return None
    return None


def _load_odds_from_db(db, league: "BaseLeagueConfig", outcome_type: str = "binary", verbose: bool = False):
    """Load odds directly from game documents when CSV lacks vegas columns.

    Returns a DataFrame with columns matching the downstream logic
    (home_col, away_col, outcome_col, Year, Month), or None if no data.

    For ``outcome_type="3way"``, also loads draw odds and encodes outcomes as
    0=away, 1=draw, 2=home (instead of binary 0/1).
    """
    import pandas as pd

    db_odds_cfg = league.raw.get("market_calibration", {}).get("db_odds")
    if not db_odds_cfg:
        return None

    home_path = db_odds_cfg["home"]
    away_path = db_odds_cfg["away"]
    draw_path = db_odds_cfg.get("draw")
    odds_format = db_odds_cfg.get("format", "decimal")

    home_col = "vegas_implied_prob|none|raw|home"
    away_col = "vegas_implied_prob|none|raw|away"
    draw_col = "vegas_implied_prob|none|raw|draw"

    is_3way = outcome_type == "3way" and draw_path is not None

    games_col = db[league.collections["games"]]

    projection = {
        "year": 1, "month": 1, "homeWon": 1,
        home_path: 1, away_path: 1,
    }
    if is_3way:
        projection[draw_path] = 1

    cursor = games_col.find(
        {
            "league": league.league_id,
            "homeWon": {"$exists": True},
            home_path: {"$exists": True, "$ne": None},
        },
        projection,
    )

    rows = []
    for doc in cursor:
        home_odds = _dot_get(doc, home_path)
        away_odds = _dot_get(doc, away_path)
        home_won = doc.get("homeWon")

        if home_odds is None:
            continue

        # For binary, skip draws (homeWon is None); for 3-way, keep them
        if not is_3way and home_won is None:
            continue

        home_prob = _odds_to_prob(home_odds, odds_format)
        away_prob = _odds_to_prob(away_odds, odds_format) if away_odds is not None else None

        if home_prob is None:
            continue

        row = {
            home_col: home_prob,
            away_col: away_prob,
            "Year": doc.get("year"),
            "Month": doc.get("month"),
        }

        if is_3way:
            draw_odds = _dot_get(doc, draw_path)
            draw_prob = _odds_to_prob(draw_odds, odds_format) if draw_odds is not None else None
            if draw_prob is None:
                continue
            row[draw_col] = draw_prob
            # 3-way outcome: 2=home, 1=draw, 0=away
            if home_won is True:
                row["Outcome"] = 2
            elif home_won is False:
                row["Outcome"] = 0
            else:
                row["Outcome"] = 1  # draw (homeWon is None)
        else:
            row["HomeWon"] = 1 if home_won is True else 0

        rows.append(row)

    if not rows:
        return None

    if verbose:
        print(f"  Loaded {len(rows)} games with odds from DB")

    return pd.DataFrame(rows)


def compute_and_store_market_calibration(
    db,
    league: "BaseLeagueConfig",
    *,
    rolling_seasons: int = 3,
    min_coverage: float = 0.50,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Read master training CSV, compute per-season market Brier/log-loss,
    store in market_calibration collection.

    Falls back to querying game documents directly when the CSV lacks
    odds columns or doesn't exist, using ``market_calibration.db_odds``
    config from the league YAML.

    Returns dict with keys:
        season_results, rolling_brier, rolling_log_loss, seasons_computed,
        collection_name, n_written
    """
    import pandas as pd
    from sportscore.services.market_calibration import compute_market_metrics, compute_market_metrics_3way

    csv_path = league.master_training_csv
    cutover_month = league.season_cutover_month

    # Determine outcome type from sport config
    outcome_type = "binary"
    if league.sport_config is not None and hasattr(league.sport_config, 'outcome_type'):
        outcome_type = league.sport_config.outcome_type

    # Column names used throughout
    home_col = "vegas_implied_prob|none|raw|home"
    away_col = "vegas_implied_prob|none|raw|away"
    draw_col = "vegas_implied_prob|none|raw|draw"
    outcome_col = "HomeWon"  # overridden to "Outcome" for 3-way DB path below

    empty_result = {
        "season_results": [], "seasons_computed": 0,
        "rolling_brier": None, "rolling_log_loss": None,
        "collection_name": None, "n_written": 0,
    }

    # --- 1. Try CSV ---
    df = None
    csv_has_odds = False

    if verbose:
        print(f"  Reading {csv_path}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        csv_has_odds = home_col in df.columns
    except FileNotFoundError:
        if verbose:
            print(f"  CSV not found.")

    # --- 2. CSV with odds — existing path (unchanged) ---
    if df is not None and csv_has_odds:
        missing = [c for c in [home_col, outcome_col, "Year", "Month"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        has_away = away_col in df.columns

        mask = df[home_col].notna() & (df[home_col] > 0)
        if has_away:
            mask = mask & df[away_col].notna() & (df[away_col] > 0)
        df_odds = df[mask].copy()

        total_games = len(df)
        games_with_odds = len(df_odds)

        if verbose:
            print(f"  Games with odds: {games_with_odds:,} / {total_games:,} "
                  f"({100 * games_with_odds / total_games:.1f}%)")

        if games_with_odds == 0:
            if verbose:
                print("  No games with odds data found.")
            return empty_result

        # Derive season labels — use full df for total counts
        df_odds["_season"] = _vectorized_season_labels(df_odds["Year"], df_odds["Month"], cutover_month)
        df["_season"] = _vectorized_season_labels(df["Year"], df["Month"], cutover_month)
        total_per_season = df.groupby("_season").size()

    # --- 3. DB fallback ---
    else:
        if verbose:
            print("  CSV missing odds columns, trying DB fallback...")
        df_odds = _load_odds_from_db(db, league, outcome_type=outcome_type, verbose=verbose)
        if df_odds is None or len(df_odds) == 0:
            if verbose:
                print("  No odds found in CSV or DB.")
            return empty_result

        has_away = away_col in df_odds.columns and df_odds[away_col].notna().any()

        df_odds["_season"] = _vectorized_season_labels(df_odds["Year"], df_odds["Month"], cutover_month)
        # For DB path, total_per_season == odds_per_season (all rows have odds)
        total_per_season = df_odds.groupby("_season").size()

    # For 3-way DB path, outcome column is "Outcome" (0/1/2) instead of "HomeWon"
    if "Outcome" in df_odds.columns:
        outcome_col = "Outcome"

    # --- Common downstream: per-season metrics ---
    has_draw = draw_col in df_odds.columns and df_odds[draw_col].notna().any()
    season_results: List[Dict[str, Any]] = []
    for season_str, group in sorted(df_odds.groupby("_season")):
        total_in_season = total_per_season.get(season_str, len(group))
        coverage = len(group) / total_in_season if total_in_season > 0 else 0.0

        if outcome_type == "3way" and has_draw:
            metrics = compute_market_metrics_3way(
                group[outcome_col].values,
                group[home_col].values,
                group[draw_col].values,
                group[away_col].values,
            )
        else:
            y_true = group[outcome_col].values
            p_home = group[home_col].values
            p_away = group[away_col].values if has_away else None
            metrics = compute_market_metrics(y_true, p_home, p_away)
        metrics["season"] = season_str
        metrics["coverage_pct"] = round(100 * coverage, 1)
        metrics["total_games"] = int(total_in_season)
        season_results.append(metrics)

    # Rolling aggregate
    eligible = [s for s in season_results if s["coverage_pct"] >= min_coverage * 100]
    eligible.sort(key=lambda s: s["season"], reverse=True)
    rolling_n = min(rolling_seasons, len(eligible))

    rolling_brier = None
    rolling_log_loss = None
    rolling_overround = None
    rolling_coverage = None
    rolling_label = None

    if rolling_n > 0:
        rolling = eligible[:rolling_n]
        total_n = sum(s["n_games"] for s in rolling)
        rolling_brier = sum(s["market_brier"] * s["n_games"] for s in rolling) / total_n
        rolling_log_loss = sum(s["market_log_loss"] * s["n_games"] for s in rolling) / total_n
        rolling_overround = sum(s["avg_overround"] * s["n_games"] for s in rolling) / total_n
        rolling_coverage = sum(s["coverage_pct"] * s["n_games"] for s in rolling) / total_n
        rolling_label = f"rolling_{rolling_n}yr"

        if verbose:
            print(f"  Rolling ({rolling_n} seasons): Brier={rolling_brier:.4f}, "
                  f"LogLoss={rolling_log_loss:.4f}, Games={total_n:,}")

    # Write to MongoDB
    coll_name = league.collections.get("market_calibration")
    if not coll_name:
        coll_name = f"{league.league_id}_market_calibration"
        if verbose:
            print(f"  Note: No 'market_calibration' collection in YAML, using {coll_name}")

    n_written = 0
    if not dry_run:
        coll = db[coll_name]
        now = datetime.now(timezone.utc)

        for s in season_results:
            doc = {
                "league": league.league_id,
                "season": s["season"],
                "market_brier": round(s["market_brier"], 6),
                "market_log_loss": round(s["market_log_loss"], 6),
                "n_games": s["n_games"],
                "coverage_pct": s["coverage_pct"],
                "avg_overround": round(s["avg_overround"], 4),
                "computed_at": now,
            }
            coll.update_one(
                {"league": league.league_id, "season": s["season"]},
                {"$set": doc},
                upsert=True,
            )

        if rolling_label and rolling_n > 0:
            rolling_doc = {
                "league": league.league_id,
                "season": rolling_label,
                "market_brier": round(rolling_brier, 6),
                "market_log_loss": round(rolling_log_loss, 6),
                "n_games": total_n,
                "coverage_pct": round(rolling_coverage, 1),
                "avg_overround": round(rolling_overround, 4),
                "seasons_included": [s["season"] for s in eligible[:rolling_n]],
                "computed_at": now,
            }
            coll.update_one(
                {"league": league.league_id, "season": rolling_label},
                {"$set": rolling_doc},
                upsert=True,
            )

        n_written = len(season_results) + (1 if rolling_label else 0)
        if verbose:
            print(f"  Wrote {n_written} docs to {coll_name}")
    elif verbose:
        print(f"  [DRY RUN] Would write {len(season_results) + (1 if rolling_label else 0)} docs to {coll_name}")

    return {
        "season_results": season_results,
        "seasons_computed": len(season_results),
        "rolling_brier": rolling_brier,
        "rolling_log_loss": rolling_log_loss,
        "collection_name": coll_name,
        "n_written": n_written,
    }
