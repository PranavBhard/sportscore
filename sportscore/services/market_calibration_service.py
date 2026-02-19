"""
Market calibration orchestration service.

Reads a master training CSV, computes per-season market Brier/log-loss
from historical vegas implied probabilities, and stores results in MongoDB.

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

    Returns dict with keys:
        season_results, rolling_brier, rolling_log_loss, seasons_computed,
        collection_name, n_written
    """
    import pandas as pd
    from sportscore.services.market_calibration import compute_market_metrics

    csv_path = league.master_training_csv
    cutover_month = league.season_cutover_month

    if verbose:
        print(f"  Reading {csv_path}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Master training CSV not found: {csv_path}")

    # Required columns
    home_col = "vegas_implied_prob|none|raw|home"
    away_col = "vegas_implied_prob|none|raw|away"
    outcome_col = "HomeWon"

    missing = [c for c in [home_col, outcome_col, "Year", "Month"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    has_away = away_col in df.columns

    # Filter to rows with valid odds
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
        return {"season_results": [], "seasons_computed": 0,
                "rolling_brier": None, "rolling_log_loss": None,
                "collection_name": None, "n_written": 0}

    # Derive season labels
    df_odds["_season"] = _vectorized_season_labels(df_odds["Year"], df_odds["Month"], cutover_month)
    df["_season"] = _vectorized_season_labels(df["Year"], df["Month"], cutover_month)
    total_per_season = df.groupby("_season").size()

    # Compute per-season metrics
    season_results: List[Dict[str, Any]] = []
    for season_str, group in sorted(df_odds.groupby("_season")):
        total_in_season = total_per_season.get(season_str, len(group))
        coverage = len(group) / total_in_season if total_in_season > 0 else 0.0

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
            print(f"  Warning: No 'market_calibration' collection in YAML, using {coll_name}")

    n_written = 0
    if not dry_run:
        coll = db[coll_name]
        now = datetime.now(timezone.utc)

        for s in season_results:
            doc = {
                "league_id": league.league_id,
                "season": s["season"],
                "market_brier": round(s["market_brier"], 6),
                "market_log_loss": round(s["market_log_loss"], 6),
                "n_games": s["n_games"],
                "coverage_pct": s["coverage_pct"],
                "avg_overround": round(s["avg_overround"], 4),
                "computed_at": now,
            }
            coll.update_one(
                {"league_id": league.league_id, "season": s["season"]},
                {"$set": doc},
                upsert=True,
            )

        if rolling_label and rolling_n > 0:
            rolling_doc = {
                "league_id": league.league_id,
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
                {"league_id": league.league_id, "season": rolling_label},
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
