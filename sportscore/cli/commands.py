"""Generic sportscore CLI commands.

These commands work with any sport plugin that provides league loading
and DB access. Adding a command here makes it available to all sports
automatically — no per-sport-app changes needed.
"""

import os
import sys

from sportscore.cli.discovery import SportCommand


class CacheLeagueStatsCommand(SportCommand):
    name = "cache_league_stats"
    help = "Compute and cache league stats (pace, PER constants) in MongoDB"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--season", "-s", type=str, default=None,
                            help="Specific season to cache (e.g., 2025-2026)")
        parser.add_argument("--list", "-l", action="store_true", dest="list_seasons",
                            help="List all cached seasons")
        parser.add_argument("--force", "-f", action="store_true",
                            help="Force recalculation even if already cached")

    def run(self, args, *, league_loader=None, db_factory=None, **kwargs) -> int:
        # Import here so sportscore itself doesn't depend on bball at import time;
        # the sport app supplies the concrete functions via its league/db layer.
        league = league_loader(args.league)
        db = db_factory()

        print(f"League: {league.display_name} ({league.league_id})")

        # Use the sport app's league_cache module.  The functions are
        # generic (any sport that stores game box-scores can compute
        # possessions/pace), so we import from the basketball package
        # which currently hosts the implementation.  If a second sport
        # needs different formulas it can override via its own command.
        try:
            from bball.stats.league_cache import (
                cache_season,
                get_all_seasons,
                list_cached_seasons,
            )
        except ImportError as exc:
            print(f"Error: league_cache module not available: {exc}",
                  file=sys.stderr)
            return 1

        if args.list_seasons:
            list_cached_seasons(db, league=league)
            return 0

        if args.season:
            print(f"\nCaching season {args.season}...")
            success = cache_season(db, args.season, force=args.force, league=league)
            if success:
                print(f"\nSeason {args.season} cached successfully.")
            else:
                print(f"\nFailed to cache season {args.season}.", file=sys.stderr)
                return 1
        else:
            print("\nDiscovering seasons...")
            seasons = get_all_seasons(db, league=league)
            print(f"Found {len(seasons)} seasons: {', '.join(seasons)}\n")

            cached_count = 0
            for season in seasons:
                if cache_season(db, season, force=args.force, league=league):
                    cached_count += 1

            print(f"\nCached {cached_count}/{len(seasons)} seasons.")

        print()
        list_cached_seasons(db, league=league)
        return 0


class ComputeMarketCalibrationCommand(SportCommand):
    name = "compute_market_calibration"
    help = "Compute market calibration (Brier/log-loss) from historical odds"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--rolling-seasons", type=int, default=3,
                            help="Number of recent seasons for rolling aggregate (default: 3)")
        parser.add_argument("--min-coverage", type=float, default=0.50,
                            help="Minimum odds coverage ratio to include a season (default: 0.50)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Print results without writing to MongoDB")

    def run(self, args, *, league_loader=None, db_factory=None, **kwargs) -> int:
        from sportscore.services.market_calibration_service import (
            compute_and_store_market_calibration,
        )

        league = league_loader(args.league)
        db = db_factory()

        print(f"League: {league.display_name} ({league.league_id})")
        print(f"Master training CSV: {league.master_training_csv}")
        print()

        try:
            stats = compute_and_store_market_calibration(
                db, league,
                rolling_seasons=args.rolling_seasons,
                min_coverage=args.min_coverage,
                dry_run=args.dry_run,
                verbose=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print()
        print(f"Seasons computed: {stats['seasons_computed']}")
        if stats['rolling_brier'] is not None:
            print(f"Rolling Brier:    {stats['rolling_brier']:.4f}")
            print(f"Rolling Log-Loss: {stats['rolling_log_loss']:.4f}")
        if stats['n_written']:
            print(f"Docs written:     {stats['n_written']} to {stats['collection_name']}")

        return 0


class ComputeBinTrustCommand(SportCommand):
    name = "compute_bin_trust"
    help = "Compute bin trust weights from Kalshi portfolio P&L"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--bin-width", type=int, default=5,
                            help="Bin width in percent (default: 5)")
        parser.add_argument("--min-samples", type=int, default=10,
                            help="Minimum positions per bin for trust (default: 10)")
        parser.add_argument("--market-type", choices=["moneyline", "spread", "both"],
                            default="moneyline",
                            help="Market type to include (default: moneyline)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Print results without writing to MongoDB")

    def run(self, args, *, league_loader=None, db_factory=None, **kwargs) -> int:
        from sportscore.market import MarketConnector
        from sportscore.services.market_bins import (
            fetch_all_fills, fetch_all_settlements, compute_market_bins,
        )
        from sportscore.services.bin_trust import (
            compute_bin_trust_weights, trust_weights_to_doc,
        )

        league = league_loader(args.league)
        market_config = league.raw.get("market", {})

        print(f"League: {league.display_name} ({league.league_id})")

        # Build ticker prefixes
        ticker_prefixes = []
        if args.market_type in ("moneyline", "both"):
            series = market_config.get("series_ticker")
            if series:
                ticker_prefixes.append(series)
        if args.market_type in ("spread", "both"):
            spread_series = market_config.get("spread_series_ticker")
            if spread_series:
                ticker_prefixes.append(spread_series)

        if not ticker_prefixes:
            print("Error: No ticker prefixes found in league market config",
                  file=sys.stderr)
            return 1

        print(f"Ticker prefixes: {ticker_prefixes}")
        print(f"Market type: {args.market_type}")
        print(f"Bin width: {args.bin_width}%")
        print(f"Min samples: {args.min_samples}")
        print()

        # Init market connector
        api_key = os.environ.get("KALSHI_API_KEY")
        private_key_dir = os.environ.get("KALSHI_PRIVATE_KEY_DIR")
        if not api_key or not private_key_dir:
            print("Error: KALSHI_API_KEY and KALSHI_PRIVATE_KEY_DIR must be set",
                  file=sys.stderr)
            return 1

        connector = MarketConnector({
            "KALSHI_API_KEY": api_key,
            "KALSHI_PRIVATE_KEY_DIR": private_key_dir,
        })

        # Fetch fills and settlements
        print("Fetching fills...")
        fills = fetch_all_fills(connector)
        print(f"  Total fills: {len(fills)}")

        print("Fetching settlements...")
        settlements = fetch_all_settlements(connector)
        print(f"  Total settlements: {len(settlements)}")
        print()

        # Compute market bins
        bins = [(i, i + args.bin_width) for i in range(0, 100, args.bin_width)]
        bins_result = compute_market_bins(
            fills, settlements, ticker_prefixes,
            bins=bins,
        )

        print(f"Positions matched: {bins_result['n_positions']}")
        print(f"Unsettled: {bins_result['n_unsettled']}")
        print()

        # Compute trust weights
        trust_weights = compute_bin_trust_weights(
            bins_result,
            min_samples=args.min_samples,
        )

        # Display results
        print(f"{'Bin':>10s}  {'Count':>5s}  {'ROI':>7s}  {'Raw':>6s}  "
              f"{'Shrunk':>6s}  {'Trust':>6s}")
        print("-" * 50)
        for tw in trust_weights:
            label = f"{tw['prob_low']:.0f}-{tw['prob_high']:.0f}%"
            print(f"{label:>10s}  {tw['count']:5d}  {tw['roi']:6.1f}%  "
                  f"{tw['raw_trust']:6.3f}  {tw['shrunk']:6.3f}  "
                  f"{tw['trust']:6.3f}")
        print()

        if args.dry_run:
            print("[DRY RUN] Skipping MongoDB write")
            return 0

        # Store to MongoDB
        db = db_factory()
        coll_name = league.collections.get("bin_trust_weights")
        if not coll_name:
            print(f"Warning: No 'bin_trust_weights' collection configured "
                  f"for {league.league_id}", file=sys.stderr)
            print("Add 'bin_trust_weights: <name>' to the league YAML "
                  "under mongo.collections", file=sys.stderr)
            return 1

        doc = trust_weights_to_doc(
            trust_weights,
            league_id=league.league_id,
            ticker_prefixes=ticker_prefixes,
            n_positions=bins_result["n_positions"],
        )

        db[coll_name].insert_one(doc)
        print(f"Stored trust weights to {coll_name}")
        print(f"  Bins: {len(trust_weights)}")
        print(f"  Positions: {bins_result['n_positions']}")

        return 0


class DbIngestionCommand(SportCommand):
    name = "db_ingestion"
    help = "Pull data from ESPN and enrich DB (indexes, games, teams, rosters, ELO)"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--seasons", type=str, default=None,
                            help="Comma-separated seasons (e.g., '2023-2024,2024-2025')")
        parser.add_argument("--max-workers", type=int, default=None,
                            help="Max parallel workers for ESPN pull")
        parser.add_argument("--skip-espn", action="store_true",
                            help="Skip ESPN data pull")
        parser.add_argument("--skip-post", action="store_true",
                            help="Skip post-processing (teams, players, venues, rosters)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Show what would be done without modifying data")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Show detailed output")

    def run(self, args, *, league_loader=None, db_factory=None,
            plugin=None, **kwargs) -> int:
        if plugin is None:
            print("Error: sport plugin not available", file=sys.stderr)
            return 1

        pipeline = plugin.get_ingestion_pipeline(
            args.league,
            seasons=args.seasons.split(",") if args.seasons else None,
            max_workers=args.max_workers,
            skip_espn=args.skip_espn,
            skip_post=args.skip_post,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if pipeline is None:
            print(
                f"Error: db_ingestion is not implemented for this sport.\n"
                f"Implement get_ingestion_pipeline() in the sport plugin.",
                file=sys.stderr,
            )
            return 1

        result = pipeline.run()
        return 0 if result.success else 1


# All generic commands — main.py iterates this to build the command map
GENERIC_COMMANDS = [
    CacheLeagueStatsCommand(),
    ComputeMarketCalibrationCommand(),
    ComputeBinTrustCommand(),
    DbIngestionCommand(),
]
