"""Sport-agnostic CSV add/update logic for training data generation.

Provides TrainingCSVManager — a reusable class that any sport's CLI can use
for incremental column updates (--add --features) and season row replacement
(--add --season) on existing training CSVs.

Supports parallel processing via --workers / --chunk-size for all modes.

Usage:
    from sportscore.pipeline.training_csv import TrainingCSVManager

    manager = TrainingCSVManager(
        meta_columns=['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'season'],
        score_columns=['home_goals', 'away_goals'],
        target_columns=['MatchOutcome'],
    )
    mode = manager.resolve_add_mode(args, output_path)
    features = manager.resolve_features(args, all_features)
"""

import csv
import fnmatch
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional


def _render_progress(done, total, start_time, label="Processing"):
    """Render a progress line to stdout."""
    elapsed = time.time() - start_time
    pct = done / total * 100 if total > 0 else 0
    rate = done / elapsed if elapsed > 0 else 0
    if rate > 0 and done < total:
        eta_secs = int((total - done) / rate)
        mins, secs = divmod(eta_secs, 60)
        eta = f"{mins:02d}m {secs:02d}s"
    else:
        eta = "--:--"

    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

    print(
        f"\r  [{bar}] {pct:5.1f}% | {done:,}/{total:,} | "
        f"{rate:,.1f}/s | ETA {eta}",
        end="", flush=True,
    )


def _process_items_parallel(items, compute_fn, workers, chunk_size, label="Processing"):
    """Process items in parallel chunks using ThreadPoolExecutor.

    Args:
        items: List of items to process.
        compute_fn: fn(item) -> result or None.
        workers: Number of parallel workers.
        chunk_size: Items per chunk.
        label: Label for progress display.

    Returns:
        List of non-None results, preserving original order.
    """
    total = len(items)

    if workers <= 1 or total == 0:
        # Serial fallback
        results = []
        start = time.time()
        for i, item in enumerate(items):
            result = compute_fn(item)
            if result is not None:
                results.append(result)
            if (i + 1) % 100 == 0:
                _render_progress(i + 1, total, start, label)
        if total >= 100:
            _render_progress(total, total, start, label)
            print()
        return results

    # Parallel processing
    chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]
    num_chunks = len(chunks)

    lock = threading.Lock()
    stats = {"done": 0, "start": time.time(), "last_render": 0.0}

    def process_chunk(chunk):
        chunk_results = []
        for item in chunk:
            result = compute_fn(item)
            if result is not None:
                chunk_results.append(result)
            with lock:
                stats["done"] += 1
                now = time.time()
                if now - stats["last_render"] >= 0.15:
                    stats["last_render"] = now
                    _render_progress(stats["done"], total, stats["start"], label)
        return chunk_results

    print(f"  {label}: {total:,} items, {num_chunks} chunks, {workers} workers")

    ordered = [None] * num_chunks
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ordered[idx] = future.result()
            except Exception as e:
                print(f"\n  [ERROR] Chunk {idx} failed: {e}")
                ordered[idx] = []

    _render_progress(total, total, stats["start"], label)
    print()

    # Flatten in order
    all_results = []
    for chunk_results in ordered:
        if chunk_results:
            all_results.extend(chunk_results)

    elapsed = time.time() - stats["start"]
    rate = total / elapsed if elapsed > 0 else 0
    print(f"  Completed: {len(all_results):,} results in {elapsed:.1f}s ({rate:,.1f}/s)")
    return all_results


class TrainingCSVManager:
    """Sport-agnostic CSV add/update logic for training data generation."""

    def __init__(
        self,
        meta_columns: List[str],
        score_columns: List[str],
        target_columns: List[str],
        season_column: str = 'season',
    ):
        """Define the CSV schema (sport-specific column groups).

        Args:
            meta_columns: Columns for game metadata (date, teams, ids).
            score_columns: Columns for actual scores.
            target_columns: Columns for prediction targets.
            season_column: Name of the column holding the season value.
        """
        self.meta_columns = meta_columns
        self.score_columns = score_columns
        self.target_columns = target_columns
        self.season_column = season_column

    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------

    @staticmethod
    def add_arguments(parser):
        """Add --add, --features, --exclude-features, --season, --workers, --chunk-size."""
        parser.add_argument(
            "--add", action="store_true",
            help="Add/update existing CSV: with --features updates columns, "
                 "with --season replaces season rows (auto-enabled when CSV exists)",
        )
        parser.add_argument(
            "--features", type=str, default=None,
            help="Comma-separated feature names or patterns (e.g., 'elo|*,xg|*')",
        )
        parser.add_argument(
            "--exclude-features", type=str, default=None,
            help="Comma-separated feature names or patterns to EXCLUDE",
        )
        parser.add_argument(
            "--season", type=str, default=None,
            help="Single season for add-mode (e.g., '2024-2025')",
        )
        parser.add_argument(
            "--workers", type=int, default=1,
            help="Number of parallel workers (default: 1 = serial)",
        )
        parser.add_argument(
            "--chunk-size", type=int, default=500,
            help="Items per chunk for parallel processing (default: 500)",
        )

    # ------------------------------------------------------------------
    # Feature pattern expansion
    # ------------------------------------------------------------------

    @staticmethod
    def expand_feature_patterns(specs: List[str], all_features: List[str]) -> List[str]:
        """Expand wildcard patterns (e.g. 'elo|*') against the full feature list.

        Args:
            specs: Feature names or glob patterns.
            all_features: Complete list of valid feature names.

        Returns:
            Expanded, deduplicated list preserving insertion order.
        """
        expanded = []
        seen: set = set()

        for spec in specs:
            spec = spec.strip()
            if not spec:
                continue

            if '*' in spec or '?' in spec:
                matched = [f for f in all_features if fnmatch.fnmatch(f, spec)]
                if matched:
                    print(f"  Pattern '{spec}' matched {len(matched)} features")
                    for f in matched:
                        if f not in seen:
                            expanded.append(f)
                            seen.add(f)
                else:
                    print(f"  Warning: Pattern '{spec}' matched no features")
            else:
                if spec not in seen:
                    expanded.append(spec)
                    seen.add(spec)

        return expanded

    @classmethod
    def resolve_features(
        cls,
        args,
        all_features: List[str],
    ) -> List[str]:
        """Parse --features, expand patterns, apply --exclude-features.

        Returns the final feature list.  When --features is not set the full
        *all_features* list is returned (minus any exclusions).
        """
        features_spec = getattr(args, 'features', None)
        exclude_spec = getattr(args, 'exclude_features', None)

        if features_spec:
            specs = [s.strip() for s in features_spec.split(",")]
            features = cls.expand_feature_patterns(specs, all_features)
        else:
            features = list(all_features)

        if exclude_spec:
            exclude_specs = [s.strip() for s in exclude_spec.split(",")]
            exclude_set = set(cls.expand_feature_patterns(exclude_specs, all_features))
            before = len(features)
            features = [f for f in features if f not in exclude_set]
            print(f"  Excluded {before - len(features)} features")

        return features

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def resolve_add_mode(self, args, output_path: str, season_args=('season',)) -> str:
        """Determine mode: ``'add_features'``, ``'add_seasons'``, or ``'full'``.

        Also handles auto-enable of ``--add`` when the CSV already exists and
        a scoped operation (``--features`` / ``--season``) is requested.

        Args:
            args: Parsed CLI arguments.
            output_path: Path to the output CSV.
            season_args: Attribute names on *args* that indicate a season filter.
                Soccer uses the default ``('season',)``; basketball passes
                ``('season', 'seasons')`` because it supports both flags.

        Raises:
            ValueError: If ``--add`` is set without ``--features`` or ``--season``.
            FileNotFoundError: If ``--add`` is set but the CSV does not exist.
        """
        has_features = (
            getattr(args, 'features', None) is not None
            or getattr(args, 'exclude_features', None) is not None
        )
        has_season = any(getattr(args, attr, None) is not None for attr in season_args)
        is_add = getattr(args, 'add', False)

        # Auto-enable --add when CSV exists and a scoped operation is requested
        if not is_add and os.path.exists(output_path):
            if has_features or has_season:
                args.add = True
                is_add = True
                print(f"[auto] Existing CSV detected — using --add mode")

        if not is_add:
            return 'full'

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"--add requires existing CSV at {output_path}"
            )

        if not has_features and not has_season:
            raise ValueError(
                "--add requires --features/--exclude-features (to update columns) "
                "or --season (to replace season rows)"
            )

        if has_features:
            return 'add_features'
        return 'add_seasons'

    # ------------------------------------------------------------------
    # CSV introspection
    # ------------------------------------------------------------------

    def detect_features_from_csv(self, csv_path: str) -> List[str]:
        """Read the header of an existing CSV and return the feature column names.

        Feature columns are those not in meta / score / target column sets.
        """
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            all_cols = list(reader.fieldnames or [])

        exclude = (
            set(self.meta_columns)
            | set(self.score_columns)
            | set(self.target_columns)
        )
        return [c for c in all_cols if c not in exclude]

    # ------------------------------------------------------------------
    # Full generation mode (parallel)
    # ------------------------------------------------------------------

    def generate_csv_parallel(
        self,
        items: List[dict],
        compute_fn: Callable[[dict], Optional[dict]],
        feature_names: List[str],
        output_path: str,
        workers: int = 1,
        chunk_size: int = 500,
    ) -> int:
        """Generate a full training CSV with optional parallel processing.

        Args:
            items: List of game dicts to process.
            compute_fn: fn(game_dict) -> row_dict (with meta+features+score+target)
                        or None to skip.
            feature_names: Feature columns for the CSV.
            output_path: Where to write the CSV.
            workers: Number of parallel workers (1 = serial).
            chunk_size: Items per chunk for parallel processing.

        Returns:
            Number of rows written.
        """
        total = len(items)
        print(f"\nProcessing {total:,} games...")

        rows = _process_items_parallel(
            items, compute_fn, workers, chunk_size, label="Generating features",
        )

        # Define column order
        all_cols = (
            self.meta_columns
            + sorted(feature_names)
            + self.score_columns
            + self.target_columns
        )

        # Write CSV
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        print(f"  Total: {len(rows):,} games written to {output_path}")
        print(f"  Columns: {len(all_cols)} ({len(self.meta_columns)} meta + "
              f"{len(feature_names)} features + {len(self.score_columns)} score + "
              f"{len(self.target_columns)} target)")
        return len(rows)

    # ------------------------------------------------------------------
    # Column-update mode (--add --features)
    # ------------------------------------------------------------------

    def add_features_to_csv(
        self,
        csv_path: str,
        compute_fn: Callable[[dict], dict],
        feature_names: List[str],
        seasons: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        workers: int = 1,
        chunk_size: int = 500,
    ) -> int:
        """Column-update mode: load CSV, compute new features, merge columns, save.

        Args:
            csv_path: Path to existing CSV.
            compute_fn: ``fn(row_dict) -> {feature_name: value}``.
            feature_names: Feature columns being added / updated.
            seasons: If set, only update rows whose season column matches.
            output_path: Defaults to *csv_path* (overwrite in place).
            workers: Number of parallel workers (1 = serial).
            chunk_size: Items per chunk for parallel processing.

        Returns:
            Number of rows updated.
        """
        output_path = output_path or csv_path

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_columns = list(reader.fieldnames or [])
            rows = list(reader)

        print(f"Loaded {len(rows)} rows from {csv_path}")

        # New feature columns that aren't already in the CSV
        existing_set = set(existing_columns)
        new_feature_cols = [c for c in sorted(feature_names) if c not in existing_set]

        # Insert new columns before score/target columns
        score_target = set(self.score_columns) | set(self.target_columns)
        insert_pos = len(existing_columns)
        for i, col in enumerate(existing_columns):
            if col in score_target:
                insert_pos = i
                break

        all_columns = (
            existing_columns[:insert_pos]
            + new_feature_cols
            + existing_columns[insert_pos:]
        )

        if new_feature_cols:
            print(f"  Adding {len(new_feature_cols)} new feature columns")
        updating = [
            c for c in feature_names
            if c in existing_set and c not in score_target
            and c not in set(self.meta_columns)
        ]
        if updating:
            print(f"  Updating {len(updating)} existing feature columns")

        # Filter to rows that need processing
        if seasons:
            season_set = set(seasons)
            to_process = [(i, row) for i, row in enumerate(rows)
                          if row.get(self.season_column) in season_set]
        else:
            to_process = list(enumerate(rows))

        # Compute features (serial or parallel)
        def _compute_for_row(idx_row):
            idx, row = idx_row
            features = compute_fn(row)
            return (idx, features) if features else None

        results = _process_items_parallel(
            to_process, _compute_for_row, workers, chunk_size,
            label="Computing features",
        )

        # Apply results back to rows
        updated = 0
        for idx, features in results:
            rows[idx].update(features)
            updated += 1

        # Write back
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        print(f"  {updated} rows updated, {len(all_columns)} columns -> {output_path}")
        return updated

    # ------------------------------------------------------------------
    # Row-replace mode (--add --season)
    # ------------------------------------------------------------------

    def replace_seasons_in_csv(
        self,
        csv_path: str,
        new_rows: List[dict],
        target_seasons: List[str],
        all_columns: List[str],
        output_path: Optional[str] = None,
    ) -> int:
        """Row-replace mode: load CSV, drop target season rows, concat new rows, save.

        Args:
            csv_path: Path to existing CSV.
            new_rows: New row dicts to insert.
            target_seasons: Seasons whose existing rows are dropped first.
            all_columns: Full ordered column list for the output CSV.
            output_path: Defaults to *csv_path* (overwrite in place).

        Returns:
            Total number of rows in the output file.
        """
        output_path = output_path or csv_path

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

        target_set = set(target_seasons)
        kept = [r for r in existing_rows if r.get(self.season_column) not in target_set]
        removed = len(existing_rows) - len(kept)
        print(f"Removed {removed} rows for seasons: {target_seasons}")

        combined = kept + new_rows
        print(f"Added {len(new_rows)} new rows, total: {len(combined)}")

        # Sort chronologically by Year / Month / Day
        def sort_key(row):
            try:
                return (
                    int(row.get('Year', 0)),
                    int(row.get('Month', 0)),
                    int(row.get('Day', 0)),
                )
            except (ValueError, TypeError):
                return (0, 0, 0)

        combined.sort(key=sort_key)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(combined)

        print(f"\n  {len(combined)} rows written to {output_path}")
        return len(combined)
