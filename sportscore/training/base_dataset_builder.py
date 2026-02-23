"""
Base Dataset Builder — sport-agnostic dataset slicing and caching.

Reads a master training CSV, applies filters (year, date, min games),
selects features, and caches results.  Subclasses override class
attributes and _resolve_blocks() for sport-specific behavior.

Usage (via sport-specific subclass):
    builder = SoccerDatasetBuilder(league=league)
    result = builder.build_dataset({
        'feature_blocks': ['outcome_strength', 'attacking_quality'],
        'begin_year': 2015,
        'min_games_played': 5,
    })
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sportscore.training.schemas import DatasetSpec


class BaseDatasetBuilder:
    """Sport-agnostic dataset builder with caching.

    Subclasses should override:
        META_COLUMNS, TARGET_COLUMNS, SCORE_COLUMNS,
        DEFAULT_BEGIN_YEAR, SEASON_START_MONTH,
        _resolve_blocks()
    """

    # Subclasses override these
    META_COLUMNS = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id']
    TARGET_COLUMNS = ['HomeWon']
    SCORE_COLUMNS = ['home_points', 'away_points']
    DEFAULT_BEGIN_YEAR = 2012
    SEASON_START_MONTH = 10  # NBA: Oct, Soccer: Aug

    def __init__(self, master_training_path: str, cache_dir: str,
                 league=None, db=None):
        self.master_training_path = master_training_path
        self.cache_dir = cache_dir
        self.league = league
        self.league_id = league.league_id if league else "default"
        self.db = db
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataset(self, dataset_spec: dict) -> dict:
        """Build a training dataset from specification.

        Args:
            dataset_spec: Dict matching DatasetSpec schema.

        Returns:
            Dict with dataset_id, schema, row_count, feature_count,
            csv_path, cached, and optional dropped_features.
        """
        try:
            spec = DatasetSpec(**dataset_spec)
        except Exception as e:
            raise ValueError(f"Invalid dataset spec: {e}")

        force_rebuild = spec.force_rebuild

        # Hash for cache key (exclude force_rebuild)
        spec_dict = spec.dict(exclude_none=True)
        spec_dict.pop('force_rebuild', None)
        dataset_id = self._hash_spec(spec_dict)

        cache_file = os.path.join(self.cache_dir, f'dataset_{dataset_id}.csv')
        cache_meta_file = os.path.join(self.cache_dir, f'dataset_{dataset_id}_meta.json')

        # Invalidate cache if force_rebuild
        if force_rebuild:
            for f in (cache_file, cache_meta_file):
                if os.path.exists(f):
                    os.remove(f)

        # Check cache
        if not force_rebuild and os.path.exists(cache_file) and os.path.exists(cache_meta_file):
            result = self._try_load_cache(cache_file, cache_meta_file, dataset_id)
            if result is not None:
                return result

        # Read master CSV
        if not os.path.exists(self.master_training_path):
            raise ValueError(
                f"Master training CSV not found at {self.master_training_path}. "
                f"Generate it first with: sportscore generate_training_data <league>"
            )

        master_df = pd.read_csv(self.master_training_path)

        # Determine which columns exist
        meta_cols = [c for c in self.META_COLUMNS if c in master_df.columns]
        target_cols = [c for c in self.TARGET_COLUMNS if c in master_df.columns]
        score_cols = [c for c in self.SCORE_COLUMNS if c in master_df.columns]
        non_feature = set(meta_cols + target_cols + score_cols)

        # Apply filters
        master_df = self._apply_year_filters(master_df, spec)
        master_df = self._apply_date_filters(master_df, spec)
        if spec.min_games_played is not None and spec.min_games_played > 0:
            master_df = self._apply_min_games_filter(master_df, spec.min_games_played)

        # Resolve features
        features = self._resolve_features(spec, master_df, non_feature)

        # Check which features are actually in the master CSV
        master_feature_set = set(master_df.columns) - non_feature
        missing = [f for f in features if f not in master_feature_set]
        available = [f for f in features if f in master_feature_set]

        if missing:
            print(f"  [WARNING] {len(missing)} requested features not in master CSV (dropped)")
            if len(missing) <= 20:
                print(f"  [WARNING] Missing: {missing}")

        if not available:
            raise ValueError(
                f"No requested features found in master CSV. "
                f"Missing (first 10): {missing[:10]}"
            )

        features = available

        # Extract columns in order: meta + features (sorted) + scores + targets
        ordered_features = sorted(features)
        columns = meta_cols + ordered_features + score_cols + target_cols
        columns = [c for c in columns if c in master_df.columns]
        extracted_df = master_df[columns].copy()

        if extracted_df.empty:
            raise ValueError("No rows remain after filtering.")

        # Write cache
        extracted_df.to_csv(cache_file, index=False)

        feature_cols = [c for c in extracted_df.columns if c not in non_feature]

        metadata = {
            'dataset_id': dataset_id,
            'spec': spec_dict,
            'schema': feature_cols,
            'row_count': len(extracted_df),
            'feature_count': len(feature_cols),
            'created_at': datetime.utcnow().isoformat(),
        }
        if missing:
            metadata['dropped_features'] = missing
            metadata['requested_feature_count'] = len(missing) + len(feature_cols)

        with open(cache_meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        result = {
            'dataset_id': dataset_id,
            'schema': feature_cols,
            'row_count': len(extracted_df),
            'feature_count': len(feature_cols),
            'csv_path': cache_file,
            'cached': False,
        }
        if missing:
            result['dropped_features'] = missing
            result['requested_feature_count'] = metadata['requested_feature_count']

        print(f"  [INFO] Dataset: {len(extracted_df)} rows, {len(feature_cols)} features")
        return result

    # ------------------------------------------------------------------
    # Season helpers
    # ------------------------------------------------------------------

    def _compute_season_start_year(self, df: pd.DataFrame) -> pd.Series:
        """Compute the season start year for each row.

        If Month >= SEASON_START_MONTH, season started in that Year;
        otherwise season started in Year - 1.
        """
        return np.where(
            df['Month'].astype(int) >= self.SEASON_START_MONTH,
            df['Year'].astype(int),
            df['Year'].astype(int) - 1,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _apply_year_filters(self, df: pd.DataFrame, spec) -> pd.DataFrame:
        """Filter by begin_year, end_year, exclude_seasons using SeasonStartYear."""
        begin_year = spec.begin_year if spec.begin_year is not None else self.DEFAULT_BEGIN_YEAR

        needs_ssy = (begin_year or spec.end_year or spec.exclude_seasons)
        if not needs_ssy:
            return df

        df = df.copy()
        df['_SeasonStartYear'] = self._compute_season_start_year(df)

        if begin_year:
            df = df[df['_SeasonStartYear'] >= int(begin_year)]

        if spec.end_year:
            df = df[df['_SeasonStartYear'] <= int(spec.end_year)]

        if spec.exclude_seasons:
            before = len(df)
            df = df[~df['_SeasonStartYear'].isin(spec.exclude_seasons)]
            print(f"  [INFO] Excluded seasons {spec.exclude_seasons}: {before} -> {len(df)} games")

        df = df.drop('_SeasonStartYear', axis=1)
        return df

    def _apply_date_filters(self, df: pd.DataFrame, spec) -> pd.DataFrame:
        """Filter by begin_date and/or end_date."""
        if not spec.begin_date and not spec.end_date:
            return df

        df = df.copy()
        df['_Date'] = pd.to_datetime(
            df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1)
        )

        if spec.begin_date:
            df = df[df['_Date'] >= pd.to_datetime(spec.begin_date)]
        if spec.end_date:
            df = df[df['_Date'] <= pd.to_datetime(spec.end_date)]

        df = df.drop('_Date', axis=1)
        return df

    def _apply_min_games_filter(self, df: pd.DataFrame, min_games: int) -> pd.DataFrame:
        """Both teams must have >= min_games prior games in same season."""
        before = len(df)
        df = df.copy()

        # Build season string
        df['_Season'] = np.where(
            df['Month'].astype(int) >= self.SEASON_START_MONTH,
            df['Year'].astype(int).astype(str) + '-' + (df['Year'].astype(int) + 1).astype(str),
            (df['Year'].astype(int) - 1).astype(str) + '-' + df['Year'].astype(int).astype(str),
        )
        df['_date_key'] = (
            df['Year'].astype(int) * 10000 +
            df['Month'].astype(int) * 100 +
            df['Day'].astype(int)
        )

        # Home prior counts
        home_keys = ['Year', 'Month', 'Day', 'Home']
        home_seq = df[home_keys + ['_Season', '_date_key']].copy()
        home_seq = home_seq.sort_values(['_Season', 'Home', '_date_key'])
        home_seq['_homePrior'] = home_seq.groupby(['_Season', 'Home']).cumcount()
        df = df.merge(home_seq[home_keys + ['_homePrior']], on=home_keys, how='left')

        # Away prior counts
        away_keys = ['Year', 'Month', 'Day', 'Away']
        away_seq = df[away_keys + ['_Season', '_date_key']].copy()
        away_seq = away_seq.sort_values(['_Season', 'Away', '_date_key'])
        away_seq['_awayPrior'] = away_seq.groupby(['_Season', 'Away']).cumcount()
        df = df.merge(away_seq[away_keys + ['_awayPrior']], on=away_keys, how='left')

        # Filter
        df = df[(df['_homePrior'] >= min_games) & (df['_awayPrior'] >= min_games)].copy()

        # Cleanup
        helper_cols = ['_date_key', '_homePrior', '_awayPrior', '_Season']
        df.drop(columns=[c for c in helper_cols if c in df.columns], inplace=True)

        print(f"  [INFO] min_games_played >= {min_games}: {before} -> {len(df)} games")

        if df.empty:
            raise ValueError(
                f"No data after min_games_played >= {min_games}. "
                f"Try a lower threshold."
            )
        return df

    # ------------------------------------------------------------------
    # Feature resolution
    # ------------------------------------------------------------------

    def _resolve_features(self, spec, master_df: pd.DataFrame,
                          non_feature: set) -> List[str]:
        """Resolve the final feature list from spec."""
        if spec.individual_features:
            return list(spec.individual_features)

        if spec.feature_blocks:
            return self._resolve_blocks(spec.feature_blocks)

        # Fallback: all features in master CSV
        return self._get_all_features_from_master(master_df, non_feature)

    def _resolve_blocks(self, blocks: List[str]) -> List[str]:
        """Map block names to feature lists. Subclasses must override."""
        raise NotImplementedError(
            "Subclass must implement _resolve_blocks()"
        )

    def _get_all_features_from_master(self, master_df: pd.DataFrame,
                                      non_feature: set) -> List[str]:
        """Return all non-meta/target/score columns from master CSV."""
        return [c for c in master_df.columns if c not in non_feature]

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _hash_spec(self, spec: dict) -> str:
        """SHA256 hash of normalized spec dict, 16 chars."""
        normalized = {}
        for key, value in sorted(spec.items()):
            if value is not None:
                normalized[key] = value
        normalized['_league_id'] = self.league_id
        spec_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(spec_str.encode()).hexdigest()[:16]

    def _try_load_cache(self, cache_file: str, cache_meta_file: str,
                        dataset_id: str) -> Optional[dict]:
        """Try to load a cached dataset. Returns result dict or None."""
        try:
            if os.path.getsize(cache_file) == 0:
                return None

            # Invalidate if the master CSV is newer than the cached file
            if os.path.exists(self.master_training_path):
                master_mtime = os.path.getmtime(self.master_training_path)
                cache_mtime = os.path.getmtime(cache_file)
                if master_mtime > cache_mtime:
                    print(f"  [INFO] Master CSV is newer than cache — rebuilding dataset")
                    return None

            with open(cache_meta_file, 'r') as f:
                metadata = json.load(f)

            if metadata.get('row_count', 0) == 0:
                return None

            # Quick sanity: can we read the CSV?
            df = pd.read_csv(cache_file, nrows=1)
            if df.empty:
                return None

            result = {
                'dataset_id': dataset_id,
                'schema': metadata['schema'],
                'row_count': metadata['row_count'],
                'feature_count': metadata['feature_count'],
                'csv_path': cache_file,
                'cached': True,
            }
            if 'dropped_features' in metadata:
                result['dropped_features'] = metadata['dropped_features']
                result['requested_feature_count'] = metadata.get(
                    'requested_feature_count', metadata['feature_count']
                )
            return result

        except Exception:
            # Invalid cache, will rebuild
            for f in (cache_file, cache_meta_file):
                if os.path.exists(f):
                    os.remove(f)
            return None
