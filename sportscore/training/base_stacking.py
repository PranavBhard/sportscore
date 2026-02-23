"""
Base Stacking Trainer - Sport-agnostic meta-model training infrastructure.

Trains meta-models that combine predictions from multiple base models.
Subclasses provide sport-specific repository access, dataset building,
and legacy model loading.
"""

import os
import re
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from bson import ObjectId

from sportscore.training.model_factory import create_model_with_c
from sportscore.training.model_evaluation import _multiclass_brier_score, _classify
from sportscore.training.run_tracker import RunTracker
from sportscore.training.cache_utils import read_csv_safe

# Supported meta-model types
META_MODEL_TYPES = ['LogisticRegression', 'SVM', 'GradientBoosting']
C_SUPPORTED_META_MODELS = ['LogisticRegression', 'SVM']


class BaseStackingTrainer(ABC):
    """Trains stacked models that combine multiple base model predictions."""

    # --- Class attributes (override in subclass for different column conventions) ---
    TARGET_COL = 'HomeWon'
    META_COLS = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id']
    PREDICTION_COLS = ['pred_home_points', 'pred_away_points', 'pred_point_total', 'pred_margin']
    POINTS_COLS = ['home_points', 'away_points']
    DEFAULT_SEASON_START_MONTH = 10
    ENSEMBLE_DIR = 'cli/models/ensembles'
    META_MODEL_TYPES = META_MODEL_TYPES
    C_SUPPORTED_META_MODELS = C_SUPPORTED_META_MODELS

    def _get_ensembles_dir(self) -> str:
        """Resolve ENSEMBLE_DIR to an absolute path using league repo root."""
        if os.path.isabs(self.ENSEMBLE_DIR):
            return self.ENSEMBLE_DIR
        if self.league and hasattr(self.league, '_repo_root'):
            return os.path.join(self.league._repo_root, self.ENSEMBLE_DIR)
        return os.path.abspath(self.ENSEMBLE_DIR)

    def __init__(self, db=None, league=None):
        """
        Initialize BaseStackingTrainer.

        Args:
            db: Database instance (optional, subclass provides default)
            league: League config instance for league-specific collections
        """
        self.db = db
        self.league = league

        # Initialize repository and builders via subclass hooks
        self._classifier_repo = self._get_model_config_repository()
        self._dataset_builder = self._get_dataset_builder()
        self.run_tracker = RunTracker(db=self.db, league=self.league)

    @abstractmethod
    def _get_model_config_repository(self):
        """Return the model config repository for loading base model configs."""
        ...

    @abstractmethod
    def _get_dataset_builder(self):
        """Return the dataset builder for constructing training datasets."""
        ...

    @abstractmethod
    def _load_legacy_model(self, run_id: str):
        """
        Load a model from a legacy run_id (fallback path).

        Returns:
            Tuple of (model, scaler, feature_names)
        """
        ...

    @property
    def _season_start_month(self) -> int:
        """Get season start month from league config or use default."""
        if self.league and hasattr(self.league, 'season_start_month'):
            return self.league.season_start_month
        return self.DEFAULT_SEASON_START_MONTH

    def train_stacked_model(
        self,
        dataset_spec: Dict,
        session_id: str,
        base_run_ids: List[str] = None,
        base_config_ids: List[str] = None,
        meta_model_type: str = 'LogisticRegression',
        meta_c_value: float = 0.1,
        stacking_mode: str = 'naive',
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False,
        use_logit: bool = False,
        logit_eps: float = 1e-6,
        meta_calibration_method: Optional[str] = None,
        meta_train_years: Optional[List[int]] = None,
        meta_calibration_years: Optional[List[int]] = None,
        meta_evaluation_year: Optional[int] = None,
    ) -> Dict:
        """
        Train a stacked model that combines multiple base model predictions.

        Args:
            dataset_spec: Dataset specification (must match base models' configs)
            session_id: Chat session ID
            base_run_ids: List of run_ids for base models (legacy support)
            base_config_ids: List of MongoDB config _id strings for base models (preferred)
            meta_model_type: Type of meta-model to train
            meta_c_value: C-value for meta-model (if applicable)
            stacking_mode: 'naive' (default) or 'informed'
            meta_features: Optional list of feature names to include in meta-model
            use_disagree: If True, include pairwise disagreement features
            use_conf: If True, include confidence features
            use_logit: If True, feed logit(p_*) to meta-learner (same logit_eps persisted for prediction)
            logit_eps: Epsilon for clipping p_* before logit when use_logit=True (persisted in artifact)

        Returns:
            Dict with run_id, metrics, diagnostics, artifacts
        """
        # Use base_config_ids if provided, otherwise fall back to base_run_ids
        if base_config_ids is not None:
            base_ids = base_config_ids
        else:
            base_ids = base_run_ids

        if len(base_ids) < 2:
            raise ValueError(f"Stacking requires at least 2 base models, got {len(base_ids)}")

        # Validate meta_model_type
        if meta_model_type not in self.META_MODEL_TYPES:
            raise ValueError(f"Invalid meta_model_type: {meta_model_type}. Must be one of {self.META_MODEL_TYPES}")

        # Validate stacking_mode
        if stacking_mode not in ['naive', 'informed']:
            raise ValueError(f"Invalid stacking_mode: {stacking_mode}. Must be 'naive' or 'informed'")

        # Load and validate base models
        base_models_info = self._load_base_models(base_ids)

        # Validate all base models have compatible configs
        self._validate_base_models_compatible(base_models_info)

        # Get reference config from first base model
        ref_config = base_models_info[0]['config']
        ref_splits = ref_config.get('splits', {})

        # Extract time-based calibration parameters
        calibration_years = ref_splits.get('calibration_years', [2023])
        evaluation_year = ref_splits.get('evaluation_year', 2024)
        begin_year = ref_splits.get('begin_year', 2012)

        # Ensure calibration_years is a list
        if not isinstance(calibration_years, list):
            calibration_years = [calibration_years]

        # Create stacking run
        stacking_config = {
            'task': 'stacking',
            'base_run_ids': base_run_ids,
            'meta_model_type': meta_model_type,
            'meta_c_value': meta_c_value if meta_model_type in self.C_SUPPORTED_META_MODELS else None,
            'stacking_mode': stacking_mode,
            'meta_features': meta_features,
            'use_disagree': use_disagree,
            'use_conf': use_conf,
            'use_logit': use_logit,
            'logit_eps': logit_eps,
            'meta_calibration_method': meta_calibration_method,
            'meta_train_years': meta_train_years,
            'meta_calibration_years': meta_calibration_years,
            'meta_evaluation_year': meta_evaluation_year,
            'splits': ref_splits,
            'features': ref_config.get('features', {})
        }

        run_id = self.run_tracker.create_run(
            config=stacking_config,
            dataset_id=None,
            model_type='Stacked',
            session_id=session_id,
            baseline=False
        )

        self.run_tracker.update_run(run_id, status='running')

        try:
            # Extract time configuration for validation
            begin_year = dataset_spec.get('begin_year')
            calibration_years = dataset_spec.get('calibration_years', [])
            evaluation_year = dataset_spec.get('evaluation_year')
            min_games_played = dataset_spec.get('min_games_played', 0)

            if not all([begin_year, calibration_years, evaluation_year]):
                raise ValueError("Missing required time configuration (begin_year, calibration_years, evaluation_year)")

            print(f"[STACKING] Time config - Begin: {begin_year}, Calibration: {calibration_years}, Evaluation: {evaluation_year}")
            print(f"[STACKING] Min games played: {min_games_played}")

            # Collect ALL unique features needed by ALL base models
            all_base_features = set()
            for model_info in base_models_info:
                model_features = model_info.get('feature_names', [])
                all_base_features.update(model_features)

            # Also include meta_features if provided (for informed stacking)
            if meta_features:
                all_base_features.update(meta_features)

            print(f"[STACKING] Collected {len(all_base_features)} unique features from {len(base_models_info)} base models")

            # Build dataset for calibration and evaluation periods
            excluded_keys = ['calibration_years', 'evaluation_year', 'use_master', 'training_csv',
                            'model_type', 'best_c_value', 'config_hash', 'feature_set_hash']
            dataset_spec_clean = {k: v for k, v in dataset_spec.items()
                                 if k not in excluded_keys}

            # Add all base model features to the dataset request
            dataset_spec_clean['individual_features'] = sorted(all_base_features)

            # Ensure begin_year is set
            if 'begin_year' not in dataset_spec_clean:
                dataset_spec_clean['begin_year'] = begin_year
            print(f"[STACKING] Building dataset with begin_year={dataset_spec_clean.get('begin_year')}, calibration_years={calibration_years}, evaluation_year={evaluation_year}")
            dataset_result = self._dataset_builder.build_dataset(dataset_spec_clean)
            csv_path = dataset_result['csv_path']

            # Load dataset
            df = read_csv_safe(csv_path)
            if df.empty:
                raise ValueError(f"Dataset is empty: {csv_path}")

            # Calculate SeasonStartYear for filtering
            df['SeasonStartYear'] = np.where(
                df['Month'] >= self._season_start_month,
                df['Year'],
                df['Year'] - 1
            )

            # Temporal split
            cal_mask = df['SeasonStartYear'].isin(calibration_years)
            eval_mask = df['SeasonStartYear'] == evaluation_year
            train_mask = df['SeasonStartYear'] < min(calibration_years) if calibration_years else df['SeasonStartYear'] < evaluation_year

            df_cal = df[cal_mask].copy()
            df_eval = df[eval_mask].copy()
            df_train = df[train_mask].copy()

            if len(df_cal) == 0:
                unique_seasons = sorted(df['SeasonStartYear'].unique().tolist())
                raise ValueError(
                    f"No data found for calibration years {calibration_years}. "
                    f"Dataset has SeasonStartYear values: {unique_seasons}"
                )
            if len(df_eval) == 0:
                unique_seasons = sorted(df['SeasonStartYear'].unique().tolist())
                raise ValueError(
                    f"No data found for evaluation year {evaluation_year}. "
                    f"Dataset has SeasonStartYear values: {unique_seasons}"
                )

            print(f"[STACKING] Temporal split:")
            print(f"[STACKING]   Base model train period: {begin_year} to {min(calibration_years)-1 if calibration_years else evaluation_year-1} ({len(df_train)} games)")
            print(f"[STACKING]   Meta-model training (calibration only): {calibration_years} ({len(df_cal)} games)")
            print(f"[STACKING]   Evaluation period: {evaluation_year} ({len(df_eval)} games)")

            # --- Meta-model data splits ---
            # When meta_calibration_method is set, we do a 3-way meta split:
            #   meta_train_years  -> train meta-model
            #   meta_calibration_years -> fit calibrator
            #   meta_evaluation_year   -> evaluate
            # Otherwise (default): df_cal -> train meta, df_eval -> evaluate, no calibrator.
            meta_calibrator = None

            if meta_calibration_method:
                if not meta_train_years or not meta_calibration_years or not meta_evaluation_year:
                    raise ValueError(
                        "When meta_calibration_method is set, meta_train_years, "
                        "meta_calibration_years, and meta_evaluation_year are all required."
                    )

                df_meta_train = df[df['SeasonStartYear'].isin(meta_train_years)].copy()
                df_meta_cal = df[df['SeasonStartYear'].isin(meta_calibration_years)].copy()
                df_meta_eval = df[df['SeasonStartYear'] == meta_evaluation_year].copy()

                if len(df_meta_train) == 0:
                    raise ValueError(f"No data found for meta_train_years {meta_train_years}")
                if len(df_meta_cal) == 0:
                    raise ValueError(f"No data found for meta_calibration_years {meta_calibration_years}")
                if len(df_meta_eval) == 0:
                    raise ValueError(f"No data found for meta_evaluation_year {meta_evaluation_year}")

                print(f"[STACKING] Meta 3-way split (with calibration):")
                print(f"[STACKING]   Meta-train: {meta_train_years} ({len(df_meta_train)} games)")
                print(f"[STACKING]   Meta-cal:   {meta_calibration_years} ({len(df_meta_cal)} games)")
                print(f"[STACKING]   Meta-eval:  {meta_evaluation_year} ({len(df_meta_eval)} games)")
            else:
                df_meta_train = df_cal
                df_meta_cal = None
                df_meta_eval = df_eval

            # Generate stacking data from meta-train period
            stacking_df = self._generate_stacking_data(
                base_models_info=base_models_info,
                df=df_meta_train,
                calibration_years=calibration_years,
                stacking_mode=stacking_mode,
                meta_features=meta_features,
                use_disagree=use_disagree,
                use_conf=use_conf,
                use_logit=use_logit,
                logit_eps=logit_eps
            )

            # Train meta-model on meta-train predictions
            meta_model, meta_scaler = self._train_meta_model(
                stacking_df=stacking_df,
                meta_model_type=meta_model_type,
                meta_c_value=meta_c_value
            )

            # Fit calibrator on meta-cal period if requested
            if meta_calibration_method and df_meta_cal is not None:
                stacking_df_cal = self._generate_stacking_data(
                    base_models_info=base_models_info,
                    df=df_meta_cal,
                    calibration_years=calibration_years,
                    stacking_mode=stacking_mode,
                    meta_features=meta_features,
                    use_disagree=use_disagree,
                    use_conf=use_conf,
                    use_logit=use_logit,
                    logit_eps=logit_eps
                )
                meta_calibrator = self._fit_meta_calibrator(
                    meta_model=meta_model,
                    meta_scaler=meta_scaler,
                    stacking_df_cal=stacking_df_cal,
                    method=meta_calibration_method
                )

            # Evaluate stacked model on evaluation period
            metrics, diagnostics = self._evaluate_stacked_model(
                meta_model=meta_model,
                meta_scaler=meta_scaler,
                base_models_info=base_models_info,
                df=df_meta_eval,
                evaluation_year=meta_evaluation_year if meta_calibration_method else evaluation_year,
                calibration_years=calibration_years,
                begin_year=begin_year,
                n_train_samples=len(df_train),
                n_cal_samples=len(df_meta_train),
                stacking_mode=stacking_mode,
                meta_features=meta_features,
                use_disagree=use_disagree,
                use_conf=use_conf,
                use_logit=use_logit,
                logit_eps=logit_eps,
                meta_calibrator=meta_calibrator,
                meta_calibration_method=meta_calibration_method,
            )

            # Save ensemble artifacts to disk
            meta_feature_cols = [c for c in stacking_df.columns if c != self.TARGET_COL]
            artifact_paths = self._save_ensemble_artifacts(
                run_id=run_id,
                meta_model=meta_model,
                meta_scaler=meta_scaler,
                base_model_ids=base_ids,
                meta_feature_cols=meta_feature_cols,
                meta_model_type=meta_model_type,
                meta_c_value=meta_c_value,
                stacking_mode=stacking_mode,
                meta_features=meta_features,
                use_disagree=use_disagree,
                use_conf=use_conf,
                use_logit=use_logit,
                logit_eps=logit_eps,
                meta_calibrator=meta_calibrator,
                meta_calibration_method=meta_calibration_method,
            )

            # Prepare artifacts with file paths
            artifacts = {
                'dataset_path': csv_path,
                'base_ids': base_ids,
                'meta_model_type': meta_model_type,
                **artifact_paths
            }

            # Update run with results
            self.run_tracker.update_run(
                run_id=run_id,
                metrics=metrics,
                diagnostics=diagnostics,
                artifacts=artifacts,
                status='completed'
            )

            return {
                'run_id': run_id,
                'metrics': metrics,
                'diagnostics': diagnostics,
                'artifacts': artifacts
            }

        except Exception as e:
            self.run_tracker.update_run(
                run_id=run_id,
                status='failed',
                diagnostics={'error': str(e)}
            )
            raise

    def _load_base_models(self, base_ids: List[str]) -> List[Dict]:
        """
        Load all base models and their metadata.
        Supports both run_ids (legacy) and MongoDB config _ids (preferred).

        Args:
            base_ids: List of identifiers (run_ids or MongoDB config _ids)

        Returns:
            List of dicts with run_id, model, scaler, feature_names, config
        """
        base_models_info = []

        for base_id in base_ids:
            # Try to load as MongoDB config first (preferred)
            try:
                print(f"[STACKING] Loading base model {base_id} from MongoDB config...")

                try:
                    obj_id = ObjectId(base_id)
                except Exception as e:
                    print(f"[STACKING] Invalid ObjectId format for {base_id}: {e}")
                    raise ValueError(f"Base model {base_id} is not a valid MongoDB ObjectId")

                config = self._classifier_repo.find_one({'_id': obj_id})
                if config:
                    print(f"[STACKING] Found MongoDB config for {base_id}")
                    model, scaler, feature_names = self._load_model_from_config(config)
                    base_models_info.append({
                        'run_id': base_id,
                        'model': model,
                        'scaler': scaler,
                        'feature_names': feature_names,
                        'config': config
                    })
                    continue
                else:
                    print(f"[STACKING] No MongoDB config found for {base_id}")
            except Exception as e:
                print(f"[STACKING] Error loading MongoDB config for {base_id}: {e}")
                pass

            # Fall back to run_id loading (legacy support)
            run = self.run_tracker.get_run(base_id)
            if not run:
                raise ValueError(f"Base model {base_id} not found as run_id or config_id")

            # Load model artifacts via subclass hook
            try:
                model, scaler, feature_names = self._load_legacy_model(base_id)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Base model {base_id} does not have saved model artifacts. "
                    f"Only models trained after adding model persistence support can be used for stacking. "
                    f"Error: {e}"
                )

            base_models_info.append({
                'run_id': base_id,
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'config': run.get('config', {})
            })

        return base_models_info

    def _load_model_from_config(self, config: dict):
        """
        Load model from config document.
        Prioritizes saved artifacts, falls back to retraining.

        Args:
            config: Config document

        Returns:
            Tuple of (model, scaler, feature_names)
        """
        import pickle
        import json

        # Priority 1: Try to load saved artifacts (fast path)
        model_artifact_path = config.get('model_artifact_path')
        scaler_artifact_path = config.get('scaler_artifact_path')
        features_path = config.get('features_path')

        if model_artifact_path and scaler_artifact_path and features_path:
            if os.path.exists(model_artifact_path) and os.path.exists(scaler_artifact_path) and os.path.exists(features_path):
                try:
                    print(f"[STACKING] Loading saved artifacts for model...")

                    with open(model_artifact_path, 'rb') as f:
                        model = pickle.load(f)

                    with open(scaler_artifact_path, 'rb') as f:
                        scaler = pickle.load(f)

                    with open(features_path, 'r') as f:
                        feature_names = json.load(f)

                    print(f"[STACKING] Successfully loaded saved artifacts")
                    return model, scaler, feature_names

                except Exception as e:
                    print(f"[STACKING] Error loading saved artifacts: {e}")
                    print(f"[STACKING] Will fall back to retraining from data...")
            else:
                print(f"[STACKING] Expected artifacts not found:")
                print(f"[STACKING]   Model: {model_artifact_path} {'OK' if os.path.exists(model_artifact_path) else 'MISSING'}")
                print(f"[STACKING]   Scaler: {scaler_artifact_path} {'OK' if os.path.exists(scaler_artifact_path) else 'MISSING'}")
                print(f"[STACKING]   Features: {features_path} {'OK' if os.path.exists(features_path) else 'MISSING'}")
                print(f"[STACKING] Will fall back to retraining from data...")

        # Priority 2: Fallback to retraining from training data
        print(f"[STACKING] Retraining model from training data...")
        training_csv = config.get('training_csv')
        if not training_csv or not os.path.exists(training_csv):
            raise FileNotFoundError(
                f"Cannot load model: No saved artifacts found and training CSV not found: {training_csv}\n"
                f"Base models must be trained with model persistence support to be used in ensembles.\n"
                f"Please retrain the base models with the current system version."
            )

        # Load and prepare data
        df = pd.read_csv(training_csv)
        meta_cols = self.META_COLS + [self.TARGET_COL] + self.POINTS_COLS
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Coerce features to numeric
        X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')

        # Drop feature columns that are entirely non-numeric
        all_nan_cols = [c for c in X_df.columns if X_df[c].isna().all()]
        if all_nan_cols:
            print(f"[STACKING] Dropping non-numeric columns: {all_nan_cols}")
            X_df = X_df.drop(columns=all_nan_cols)
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]

        # Coerce y to numeric
        y_series = pd.to_numeric(df[self.TARGET_COL], errors='coerce')

        X = X_df.to_numpy(dtype=float)
        y = y_series.to_numpy()

        # Handle NaN values
        nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
        if nan_mask.sum() > 0:
            print(f"[STACKING] Dropping {nan_mask.sum()} rows with NaN values")
            df = df[~nan_mask].reset_index(drop=True)
            X = X[~nan_mask]
            y = y[~nan_mask]

        # Apply temporal split to prevent data leakage
        calibration_years = config.get('calibration_years', [])
        if not calibration_years:
            splits_cfg = config.get('splits', {})
            calibration_years = splits_cfg.get('calibration_years', [])

        if calibration_years:
            df_tmp = df.copy()
            df_tmp['SeasonStartYear'] = np.where(
                df_tmp['Month'] >= self._season_start_month,
                df_tmp['Year'],
                df_tmp['Year'] - 1
            )
            temporal_train_mask = (df_tmp['SeasonStartYear'] < min(calibration_years)).values
            n_before = len(X)
            X = X[temporal_train_mask]
            y = y[temporal_train_mask]
            print(f"[STACKING] Temporal split for retrain: {n_before} -> {len(X)} rows (pre-calibration only)")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create and train model
        model_type = config.get('model_type', 'LogisticRegression')
        best_c_value = config.get('best_c_value', 0.1)
        model = create_model_with_c(model_type, c_value=best_c_value)
        model.fit(X_scaled, y)

        return model, scaler, feature_cols

    def _save_ensemble_artifacts(
        self,
        run_id: str,
        meta_model,
        meta_scaler,
        base_model_ids: List[str],
        meta_feature_cols: List[str],
        meta_model_type: str,
        meta_c_value: float,
        stacking_mode: str,
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False,
        use_logit: bool = False,
        logit_eps: float = 1e-6,
        meta_calibrator=None,
        meta_calibration_method: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save ensemble model artifacts to disk for later loading.

        Returns:
            Dict with artifact file paths
        """
        import pickle
        import json

        ensembles_dir = self._get_ensembles_dir()
        os.makedirs(ensembles_dir, exist_ok=True)

        model_path = os.path.join(ensembles_dir, f'{run_id}_meta_model.pkl')
        scaler_path = os.path.join(ensembles_dir, f'{run_id}_meta_scaler.pkl')
        config_path = os.path.join(ensembles_dir, f'{run_id}_ensemble_config.json')
        calibrator_path = os.path.join(ensembles_dir, f'{run_id}_meta_calibrator.pkl') if meta_calibrator else None

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(meta_model, f)
            print(f"[STACKING] Saved meta-model: {model_path}")

            if meta_scaler is not None:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(meta_scaler, f)
                print(f"[STACKING] Saved meta-scaler: {scaler_path}")

            if meta_calibrator is not None:
                with open(calibrator_path, 'wb') as f:
                    pickle.dump(meta_calibrator, f)
                print(f"[STACKING] Saved meta-calibrator: {calibrator_path}")

            ensemble_config = {
                'run_id': run_id,
                'base_model_ids': base_model_ids,
                'meta_feature_cols': meta_feature_cols,
                'meta_model_type': meta_model_type,
                'meta_c_value': meta_c_value,
                'stacking_mode': stacking_mode,
                'meta_features': meta_features or [],
                'use_disagree': use_disagree,
                'use_conf': use_conf,
                'use_logit': use_logit,
                'logit_eps': logit_eps,
                'meta_calibration_method': meta_calibration_method,
            }
            with open(config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=2)
            print(f"[STACKING] Saved ensemble config: {config_path}")

            result = {
                'meta_model_path': model_path,
                'meta_scaler_path': scaler_path,
                'ensemble_config_path': config_path,
                'meta_model_type': meta_model_type,
                'meta_c_value': meta_c_value
            }
            if calibrator_path:
                result['meta_calibrator_path'] = calibrator_path
            return result

        except Exception as e:
            print(f"[STACKING] Error saving ensemble artifacts: {e}")
            return {}

    def _validate_base_models_compatible(self, base_models_info: List[Dict]):
        """
        Validate that all base models have compatible configurations.

        Checks same time-based calibration config. Feature sets can differ.
        """
        if len(base_models_info) < 2:
            return

        ref_config = base_models_info[0]['config']
        ref_splits = ref_config.get('splits', {})

        for i, model_info in enumerate(base_models_info[1:], 1):
            config = model_info['config']
            splits = config.get('splits', {})

            if splits.get('begin_year') != ref_splits.get('begin_year'):
                raise ValueError(
                    f"Base model {model_info['run_id']} has incompatible begin_year. "
                    f"Expected {ref_splits.get('begin_year')}, got {splits.get('begin_year')}"
                )

            if splits.get('calibration_years') != ref_splits.get('calibration_years'):
                raise ValueError(
                    f"Base model {model_info['run_id']} has incompatible calibration_years. "
                    f"Expected {ref_splits.get('calibration_years')}, got {splits.get('calibration_years')}"
                )

            if splits.get('evaluation_year') != ref_splits.get('evaluation_year'):
                raise ValueError(
                    f"Base model {model_info['run_id']} has incompatible evaluation_year. "
                    f"Expected {ref_splits.get('evaluation_year')}, got {splits.get('evaluation_year')}"
                )

    def _generate_stacking_data(
        self,
        base_models_info: List[Dict],
        df: pd.DataFrame,
        calibration_years: List[int],
        stacking_mode: str = 'naive',
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False,
        use_logit: bool = False,
        logit_eps: float = 1e-6
    ) -> pd.DataFrame:
        """
        Generate stacking training data using base model predictions.

        For each game in df, extracts features per base model, gets predictions,
        and builds the stacking DataFrame.

        Returns:
            DataFrame with columns [p_model1, ..., p_modelN, <derived>, <meta_feats>, TARGET_COL]
        """
        # Extract metadata and target columns
        meta_cols = self.META_COLS + ['SeasonStartYear']
        target_cols = [self.TARGET_COL]
        excluded_cols = meta_cols + target_cols
        # Exclude prediction columns EXCEPT those explicitly requested as meta-features
        pred_cols = list(self.PREDICTION_COLS)
        # Only exclude pred_margin if NOT explicitly requested as meta-feature
        if meta_features and 'pred_margin' in meta_features:
            pred_cols = [c for c in pred_cols if c != 'pred_margin']
        excluded_cols.extend([c for c in pred_cols if c in df.columns])

        # Get all available features in dataset
        available_features = set([c for c in df.columns if c not in excluded_cols])

        # Get predictions from each base model
        stacking_data = {}
        used_model_names = set()

        for i, model_info in enumerate(base_models_info):
            model = model_info['model']
            scaler = model_info['scaler']
            model_feature_names = model_info['feature_names']

            # Check which features are available
            missing_features = [f for f in model_feature_names if f not in available_features]
            available_model_features = [f for f in model_feature_names if f in available_features]

            if len(available_model_features) == 0:
                raise ValueError(
                    f"Base model {model_info['run_id']} requires features that are not in the dataset. "
                    f"Missing all {len(model_feature_names)} features. "
                    f"Model was trained with: {model_feature_names[:5]}... "
                )

            if len(missing_features) > 0:
                print(
                    f"Warning: Base model {model_info['run_id']} is missing {len(missing_features)}/{len(model_feature_names)} features "
                    f"in dataset: {sorted(missing_features)[:10]}... "
                    f"Missing features will be set to 0.0 for prediction."
                )

            # Build feature matrix
            X = np.zeros((len(df), len(model_feature_names)))
            for idx, feature_name in enumerate(model_feature_names):
                if feature_name in available_features:
                    X[:, idx] = df[feature_name].values
                else:
                    X[:, idx] = 0.0

            # Handle NaN and Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            if scaler is not None:
                try:
                    X_scaled = scaler.transform(X)
                except Exception as e:
                    raise ValueError(
                        f"Cannot scale features for model {model_info['run_id']}. "
                        f"Scaler expects {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'unknown'} features "
                        f"but got {X.shape[1]}. "
                        f"Error: {e}"
                    )
            else:
                X_scaled = X

            # Get predictions
            y_proba = model.predict_proba(X_scaled)
            n_classes = y_proba.shape[1]

            # Store predictions with model identifier
            config = model_info.get('config', {})
            model_name = config.get('name')
            if model_name:
                model_id_short = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
            else:
                model_id = model_info['run_id']
                model_id_short = model_id[:8] if len(model_id) > 8 else model_id

            # Handle name collisions
            base_name = model_id_short
            counter = 1
            while model_id_short in used_model_names:
                model_id_short = f"{base_name}_{counter}"
                counter += 1
            used_model_names.add(model_id_short)

            if n_classes == 2:
                # Binary: single P(home) column
                stacking_data[f'p_{model_id_short}'] = y_proba[:, 1]
            else:
                # Ternary: store P(home) and P(draw), P(away) is redundant
                stacking_data[f'p_home_{model_id_short}'] = y_proba[:, 2]  # home = class 2
                stacking_data[f'p_draw_{model_id_short}'] = y_proba[:, 1]  # draw = class 1

        # p_* column names (used for informed derived features and for logit transform)
        pred_col_names = [col for col in stacking_data.keys() if col.startswith('p_')]

        # For informed stacking, add derived features
        if stacking_mode == 'informed':
            n_models = len(pred_col_names)

            if use_disagree:
                for i in range(n_models):
                    for j in range(i + 1, n_models):
                        col_i = pred_col_names[i]
                        col_j = pred_col_names[j]
                        id_i = col_i.replace('p_', '')
                        id_j = col_j.replace('p_', '')
                        disagree_name = f'disagree_{id_i}_{id_j}'
                        stacking_data[disagree_name] = np.abs(stacking_data[col_i] - stacking_data[col_j])

            if use_conf:
                if n_classes == 2:
                    # Binary: distance from 0.5
                    for col in pred_col_names:
                        model_id = col.replace('p_', '')
                        conf_name = f'conf_{model_id}'
                        stacking_data[conf_name] = np.abs(stacking_data[col] - 0.5)
                else:
                    # Ternary: max probability - 1/3 for each model
                    for mid in used_model_names:
                        p_home = stacking_data[f'p_home_{mid}']
                        p_draw = stacking_data[f'p_draw_{mid}']
                        p_away = 1.0 - p_home - p_draw
                        max_p = np.maximum(p_home, np.maximum(p_draw, p_away))
                        stacking_data[f'conf_{mid}'] = max_p - 1.0 / 3

            # Add user-provided features
            if meta_features:
                print(f"[STACKING] Adding user meta-features: {meta_features}")
                avail = set(df.columns)
                excluded_set = set(self.META_COLS + ['SeasonStartYear'] + [self.TARGET_COL])
                pred_cols_exclude = list(self.PREDICTION_COLS)
                if 'pred_margin' in meta_features:
                    pred_cols_exclude = [c for c in pred_cols_exclude if c != 'pred_margin']
                excluded_set.update(c for c in pred_cols_exclude if c in df.columns)

                for feat_name in meta_features:
                    if feat_name in excluded_set:
                        print(f"Warning: Meta-feature '{feat_name}' is in excluded columns. Skipping.")
                        continue
                    if feat_name in avail:
                        stacking_data[feat_name] = df[feat_name].values
                    else:
                        print(f"Warning: Meta-feature '{feat_name}' not found in dataset. Skipping.")

        # Apply logit transform to p_* columns (after conf/disagree computed on raw scale)
        if use_logit:
            logit_pred_cols = [c for c in pred_col_names]
            for col in logit_pred_cols:
                p = stacking_data[col]
                p_clipped = np.clip(p, logit_eps, 1 - logit_eps)
                stacking_data[col] = np.log(p_clipped / (1 - p_clipped))

        # Add true labels
        stacking_data[self.TARGET_COL] = df[self.TARGET_COL].values

        # Create DataFrame
        stacking_df = pd.DataFrame(stacking_data)

        # Handle NaN values
        feature_cols = [c for c in stacking_df.columns if c != self.TARGET_COL]
        stacking_df[feature_cols] = stacking_df[feature_cols].fillna(0.0)

        # Drop rows where target is NaN
        n_before = len(stacking_df)
        stacking_df = stacking_df.dropna(subset=[self.TARGET_COL])
        n_after = len(stacking_df)

        if n_after == 0:
            target_null = df[self.TARGET_COL].isna().sum() if self.TARGET_COL in df.columns else 'column missing'
            raise ValueError(
                f"Stacking data has 0 rows after dropping NaN targets "
                f"(had {n_before} before dropna, {len(df)} input rows, "
                f"target NaN count: {target_null}, "
                f"features: {feature_cols})"
            )

        return stacking_df

    def _train_meta_model(
        self,
        stacking_df: pd.DataFrame,
        meta_model_type: str = 'LogisticRegression',
        meta_c_value: float = 0.1
    ):
        """
        Train meta-model on stacking training data.

        Returns:
            Tuple of (trained meta-model, fitted StandardScaler)
        """
        feature_cols = [c for c in stacking_df.columns if c != self.TARGET_COL]
        X_meta = stacking_df[feature_cols].values
        y_meta = stacking_df[self.TARGET_COL].values

        if len(X_meta) == 0:
            raise ValueError(
                f"Meta-model training data is empty (0 rows, {len(feature_cols)} features: {feature_cols}). "
                f"Check that calibration period has games with valid target values."
            )

        print(f"[STACKING] Training {meta_model_type} meta-model on {len(X_meta)} samples, {len(feature_cols)} features")

        # Handle NaN/Inf
        X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        meta_scaler = StandardScaler()
        X_meta = meta_scaler.fit_transform(X_meta)

        # Create and train meta-model
        meta_model = create_model_with_c(meta_model_type, c_value=meta_c_value)
        meta_model.fit(X_meta, y_meta)

        return meta_model, meta_scaler

    def _evaluate_stacked_model(
        self,
        meta_model,
        meta_scaler,
        base_models_info: List[Dict],
        df: pd.DataFrame,
        evaluation_year: int,
        calibration_years: List[int],
        begin_year: int,
        n_train_samples: int = 0,
        n_cal_samples: int = 0,
        stacking_mode: str = 'naive',
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False,
        use_logit: bool = False,
        logit_eps: float = 1e-6,
        meta_calibrator=None,
        meta_calibration_method: Optional[str] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate stacked model on evaluation period.

        Returns:
            Tuple of (metrics_dict, diagnostics_dict)
        """
        # Generate stacking data for evaluation period
        stacking_df = self._generate_stacking_data(
            base_models_info=base_models_info,
            df=df,
            calibration_years=[],
            stacking_mode=stacking_mode,
            meta_features=meta_features,
            use_disagree=use_disagree,
            use_conf=use_conf,
            use_logit=use_logit,
            logit_eps=logit_eps
        )

        # Extract features and labels
        feature_cols = [c for c in stacking_df.columns if c != self.TARGET_COL]
        X_meta = stacking_df[feature_cols].values
        y_true = stacking_df[self.TARGET_COL].values

        # Handle NaN/Inf
        X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply same scaler used during training
        if meta_scaler is not None:
            X_meta = meta_scaler.transform(X_meta)

        # Get meta-model predictions
        y_proba_meta = meta_model.predict_proba(X_meta)

        # Apply meta-calibrator if present
        if meta_calibrator is not None:
            y_proba_meta = self._apply_meta_calibrator(y_proba_meta, meta_calibrator)

        y_pred_meta = _classify(y_proba_meta)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_meta) * 100
        log_loss_val = log_loss(y_true, y_proba_meta)
        brier = _multiclass_brier_score(y_true, y_proba_meta)

        try:
            if y_proba_meta.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba_meta[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba_meta, multi_class='ovr')
        except ValueError:
            auc = 0.0

        metrics = {
            'accuracy_mean': float(accuracy),
            'accuracy_std': 0.0,
            'log_loss_mean': float(log_loss_val),
            'log_loss_std': 0.0,
            'brier_mean': float(brier),
            'brier_std': 0.0,
            'auc_mean': float(auc),
            'auc_std': 0.0,
            'n_folds': 1,
            'split_type': 'time_based_calibration',
            'evaluation_year': evaluation_year
        }

        # Calculate meta-model feature importances
        meta_feature_importances = {}
        if hasattr(meta_model, 'coef_'):
            if meta_model.coef_.ndim == 1 or meta_model.coef_.shape[0] == 1:
                coef = np.abs(meta_model.coef_.ravel())
            else:
                coef = np.mean(np.abs(meta_model.coef_), axis=0)
            meta_feature_importances = dict(zip(feature_cols, coef.tolist()))
            meta_feature_importances = dict(sorted(
                meta_feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            ))

        # Re-evaluate base models on evaluation set
        base_models_summary = []
        for model_info in base_models_info:
            base_run_id = model_info['run_id']
            base_run = self.run_tracker.get_run(base_run_id)
            model = model_info['model']
            scaler = model_info['scaler']
            model_feature_names = model_info['feature_names']

            base_model_metrics = {}
            try:
                meta_cols_eval = self.META_COLS + ['SeasonStartYear']
                target_cols_eval = [self.TARGET_COL]
                excluded_cols_eval = meta_cols_eval + target_cols_eval
                pred_cols_eval = list(self.PREDICTION_COLS)
                excluded_cols_eval.extend([c for c in pred_cols_eval if c in df.columns])

                X_base = np.zeros((len(df), len(model_feature_names)))
                for idx_f, feat_name in enumerate(model_feature_names):
                    if feat_name in df.columns:
                        X_base[:, idx_f] = df[feat_name].values
                    else:
                        X_base[:, idx_f] = 0.0

                X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)

                if scaler:
                    X_base_scaled = scaler.transform(X_base)
                else:
                    X_base_scaled = X_base

                y_proba_base = model.predict_proba(X_base_scaled)
                y_pred_base = _classify(y_proba_base)
                y_true_base = df[self.TARGET_COL].values

                base_accuracy = accuracy_score(y_true_base, y_pred_base) * 100
                base_log_loss = log_loss(y_true_base, y_proba_base)
                base_brier = _multiclass_brier_score(y_true_base, y_proba_base)
                try:
                    if y_proba_base.shape[1] == 2:
                        base_auc = roc_auc_score(y_true_base, y_proba_base[:, 1])
                    else:
                        base_auc = roc_auc_score(y_true_base, y_proba_base, multi_class='ovr')
                except ValueError:
                    base_auc = 0.0

                base_model_metrics = {
                    'accuracy_mean': float(base_accuracy),
                    'log_loss_mean': float(base_log_loss),
                    'brier_mean': float(base_brier),
                    'auc_mean': float(base_auc),
                    'n_samples_evaluation': len(y_true_base)
                }
            except Exception as e:
                print(f"Warning: Could not re-evaluate base model {base_run_id}: {e}")
                if base_run:
                    base_model_metrics = base_run.get('metrics', {})

            if base_run:
                base_config = base_run.get('config', {})
                base_splits = base_config.get('splits', {})
                base_models_summary.append({
                    'run_id': base_run_id,
                    'metrics': base_model_metrics,
                    'config': base_config,
                    'model_type': base_run.get('model_type', 'Unknown'),
                    'feature_names': model_info.get('feature_names', []),
                    'begin_year': base_splits.get('begin_year'),
                    'calibration_years': base_splits.get('calibration_years', []),
                    'evaluation_year': base_splits.get('evaluation_year'),
                    'n_samples_train': base_run.get('metrics', {}).get('train_set_size'),
                    'n_samples_calibration': base_run.get('metrics', {}).get('calibrate_set_size'),
                    'n_samples_evaluation': base_model_metrics.get('n_samples_evaluation')
                })

        # Extract derived and user features for diagnostics
        derived_features_used = []
        meta_features_used = []
        if stacking_mode == 'informed':
            all_cols = set(stacking_df.columns)
            pred_cols_set = {col for col in all_cols if col.startswith('p_') or col.startswith('p_home_') or col.startswith('p_draw_')}
            target_cols_set = {self.TARGET_COL}

            derived_features_used = [col for col in all_cols if col.startswith('disagree_') or col.startswith('conf_')]

            remaining_cols = all_cols - pred_cols_set - target_cols_set - set(derived_features_used)
            if meta_features:
                meta_features_used = [col for col in remaining_cols if col in meta_features]

        diagnostics = {
            'meta_model_type': 'LogisticRegression',
            'meta_feature_importances': meta_feature_importances,
            'n_base_models': len(base_models_info),
            'base_run_ids': [info['run_id'] for info in base_models_info],
            'base_models_summary': base_models_summary,
            'n_samples_train': n_train_samples,
            'n_samples_calibration': n_cal_samples,
            'n_samples_evaluation': len(y_true),
            'evaluation_year': evaluation_year,
            'calibration_years': calibration_years,
            'begin_year': begin_year,
            'split_type': 'time_based_calibration',
            'stacking_mode': stacking_mode,
            'use_disagree': use_disagree,
            'use_conf': use_conf,
            'use_logit': use_logit,
            'logit_eps': logit_eps,
            'meta_features_used': meta_features_used,
            'derived_features_used': derived_features_used,
            'meta_calibration_method': meta_calibration_method,
            'meta_calibrated': meta_calibrator is not None,
        }

        return metrics, diagnostics

    def _fit_meta_calibrator(
        self,
        meta_model,
        meta_scaler,
        stacking_df_cal: pd.DataFrame,
        method: str,
    ):
        """
        Fit a calibrator on held-out meta-model predictions.

        Args:
            meta_model: Trained meta-model
            meta_scaler: Fitted scaler for meta features
            stacking_df_cal: Stacking data for calibration period
            method: 'isotonic' or 'sigmoid'

        Returns:
            Fitted calibrator object (IsotonicRegression, dict with 'sigmoid' LR, or dict with 'temperature')
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        feature_cols = [c for c in stacking_df_cal.columns if c != self.TARGET_COL]
        X_cal = stacking_df_cal[feature_cols].values
        y_cal = stacking_df_cal[self.TARGET_COL].values

        X_cal = np.nan_to_num(X_cal, nan=0.0, posinf=0.0, neginf=0.0)
        if meta_scaler is not None:
            X_cal = meta_scaler.transform(X_cal)

        y_proba_raw = meta_model.predict_proba(X_cal)
        n_classes = y_proba_raw.shape[1]

        if n_classes == 2:
            # Binary calibration
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_proba_raw[:, 1], y_cal)
                print(f"[STACKING] Fitted isotonic meta-calibrator on {len(y_cal)} samples")
                return calibrator
            elif method == 'sigmoid':
                sigmoid_lr = LR()
                sigmoid_lr.fit(y_proba_raw[:, 1].reshape(-1, 1), y_cal)
                print(f"[STACKING] Fitted sigmoid meta-calibrator on {len(y_cal)} samples")
                return {'type': 'sigmoid', 'model': sigmoid_lr}
            else:
                raise ValueError(f"Unknown meta_calibration_method: {method}")
        else:
            # Multi-class: temperature scaling
            from sklearn.metrics import log_loss as sk_log_loss
            from scipy.optimize import minimize_scalar

            def _temperature_scale(proba, T):
                logits = np.log(np.clip(proba, 1e-15, 1.0))
                scaled = logits / T
                exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                return exp_s / exp_s.sum(axis=1, keepdims=True)

            def _nll(T):
                return sk_log_loss(y_cal, _temperature_scale(y_proba_raw, T))

            result = minimize_scalar(_nll, bounds=(0.1, 10.0), method='bounded')
            T_opt = result.x
            print(f"[STACKING] Fitted temperature scaling meta-calibrator: T={T_opt:.3f}")
            return {'type': 'temperature', 'T': T_opt}

    @staticmethod
    def _apply_meta_calibrator(y_proba_raw: np.ndarray, calibrator) -> np.ndarray:
        """
        Apply a fitted calibrator to raw meta-model probabilities.

        Args:
            y_proba_raw: Raw probabilities from meta-model, shape (n_samples, n_classes)
            calibrator: Fitted calibrator (IsotonicRegression, sigmoid dict, or temperature dict)

        Returns:
            Calibrated probabilities, same shape as input
        """
        from sklearn.isotonic import IsotonicRegression

        if isinstance(calibrator, IsotonicRegression):
            # Binary isotonic
            cal_1 = calibrator.predict(y_proba_raw[:, 1])
            cal_1 = np.clip(cal_1, 0.0, 1.0)
            return np.column_stack([1 - cal_1, cal_1])
        elif isinstance(calibrator, dict):
            cal_type = calibrator.get('type')
            if cal_type == 'sigmoid':
                lr_model = calibrator['model']
                cal_1 = lr_model.predict_proba(y_proba_raw[:, 1].reshape(-1, 1))[:, 1]
                cal_1 = np.clip(cal_1, 0.0, 1.0)
                return np.column_stack([1 - cal_1, cal_1])
            elif cal_type == 'temperature':
                T = calibrator['T']
                logits = np.log(np.clip(y_proba_raw, 1e-15, 1.0))
                scaled = logits / T
                exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                return exp_s / exp_s.sum(axis=1, keepdims=True)
            else:
                raise ValueError(f"Unknown calibrator type: {cal_type}")
        else:
            raise ValueError(f"Unknown calibrator object: {type(calibrator)}")
