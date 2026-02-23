"""
Base Ensemble Predictor with shared feature generation.

Provides the sport-agnostic ensemble prediction pipeline:
1. Load base model configs and meta-model from artifacts
2. Generate features (via subclass hook)
3. Get predictions from each base model
4. Combine with meta-model
5. Format result (via subclass hook)

Subclasses implement sport-specific feature generation and result formatting.
"""

import os
import re
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from bson import ObjectId


class BaseEnsemblePredictor(ABC):
    """
    Sport-agnostic ensemble prediction using shared feature generation and cached models.

    Subclasses must implement:
    - _get_model_config_collection_name() -> str
    - _create_feature_generator() -> feature generator instance
    - _generate_game_features(**kwargs) -> dict of feature values
    - _format_prediction_result(...) -> dict with prediction result
    """

    # --- Class attributes (override in subclass for different column conventions) ---
    # Relative to sport app repo root; resolved to absolute in _get_ensembles_dir()
    ENSEMBLE_DIR = 'cli/models/ensembles'

    def _resolve_path(self, path: str) -> str:
        """Resolve a potentially relative path against the sport app repo root."""
        if not path or os.path.isabs(path):
            return path
        if self.league and hasattr(self.league, '_repo_root'):
            resolved = os.path.join(self.league._repo_root, path)
            if os.path.exists(resolved):
                return resolved
        return path

    def _get_ensembles_dir(self) -> str:
        """Resolve ENSEMBLE_DIR to an absolute path using league repo root."""
        if os.path.isabs(self.ENSEMBLE_DIR):
            return self.ENSEMBLE_DIR
        if self.league and hasattr(self.league, '_repo_root'):
            return os.path.join(self.league._repo_root, self.ENSEMBLE_DIR)
        return os.path.abspath(self.ENSEMBLE_DIR)

    def __init__(self, db, ensemble_config: Dict, league=None):
        """
        Initialize the ensemble predictor.

        Args:
            db: Database connection
            ensemble_config: Ensemble configuration with:
                - ensemble_run_id: ID for loading meta-model artifacts
                - ensemble_models: List of base model config IDs
                - ensemble_meta_features: List of meta-feature names
                - meta_model_path: Path to meta-model pickle (optional)
                - ensemble_config_path: Path to ensemble config JSON (optional)
            league: Optional league config for collection routing
        """
        self.db = db
        self.ensemble_config = ensemble_config
        self.league = league

        # Extract configuration
        self.ensemble_run_id = ensemble_config.get('ensemble_run_id')
        self.base_model_ids = ensemble_config.get('ensemble_models', [])
        self.meta_feature_names = ensemble_config.get('ensemble_meta_features', [])

        if not self.ensemble_run_id:
            raise ValueError("Ensemble config missing ensemble_run_id")
        if not self.base_model_ids or len(self.base_model_ids) < 2:
            raise ValueError("Ensemble config must have at least 2 base models")

        # Load base model configs and collect their features
        self.base_model_configs: List[Dict] = []
        self.base_model_features: List[List[str]] = []
        self._load_base_model_configs()

        # Collect all unique features needed (base model features + extra meta-features)
        self.all_features = self._collect_unique_features(
            self.base_model_features + ([self.meta_feature_names] if self.meta_feature_names else [])
        )

        # Initialize feature generator via subclass hook
        self.feature_generator = self._create_feature_generator()

        # Load meta-model, scaler, calibrator, and config
        self.meta_model = None
        self.meta_scaler = None
        self.meta_calibrator = None
        self.meta_config = None
        self._load_meta_model()

        # Prediction context (set via set_prediction_context)
        self._prediction_context = None

    @abstractmethod
    def _get_model_config_collection_name(self) -> str:
        """Return the MongoDB collection name for model configs."""
        ...

    @abstractmethod
    def _create_feature_generator(self):
        """Create and return the sport-specific feature generator."""
        ...

    @abstractmethod
    def _generate_game_features(self, **kwargs) -> Dict:
        """Generate game features using the feature generator. Returns dict of feature_name -> value."""
        ...

    @abstractmethod
    def _format_prediction_result(self, ensemble_home_prob: float, home_team: str, away_team: str,
                                   base_model_breakdowns: List[Dict], meta_info: Dict,
                                   all_feature_dict: Dict,
                                   ensemble_draw_prob: float = None,
                                   ensemble_away_prob: float = None) -> Dict:
        """Format the final prediction result dict. Sport-specific."""
        ...

    @staticmethod
    def _collect_unique_features(feature_lists: List[List[str]]) -> List[str]:
        """Collect unique features across multiple feature lists, preserving order."""
        seen = set()
        unique = []
        for flist in feature_lists:
            for f in flist:
                if f not in seen:
                    seen.add(f)
                    unique.append(f)
        return unique

    def set_prediction_context(self, context) -> None:
        """
        Inject preloaded prediction context to avoid per-feature DB calls.

        Propagates the context to the internal feature generator.

        Args:
            context: PredictionContext instance with preloaded data
        """
        self._prediction_context = context
        if self.feature_generator:
            self.feature_generator.set_prediction_context(context)

    def _load_base_model_configs(self):
        """Load base model configurations from the database."""
        from sportscore.models.base_artifact_loader import BaseArtifactLoader

        config_collection = self._get_model_config_collection_name()

        for base_id in self.base_model_ids:
            base_id_str = str(base_id)
            try:
                config = self.db[config_collection].find_one({'_id': ObjectId(base_id_str)})
                if not config:
                    raise ValueError(f"Base model config not found: {base_id_str}")

                # Resolve relative artifact paths against sport app repo root
                for path_key in ('model_artifact_path', 'scaler_artifact_path', 'features_path'):
                    if config.get(path_key):
                        config[path_key] = self._resolve_path(config[path_key])

                self.base_model_configs.append(config)

                # Get feature names from config or load from artifacts
                feature_names = config.get('feature_names')
                if not feature_names:
                    features_path = config.get('features_path')
                    if features_path and os.path.exists(features_path):
                        with open(features_path, 'r') as f:
                            feature_names = json.load(f)
                    else:
                        # Try to load via ArtifactLoader (will also cache the model)
                        _, _, feature_names = BaseArtifactLoader.create_model(config, use_artifacts=True)

                self.base_model_features.append(feature_names or [])

            except Exception as e:
                raise ValueError(f"Failed to load base model {base_id_str}: {e}")

    def _load_meta_model(self):
        """Load meta-model, scaler, and configuration from artifacts."""
        meta_model_path = self._resolve_path(self.ensemble_config.get('meta_model_path') or '')
        meta_scaler_path = self._resolve_path(self.ensemble_config.get('meta_scaler_path') or '')
        ensemble_cfg_path = self._resolve_path(self.ensemble_config.get('ensemble_config_path') or '')
        # Treat empty strings as None after resolution
        meta_model_path = meta_model_path or None
        meta_scaler_path = meta_scaler_path or None
        ensemble_cfg_path = ensemble_cfg_path or None

        # Derive missing paths from a known stored path (sibling files share the same
        # directory and run_id prefix) or fall back to ENSEMBLE_DIR + run_id.
        # We avoid __file__-based derivation because sportscore's __file__ resolves to
        # the sportscore package, not the sport app that owns the artifacts.
        if meta_model_path:
            ensembles_dir = os.path.dirname(meta_model_path)
        elif ensemble_cfg_path:
            ensembles_dir = os.path.dirname(ensemble_cfg_path)
        else:
            ensembles_dir = self._get_ensembles_dir()
        meta_model_path = meta_model_path or os.path.join(ensembles_dir, f'{self.ensemble_run_id}_meta_model.pkl')
        meta_scaler_path = meta_scaler_path or os.path.join(ensembles_dir, f'{self.ensemble_run_id}_meta_scaler.pkl')
        ensemble_cfg_path = ensemble_cfg_path or os.path.join(ensembles_dir, f'{self.ensemble_run_id}_ensemble_config.json')

        if not os.path.exists(meta_model_path):
            raise ValueError(f"Meta-model not found: {meta_model_path}")
        if not os.path.exists(ensemble_cfg_path):
            raise ValueError(f"Ensemble config not found: {ensemble_cfg_path}")

        with open(meta_model_path, 'rb') as f:
            self.meta_model = pickle.load(f)

        # Load scaler (may not exist for older ensembles trained without scaling)
        if meta_scaler_path and os.path.exists(meta_scaler_path):
            with open(meta_scaler_path, 'rb') as f:
                self.meta_scaler = pickle.load(f)

        with open(ensemble_cfg_path, 'r') as f:
            self.meta_config = json.load(f)

        # Load meta-calibrator if present
        meta_cal_method = self.meta_config.get('meta_calibration_method')
        if meta_cal_method:
            # Try explicit path from MongoDB doc first, then derive from sibling file
            calibrator_path = self.ensemble_config.get('meta_calibrator_path')
            if not calibrator_path or not os.path.exists(calibrator_path):
                calibrator_path = os.path.join(ensembles_dir, f'{self.ensemble_run_id}_meta_calibrator.pkl')

            if os.path.exists(calibrator_path):
                with open(calibrator_path, 'rb') as f:
                    self.meta_calibrator = pickle.load(f)

    def predict(self, home_team: str, away_team: str, season: str, game_date: str, **kwargs) -> Dict:
        """
        Make an ensemble prediction for a game.

        Steps:
        1. Generate features (via subclass hook)
        2. Get predictions from each base model
        3. Build meta-features and predict with meta-model
        4. Format result (via subclass hook)

        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season string (e.g., '2024-2025')
            game_date: Game date string (YYYY-MM-DD)
            **kwargs: Sport-specific arguments passed to _generate_game_features

        Returns:
            Dict with prediction results
        """
        from sportscore.models.base_artifact_loader import BaseArtifactLoader

        # STEP 1: Generate features via subclass hook
        all_feature_dict = self._generate_game_features(
            home_team=home_team,
            away_team=away_team,
            season=season,
            game_date=game_date,
            **kwargs
        )

        # STEP 2: Get predictions from each base model
        # Prediction uses artifact config only (use_logit/logit_eps); no Mongo/request override.
        use_logit = bool(self.meta_config.get('use_logit', False))
        logit_eps = float(self.meta_config.get('logit_eps', 1e-6))

        base_home_probs: Dict[str, float] = {}
        base_model_breakdowns: List[Dict] = []

        used_model_names = set()
        for i, (config, feature_names) in enumerate(zip(self.base_model_configs, self.base_model_features)):
            base_id_str = str(self.base_model_ids[i])

            # Use config 'name' field if available
            model_name = config.get('name')
            if model_name:
                base_id_short = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
            else:
                base_id_short = base_id_str[:8]

            # Handle name collisions
            original_name = base_id_short
            counter = 1
            while base_id_short in used_model_names:
                base_id_short = f"{original_name}_{counter}"
                counter += 1
            used_model_names.add(base_id_short)

            # Get cached sklearn model
            model, scaler, _ = BaseArtifactLoader.create_model(config, use_artifacts=True)

            # Build feature vector in correct order
            additional_features = kwargs.get('additional_features')
            model_additional = None
            if additional_features:
                model_additional = {k: v for k, v in additional_features.items() if k in feature_names}

            feature_values = []
            for fname in feature_names:
                if model_additional and fname in model_additional:
                    feature_values.append(model_additional[fname])
                else:
                    feature_values.append(all_feature_dict.get(fname, 0.0))

            # Scale and predict
            X = np.array(feature_values).reshape(1, -1)
            if scaler:
                X = scaler.transform(X)

            proba = model.predict_proba(X)[0]
            n_classes = len(proba)

            if n_classes == 2:
                # Binary: P(home) = proba[1]
                home_win_prob = float(proba[1])
                if use_logit:
                    home_win_prob = max(logit_eps, min(home_win_prob, 1 - logit_eps))
                else:
                    home_win_prob = max(0.01, min(home_win_prob, 0.99))

                base_home_probs[f"p_{base_id_short}"] = home_win_prob

                breakdown = {
                    'config_id': base_id_str,
                    'config_id_short': base_id_short,
                    'name': config.get('name') or f'Base Model {base_id_short}',
                    'model_type': config.get('model_type') or 'Unknown',
                    'home_win_prob_pct': round(home_win_prob * 100, 1),
                    'features_dict': {fname: all_feature_dict.get(fname, 0.0) for fname in feature_names}
                }
            else:
                # Ternary: class 0=away, 1=draw, 2=home
                home_prob = float(proba[2])
                draw_prob = float(proba[1])
                away_prob = float(proba[0])

                base_home_probs[f"p_home_{base_id_short}"] = home_prob
                base_home_probs[f"p_draw_{base_id_short}"] = draw_prob

                breakdown = {
                    'config_id': base_id_str,
                    'config_id_short': base_id_short,
                    'name': config.get('name') or f'Base Model {base_id_short}',
                    'model_type': config.get('model_type') or 'Unknown',
                    'home_win_prob_pct': round(home_prob * 100, 1),
                    'draw_prob_pct': round(draw_prob * 100, 1),
                    'away_win_prob_pct': round(away_prob * 100, 1),
                    'features_dict': {fname: all_feature_dict.get(fname, 0.0) for fname in feature_names}
                }

            base_model_breakdowns.append(breakdown)

        # STEP 3: Build meta-features and predict with meta-model
        meta_feature_cols = self.meta_config.get('meta_feature_cols', [])
        stacking_mode = self.meta_config.get('stacking_mode', 'naive')
        use_disagree = bool(self.meta_config.get('use_disagree', False))
        use_conf = bool(self.meta_config.get('use_conf', False))

        # Reconstruct meta feature columns if missing (naive stacking)
        if not meta_feature_cols:
            reconstructed_cols = []
            seen_names = set()
            for bid, cfg in zip(self.base_model_ids, self.base_model_configs):
                model_name = cfg.get('name')
                if model_name:
                    col_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
                else:
                    col_name = str(bid)[:8]
                original = col_name
                counter = 1
                while col_name in seen_names:
                    col_name = f"{original}_{counter}"
                    counter += 1
                seen_names.add(col_name)
                reconstructed_cols.append(f"p_{col_name}")
            meta_feature_cols = reconstructed_cols

        meta_values: Dict[str, float] = {}
        meta_values.update(base_home_probs)

        # Derived meta-features for informed stacking
        if stacking_mode == 'informed':
            pred_cols = sorted([c for c in meta_values.keys() if c.startswith('p_')])

            if use_disagree:
                for i in range(len(pred_cols)):
                    for j in range(i + 1, len(pred_cols)):
                        id_i = pred_cols[i].replace('p_', '')
                        id_j = pred_cols[j].replace('p_', '')
                        meta_values[f'disagree_{id_i}_{id_j}'] = abs(meta_values[pred_cols[i]] - meta_values[pred_cols[j]])

            if use_conf:
                # Detect binary vs ternary from the column naming pattern
                has_ternary_cols = any(c.startswith('p_home_') for c in pred_cols)
                if not has_ternary_cols:
                    # Binary: distance from 0.5
                    for col in pred_cols:
                        mid = col.replace('p_', '')
                        meta_values[f'conf_{mid}'] = abs(meta_values[col] - 0.5)
                else:
                    # Ternary: max probability - 1/3 for each model
                    model_ids = set()
                    for col in pred_cols:
                        if col.startswith('p_home_'):
                            model_ids.add(col.replace('p_home_', ''))
                    for mid in model_ids:
                        p_home = meta_values.get(f'p_home_{mid}', 0.0)
                        p_draw = meta_values.get(f'p_draw_{mid}', 0.0)
                        p_away = 1.0 - p_home - p_draw
                        meta_values[f'conf_{mid}'] = max(p_home, p_draw, p_away) - 1.0 / 3

            # Add extra meta-features from the game features if needed
            extra_cols = [
                c for c in meta_feature_cols
                if not (c.startswith('p_') or c.startswith('disagree_') or c.startswith('conf_'))
            ]
            for col in extra_cols:
                if col in all_feature_dict:
                    meta_values[col] = all_feature_dict[col]

        # Apply logit transform to p_* columns (same list as meta_feature_cols for consistency)
        if use_logit:
            logit_pred_cols = [c for c in meta_feature_cols if c.startswith('p_')]
            for col in logit_pred_cols:
                p = meta_values.get(col, 0.0)
                p_clipped = max(logit_eps, min(p, 1 - logit_eps))
                meta_values[col] = np.log(p_clipped / (1 - p_clipped))

        # Build meta-feature vector in correct order
        meta_X = np.array([meta_values.get(col, 0.0) for col in meta_feature_cols]).reshape(1, -1)

        # Apply scaler if available
        meta_normalized_values = {}
        if self.meta_scaler is not None:
            meta_X = self.meta_scaler.transform(meta_X)
            for idx, col in enumerate(meta_feature_cols):
                raw = meta_values.get(col, 0.0)
                norm = float(meta_X[0, idx])
                if abs(raw - norm) > 1e-6:
                    meta_normalized_values[col] = norm

        # Meta-model prediction
        meta_proba = self.meta_model.predict_proba(meta_X)[0]

        # Apply meta-calibrator if present
        if self.meta_calibrator is not None:
            from sportscore.training.base_stacking import BaseStackingTrainer
            meta_proba_2d = meta_proba.reshape(1, -1)
            meta_proba_2d = BaseStackingTrainer._apply_meta_calibrator(meta_proba_2d, self.meta_calibrator)
            meta_proba = meta_proba_2d[0]

        n_meta_classes = len(meta_proba)

        if n_meta_classes == 2:
            ensemble_home_prob = float(meta_proba[1])
            ensemble_home_prob = max(0.01, min(ensemble_home_prob, 0.99))
            ensemble_draw_prob = None
            ensemble_away_prob = None
        else:
            ensemble_home_prob = float(meta_proba[2])   # home = class 2
            ensemble_draw_prob = float(meta_proba[1])   # draw = class 1
            ensemble_away_prob = float(meta_proba[0])   # away = class 0

        # Build meta_info for subclass formatting
        meta_info = {
            'meta_values': meta_values,
            'meta_feature_cols': meta_feature_cols,
            'meta_normalized_values': meta_normalized_values,
            'stacking_mode': stacking_mode,
            'use_disagree': use_disagree,
            'use_conf': use_conf,
            'use_logit': use_logit,
            'logit_eps': logit_eps,
        }

        # STEP 4: Format result via subclass hook
        return self._format_prediction_result(
            ensemble_home_prob=ensemble_home_prob,
            home_team=home_team,
            away_team=away_team,
            base_model_breakdowns=base_model_breakdowns,
            meta_info=meta_info,
            all_feature_dict=all_feature_dict,
            ensemble_draw_prob=ensemble_draw_prob,
            ensemble_away_prob=ensemble_away_prob,
        )
