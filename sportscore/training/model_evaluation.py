"""
Model Evaluation

Functions for evaluating classifier models with cross-validation
and time-based calibration. Fully sport-agnostic.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from sportscore.training.model_factory import create_model_with_c


def evaluate_model_combo(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    c_value: float = None,
    n_splits: int = 5
) -> dict:
    """
    Evaluate a model/C-value combination using time-series cross-validation.

    Args:
        X: Feature matrix (scaled)
        y: Target vector
        model_type: Model type (e.g., 'LogisticRegression')
        c_value: C-value for regularization (optional)
        n_splits: Number of CV splits

    Returns:
        Dict with accuracy, std, log_loss, brier, etc., including per-fold metrics
    """
    model = create_model_with_c(model_type, c_value)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    accuracies = []
    log_losses = []
    briers = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_copy = create_model_with_c(model_type, c_value)
        model_copy.fit(X_train, y_train)

        y_pred = model_copy.predict(X_val)
        y_proba = model_copy.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred) * 100
        ll = log_loss(y_val, y_proba)
        brier = brier_score_loss(y_val, y_proba[:, 1])

        accuracies.append(acc)
        log_losses.append(ll)
        briers.append(brier)

    return {
        'model_type': model_type,
        'c_value': c_value,
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'accuracy_folds': accuracies,
        'log_loss_mean': float(np.mean(log_losses)),
        'log_loss_std': float(np.std(log_losses)),
        'log_loss_folds': log_losses,
        'brier_mean': float(np.mean(briers)),
        'brier_std': float(np.std(briers)),
        'brier_folds': briers,
        'n_folds': n_splits
    }


def evaluate_model_combo_with_calibration(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    y: np.ndarray,
    model_type: str,
    c_value: float = None,
    calibration_method: str = 'isotonic',
    calibration_years: list = None,
    evaluation_year: int = None,
    logger=None,
    season_start_month: int = 10,
) -> dict:
    """
    Evaluate a model with time-based calibration using year-based temporal splits.

    Splits data chronologically by year:
    - Train: All data before first calibration_year
    - Calibrate: Data from calibration_years (combined)
    - Evaluate: Data from evaluation_year

    Args:
        df: DataFrame with Year, Month, Day columns for sorting
        X_scaled: Scaled feature matrix (must match df rows)
        y: Target vector (must match df rows)
        model_type: Model type (e.g., 'LogisticRegression')
        c_value: C-value for regularization (optional)
        calibration_method: 'isotonic' or 'sigmoid'
        calibration_years: List of years to use for calibration
        evaluation_year: Year to use for evaluation
        logger: Logger instance for logging
    """
    from sklearn.calibration import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as LR

    if calibration_years is None:
        raise ValueError("calibration_years must be specified for time-based calibration")

    if not isinstance(calibration_years, list):
        calibration_years = [calibration_years]

    if logger:
        logger.info(f"Using time-based calibration (method: {calibration_method})")
        logger.info(f"Calibration years: {calibration_years}, Evaluation year: {evaluation_year}")

    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy[['Year', 'Month', 'Day']])
    df_copy['SeasonStartYear'] = np.where(df_copy['Month'] >= season_start_month, df_copy['Year'], df_copy['Year'] - 1)

    if evaluation_year is None:
        raise ValueError("evaluation_year must be specified for time-based calibration")

    cal_seasons = [int(y) for y in calibration_years]
    eval_season = int(evaluation_year)
    earliest_cal_year = min(cal_seasons)

    train_mask = df_copy['SeasonStartYear'] < earliest_cal_year
    X_train = X_scaled[train_mask]
    y_train = y[train_mask]

    cal_mask = df_copy['SeasonStartYear'].isin(cal_seasons)
    X_cal = X_scaled[cal_mask]
    y_cal = y[cal_mask]

    eval_mask = df_copy['SeasonStartYear'] == eval_season
    X_eval = X_scaled[eval_mask]
    y_eval = y[eval_mask]

    if logger:
        logger.info(f"Train set: {len(X_train)} games (SeasonStartYear < {earliest_cal_year})")
        logger.info(f"Calibration set: {len(X_cal)} games (Seasons in {cal_seasons})")
        logger.info(f"Evaluation set: {len(X_eval)} games (SeasonStartYear == {eval_season})")

        if len(X_train) == 0:
            logger.warning(f"Training set is empty! Check begin_year and calibration_years.")
        if len(X_cal) == 0:
            logger.warning(f"Calibration set is empty! Check calibration_years: {calibration_years}")
        if len(X_eval) == 0:
            logger.warning(f"Evaluation set is empty! Check evaluation_year: {evaluation_year}")

    model = create_model_with_c(model_type, c_value)
    model.fit(X_train, y_train)

    y_proba_raw_cal = model.predict_proba(X_cal)

    if calibration_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_proba_raw_cal[:, 1], y_cal)
    elif calibration_method == 'sigmoid':
        sigmoid_calibrator = LR()
        sigmoid_calibrator.fit(y_proba_raw_cal[:, 1].reshape(-1, 1), y_cal)

        class SigmoidCalibratedModel:
            def __init__(self, base_model, calibrator):
                self.base_model = base_model
                self.calibrator = calibrator

            def predict(self, X):
                return self.base_model.predict(X)

            def predict_proba(self, X):
                raw_proba = self.base_model.predict_proba(X)
                calibrated_1 = self.calibrator.predict_proba(raw_proba[:, 1].reshape(-1, 1))[:, 1]
                calibrated_1 = np.clip(calibrated_1, 0.0, 1.0)
                return np.column_stack([1 - calibrated_1, calibrated_1])

        calibrator = SigmoidCalibratedModel(model, sigmoid_calibrator)
    else:
        if logger:
            logger.warning(f"Unknown calibration method: {calibration_method}, using raw probabilities")
        calibrator = None

    y_proba_raw_eval = model.predict_proba(X_eval)

    if calibrator is not None:
        if calibration_method == 'isotonic':
            y_proba_calibrated = np.column_stack([
                1 - calibrator.predict(y_proba_raw_eval[:, 1]),
                calibrator.predict(y_proba_raw_eval[:, 1])
            ])
        else:
            y_proba_calibrated = calibrator.predict_proba(X_eval)
    else:
        y_proba_calibrated = y_proba_raw_eval

    y_pred_calibrated = (y_proba_calibrated[:, 1] >= 0.5).astype(int)

    acc = accuracy_score(y_eval, y_pred_calibrated) * 100
    ll = log_loss(y_eval, y_proba_calibrated)
    brier = brier_score_loss(y_eval, y_proba_calibrated[:, 1])

    if logger:
        logger.info(f"Calibrated results on season starting {eval_season} - Accuracy: {acc:.2f}%, Log Loss: {ll:.4f}, Brier: {brier:.4f}")

    return {
        'model_type': model_type,
        'c_value': c_value,
        'accuracy_mean': float(acc),
        'accuracy_std': 0.0,
        'accuracy_folds': [acc],
        'log_loss_mean': float(ll),
        'log_loss_std': 0.0,
        'log_loss_folds': [ll],
        'brier_mean': float(brier),
        'brier_std': 0.0,
        'brier_folds': [brier],
        'n_folds': 1,
        'split_type': 'time_based_calibration',
        'calibration_years': calibration_years,
        'evaluation_year': evaluation_year,
        'train_set_size': len(X_train),
        'calibrate_set_size': len(X_cal),
        'eval_set_size': len(X_eval),
    }


def compute_feature_importance(model, feature_names: list, model_type: str) -> dict:
    """
    Compute feature importance for a trained model.

    Returns dict mapping feature names to importance scores.
    """
    importance = {}

    if model_type in ('GradientBoosting', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, imp in zip(feature_names, importances):
                importance[name] = float(imp)
    elif model_type == 'LogisticRegression':
        if hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            for name, coef in zip(feature_names, coefs):
                importance[name] = float(coef)

    return importance
