"""
Model Factory

Factory functions for creating sklearn classifier models.
Sport-agnostic - works for any binary classification task.
"""

import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    if 'XGBoostError' in str(type(e).__name__) or 'libomp' in str(e).lower():
        warnings.warn(f"XGBoost is not available: {e}. Install OpenMP with 'brew install libomp' if needed.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except (ImportError, Exception):
    CATBOOST_AVAILABLE = False


def create_model_with_c(model_type: str, c_value: float = None):
    """
    Create a classifier model with optional C-value.

    Args:
        model_type: Name of the model type (e.g., 'LogisticRegression', 'GradientBoosting')
        c_value: C-value for regularization (only applies to LogisticRegression, SVM)

    Returns:
        sklearn classifier instance

    Raises:
        ValueError: If model_type is unknown
        ImportError: If required library for model_type is not installed
    """
    if model_type == 'LogisticRegression':
        c = c_value if c_value is not None else 0.1
        return LogisticRegression(C=c, max_iter=10000, random_state=42)
    elif model_type == 'SVM':
        c = c_value if c_value is not None else 0.1
        return SVC(C=c, probability=True, random_state=42)
    elif model_type == 'GradientBoosting':
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        return xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    elif model_type == 'LightGBM':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        return lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    elif model_type == 'CatBoost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        return cb.CatBoostClassifier(iterations=100, random_state=42, verbose=False)
    elif model_type == 'NaiveBayes':
        return GaussianNB()
    elif model_type == 'NeuralNetwork':
        return MLPClassifier(max_iter=10000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
