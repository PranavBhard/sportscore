"""
Experiment Configuration Schemas - Pydantic models for typed config validation.

Sport-agnostic training configuration. The FeatureConfig.blocks validator
is intentionally open here -- sport-specific apps should add their own
validation of valid block names by subclassing FeatureConfig.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model type and hyperparameters for classification."""
    type: Literal[
        'LogisticRegression',
        'RandomForest',
        'GradientBoosting',
        'SVM',
        'NaiveBayes',
        'NeuralNetwork',
        'XGBoost',
        'LightGBM',
        'CatBoost'
    ]

    c_value: Optional[float] = Field(None, ge=0.001, le=100.0, description="Regularization for LogisticRegression/SVM")
    n_estimators: Optional[int] = Field(None, ge=10, le=1000, description="Number of trees for ensemble methods")
    max_depth: Optional[int] = Field(None, ge=1, le=20, description="Max depth for tree models")
    learning_rate: Optional[float] = Field(None, ge=0.001, le=1.0, description="Learning rate for boosting methods")

    @validator('c_value')
    def validate_c_value(cls, v, values):
        if v is not None and values.get('type') not in ['LogisticRegression', 'SVM']:
            raise ValueError(f"c_value only applies to LogisticRegression and SVM, not {values.get('type')}")
        return v


class RegressionModelConfig(BaseModel):
    """Model type and hyperparameters for score/goal regression."""
    type: Literal[
        'Ridge',
        'ElasticNet',
        'RandomForest',
        'XGBoost'
    ]
    target: Literal['home_away', 'margin'] = Field(
        'home_away',
        description="'home_away' trains separate models per side, 'margin' trains on differential"
    )

    alpha: Optional[float] = Field(None, ge=0.01, le=1000.0, description="Regularization for Ridge/ElasticNet")
    l1_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="L1 ratio for ElasticNet")
    n_estimators: Optional[int] = Field(None, ge=10, le=1000, description="Number of trees for ensemble methods")
    max_depth: Optional[int] = Field(None, ge=1, le=20, description="Max depth for tree models")

    @validator('alpha')
    def validate_alpha(cls, v, values):
        if v is not None and values.get('type') not in ['Ridge', 'ElasticNet']:
            raise ValueError(f"alpha only applies to Ridge and ElasticNet, not {values.get('type')}")
        return v

    @validator('l1_ratio')
    def validate_l1_ratio(cls, v, values):
        if v is not None and values.get('type') != 'ElasticNet':
            raise ValueError(f"l1_ratio only applies to ElasticNet, not {values.get('type')}")
        return v


class FeatureConfig(BaseModel):
    """
    Feature selection configuration.

    Sport-specific apps should subclass this and add a blocks validator
    that checks against their own valid feature block names.
    """
    blocks: List[str] = Field(
        default_factory=list,
        description="Feature set names (sport-specific, e.g., 'shooting_efficiency' or 'possession_control')"
    )
    features: Optional[List[str]] = Field(
        None,
        description="Specific feature names (overrides blocks if provided)"
    )
    diff_mode: Literal['home_minus_away', 'away_minus_home', 'absolute', 'mixed', 'all'] = 'home_minus_away'
    flip_augment: bool = Field(False, description="Include perspective-flipped training examples")


class SplitConfig(BaseModel):
    """Data splitting strategy."""
    type: Literal['time_split', 'rolling_cv', 'year_based_calibration'] = 'year_based_calibration'

    test_size: Optional[float] = Field(None, ge=0.1, le=0.5)
    n_splits: Optional[int] = Field(None, ge=3, le=10)
    train_end_year: Optional[int] = Field(default=2022, ge=2000, le=2100)
    calibration_years: Optional[List[int]] = Field(default=[2023])
    evaluation_year: Optional[int] = Field(default=2024, ge=2000, le=2100)
    begin_year: Optional[int] = Field(2012, ge=2000, le=2100)
    min_games_played: Optional[int] = Field(15, ge=0, le=100)
    exclude_seasons: Optional[List[int]] = Field(None, description="Season start years to exclude from training data")

    @validator('type')
    def validate_split_type(cls, v, values):
        if v == 'time_split' and values.get('test_size') is None:
            values['test_size'] = 0.2
        elif v == 'rolling_cv' and values.get('n_splits') is None:
            values['n_splits'] = 5
        elif v == 'year_based_calibration':
            if values.get('train_end_year') is None:
                values['train_end_year'] = 2022
            if values.get('calibration_years') is None:
                values['calibration_years'] = [2023]
            if values.get('evaluation_year') is None:
                values['evaluation_year'] = 2024
        return v


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration."""
    scaler: Literal['standard', 'none'] = 'standard'
    impute: Literal['median', 'mean', 'zero', 'none'] = 'median'
    clip_outliers: bool = Field(False, description="Clip outliers to 3 standard deviations")
    clip_range: Optional[tuple] = Field(None, description="Custom clip range (min, max)")


class ConstraintsConfig(BaseModel):
    """Resource and data constraints."""
    max_features: Optional[int] = Field(None, ge=1, le=1000)
    max_train_rows: Optional[int] = Field(None, ge=100, le=1000000)
    max_time_window_days: Optional[int] = Field(None, ge=1, le=3650)


class StackingConfig(BaseModel):
    """Configuration for stacking ensemble models."""
    base_run_ids: List[str] = Field(
        ...,
        min_items=2,
        description="List of run_ids for base models to stack"
    )
    meta_model_type: Literal['LogisticRegression'] = 'LogisticRegression'
    meta_c_value: Optional[float] = Field(0.1, ge=0.001, le=100.0)
    stacking_mode: Literal['naive', 'informed'] = 'naive'
    meta_features: Optional[List[str]] = Field(
        None,
        description="Feature names to include in meta-model (only for stacking_mode='informed')"
    )


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    task: Literal['binary_home_win', 'score_regression', 'points_regression', 'stacking'] = 'binary_home_win'

    model: Optional[ModelConfig] = Field(None, description="Classification model config")
    regression_model: Optional[RegressionModelConfig] = Field(None, description="Regression model config")
    stacking: Optional[StackingConfig] = None

    features: FeatureConfig
    splits: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    constraints: Optional[ConstraintsConfig] = None

    use_time_calibration: bool = Field(True, description="Use time-based probability calibration")
    calibration_method: Literal['isotonic', 'sigmoid'] = 'sigmoid'

    description: Optional[str] = None
    tags: Optional[List[str]] = None

    @validator('model')
    def validate_model_for_classification(cls, v, values):
        task = values.get('task', 'binary_home_win')
        if task == 'binary_home_win' and v is None:
            raise ValueError("model is required when task='binary_home_win'")
        return v

    @validator('regression_model')
    def validate_regression_model(cls, v, values):
        task = values.get('task', 'binary_home_win')
        if task == 'score_regression' and v is None:
            raise ValueError("regression_model is required when task='score_regression'")
        return v

    @validator('stacking')
    def validate_stacking_for_task(cls, v, values):
        task = values.get('task', 'binary_home_win')
        if task == 'stacking' and v is None:
            raise ValueError("stacking is required when task='stacking'")
        return v

    class Config:
        extra = 'allow'


class DatasetSpec(BaseModel):
    """Dataset specification for build_dataset()."""
    label: Literal['home_win'] = 'home_win'
    unit: Literal['game'] = 'game'

    feature_blocks: List[str] = Field(default_factory=list)
    individual_features: Optional[List[str]] = None

    begin_year: Optional[int] = Field(default=2012, ge=2000, le=2100)
    end_year: Optional[int] = None
    begin_date: Optional[str] = None
    end_date: Optional[str] = None

    min_games_played: Optional[int] = None
    exclude_preseason: bool = True
    exclude_seasons: Optional[List[int]] = Field(None, description="Season start years to exclude from dataset")

    diff_mode: Literal['home_minus_away', 'away_minus_home', 'absolute', 'mixed', 'all'] = 'home_minus_away'

    class Config:
        extra = 'allow'


# Backward-compatible alias
PointsRegressionModelConfig = RegressionModelConfig
