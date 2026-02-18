"""
Models Module.

Sport-agnostic ML model infrastructure:
- BaseArtifactLoader: Model loading, caching, and artifact management
- BaseEnsemblePredictor: Ensemble prediction with meta-model stacking
"""

from sportscore.models.base_artifact_loader import BaseArtifactLoader
from sportscore.models.base_ensemble import BaseEnsemblePredictor

__all__ = [
    "BaseArtifactLoader",
    "BaseEnsemblePredictor",
]
