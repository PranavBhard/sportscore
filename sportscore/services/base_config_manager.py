"""
Base Config Manager - Sport-agnostic config management infrastructure.

Provides the framework for absorbing duplicate configs created by
hash-based upsert training. Sport-specific apps subclass and provide
their own classifier repository via _get_classifier_repo().
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from bson import ObjectId


class BaseConfigManager(ABC):
    """
    Abstract base for model configuration management.

    Subclasses must implement _get_classifier_repo() to provide
    the repository used for config DB operations.
    """

    def __init__(self, db, league=None):
        self.db = db
        self.league = league
        self._classifier_repo = self._get_classifier_repo()

    @abstractmethod
    def _get_classifier_repo(self):
        """Return the classifier config repository instance."""
        ...

    def absorb_duplicate_config(self, original_config_id: str, duplicate_config_id: str,
                                training_fields: Optional[List[str]] = None,
                                ensemble_ref_field: str = 'ensemble_models') -> bool:
        """
        Absorb an auto-created duplicate config into the original.

        When training by config ID, the hash-based upsert in train_model_grid
        creates a duplicate config. This method copies training results back
        to the original, repoints ensemble references, and deletes the duplicate.

        Args:
            original_config_id: The config to keep (receives training results)
            duplicate_config_id: The auto-created config to absorb and delete
            training_fields: Fields to copy (uses default list if None)
            ensemble_ref_field: Field name for ensemble model references

        Returns:
            True if successful
        """
        if training_fields is None:
            training_fields = [
                'run_id', 'trained_at', 'accuracy', 'std_dev', 'log_loss',
                'brier_score', 'auc', 'model_artifact_path', 'scaler_artifact_path',
                'features_path', 'artifacts_saved_at', 'dataset_id', 'training_csv',
                'features_ranked', 'features_ranked_by_importance', 'c_values',
                'best_c_value', 'best_c_accuracy', 'training_stats',
            ]

        duplicate_doc = self._classifier_repo.find_by_id(duplicate_config_id)
        if not duplicate_doc:
            return False

        # Copy training fields to original
        updates = {}
        for field in training_fields:
            if field in duplicate_doc:
                updates[field] = duplicate_doc[field]
        updates['updated_at'] = datetime.utcnow()

        self._classifier_repo.update_one(
            {'_id': ObjectId(original_config_id)},
            {'$set': updates}
        )

        # Repoint any ensemble references from duplicate -> original
        self._classifier_repo.update_many(
            {ensemble_ref_field: duplicate_config_id},
            {'$set': {f'{ensemble_ref_field}.$': original_config_id}}
        )

        # Delete the duplicate
        self._classifier_repo.delete_one({'_id': ObjectId(duplicate_config_id)})
        return True
