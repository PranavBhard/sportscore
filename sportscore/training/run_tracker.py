"""
Run Tracker - MongoDB-based experiment run tracking.

Sport-agnostic experiment run management. Sport-specific apps provide
the DB connection and repository.
"""

from datetime import datetime
from typing import Dict, List, Optional
import uuid

from sportscore.db.mongo import Mongo


class RunTracker:
    """Manages experiment run tracking in MongoDB."""

    def __init__(self, db=None, league=None, runs_repo=None):
        """
        Initialize RunTracker.

        Args:
            db: MongoDB database instance (optional, will create if not provided)
            league: LeagueConfig instance for league-specific collections
            runs_repo: Optional ExperimentRunsRepository instance.
                       If not provided, uses raw collection access.
        """
        if db is None:
            mongo = Mongo()
            self.db = mongo.db
        else:
            self.db = db

        self.league = league
        self._repo = runs_repo

        # Fallback: use raw collection if no repo provided
        if self._repo is None:
            coll_name = 'experiment_runs'
            if league is not None:
                coll_name = league.collections.get('experiment_runs', coll_name)
            self._collection = self.db[coll_name]
        else:
            self._collection = None

    def create_run(
        self,
        config: Dict,
        dataset_id: Optional[str],
        model_type: str,
        session_id: str,
        baseline: bool = False
    ) -> str:
        """
        Create a new experiment run.

        Args:
            config: Full experiment configuration dict
            dataset_id: ID of the dataset used (optional, can be None for stacking)
            model_type: Type of model (e.g., 'LogisticRegression', 'Stacked')
            session_id: Chat session ID
            baseline: Whether this is the baseline run

        Returns:
            run_id: Unique identifier for the run
        """
        run_id = str(uuid.uuid4())

        if baseline and self._repo is not None:
            self._repo.clear_baselines(session_id)

        run_doc = {
            'run_id': run_id,
            'created_at': datetime.utcnow(),
            'config': config,
            'dataset_id': dataset_id,
            'model_type': model_type,
            'metrics': {},
            'diagnostics': {},
            'artifacts': {},
            'baseline': baseline,
            'session_id': session_id,
            'status': 'created'
        }

        if self._repo is not None:
            self._repo.insert_one(run_doc)
        else:
            self._collection.insert_one(run_doc)

        return run_id

    def get_run(self, run_id: str) -> Optional[Dict]:
        """
        Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run document or None if not found
        """
        if self._repo is not None:
            run = self._repo.find_by_run_id(run_id)
        else:
            run = self._collection.find_one({'run_id': run_id})

        if run:
            run['_id'] = str(run['_id'])
            if 'created_at' in run and isinstance(run['created_at'], datetime):
                run['created_at'] = run['created_at'].isoformat()
        return run

    def update_run(
        self,
        run_id: str,
        metrics: Optional[Dict] = None,
        diagnostics: Optional[Dict] = None,
        artifacts: Optional[Dict] = None,
        status: Optional[str] = None
    ) -> bool:
        """
        Update a run with results.

        Args:
            run_id: Run identifier
            metrics: Metrics dict
            diagnostics: Diagnostics dict
            artifacts: Artifacts dict
            status: Status string (created, running, completed, failed)

        Returns:
            True if update was successful
        """
        update_dict = {}

        if metrics is not None:
            update_dict['metrics'] = metrics
        if diagnostics is not None:
            update_dict['diagnostics'] = diagnostics
        if artifacts is not None:
            update_dict['artifacts'] = artifacts
        if status is not None:
            update_dict['status'] = status

        if not update_dict:
            return False

        if self._repo is not None:
            return self._repo.update_run(run_id, update_dict)
        else:
            result = self._collection.update_one(
                {'run_id': run_id},
                {'$set': update_dict}
            )
            return result.modified_count > 0

    def set_baseline(self, run_id: str, session_id: str) -> bool:
        """
        Set a run as the baseline for a session.

        Args:
            run_id: Run identifier
            session_id: Chat session ID

        Returns:
            True if successful
        """
        if self._repo is not None:
            return self._repo.set_baseline(run_id, session_id)

        # Fallback: raw collection
        self._collection.update_many(
            {'session_id': session_id, 'baseline': True},
            {'$set': {'baseline': False}}
        )
        result = self._collection.update_one(
            {'run_id': run_id},
            {'$set': {'baseline': True}}
        )
        return result.modified_count > 0

    def get_baseline(self, session_id: str) -> Optional[Dict]:
        """
        Get the baseline run for a session.

        Args:
            session_id: Chat session ID

        Returns:
            Baseline run document or None
        """
        if self._repo is not None:
            run = self._repo.find_baseline(session_id)
        else:
            run = self._collection.find_one({'session_id': session_id, 'baseline': True})

        if run:
            run['_id'] = str(run['_id'])
            if 'created_at' in run and isinstance(run['created_at'], datetime):
                run['created_at'] = run['created_at'].isoformat()

        return run

    def list_runs(
        self,
        session_id: Optional[str] = None,
        model_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        List runs with optional filters.

        Args:
            session_id: Filter by session ID
            model_type: Filter by model type
            date_from: Filter runs created after this date
            date_to: Filter runs created before this date
            limit: Maximum number of runs to return

        Returns:
            List of run documents
        """
        if self._repo is not None:
            runs = self._repo.find_by_date_range(
                date_from=date_from,
                date_to=date_to,
                session_id=session_id,
                model_type=model_type,
                limit=limit
            )
        else:
            query = {}
            if session_id:
                query['session_id'] = session_id
            if model_type:
                query['model_type'] = model_type
            if date_from or date_to:
                query['created_at'] = {}
                if date_from:
                    query['created_at']['$gte'] = date_from
                if date_to:
                    query['created_at']['$lte'] = date_to

            cursor = self._collection.find(query).sort('created_at', -1).limit(limit)
            runs = list(cursor)

        for run in runs:
            run['_id'] = str(run['_id'])
            if 'created_at' in run and isinstance(run['created_at'], datetime):
                run['created_at'] = run['created_at'].isoformat()

        return runs

    def get_run_count(self, session_id: str) -> int:
        """
        Get the number of runs for a session.

        Args:
            session_id: Chat session ID

        Returns:
            Number of runs
        """
        if self._repo is not None:
            return self._repo.count_by_session(session_id)
        else:
            return self._collection.count_documents({'session_id': session_id})
