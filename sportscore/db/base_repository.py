"""
Base Repository - Foundation for all data access classes.

Provides common patterns for MongoDB operations while allowing
repositories to define collection-specific query methods.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult

T = TypeVar('T', bound=Dict[str, Any])


class BaseRepository(Generic[T]):
    """
    Base repository providing common MongoDB operations.

    Subclasses should:
    1. Set `collection_name` class attribute
    2. Add domain-specific query methods

    Example:
        class GamesRepository(BaseRepository):
            collection_name = 'nhl_games'

            def find_by_date(self, date: str) -> List[Dict]:
                return self.find({'date': date})
    """

    collection_name: str = None

    def __init__(self, db: Database, collection_name: Optional[str] = None):
        """
        Create a repository bound to a MongoDB collection.

        Args:
            db: MongoDB database instance
            collection_name: Optional override for the collection to bind to.
                If omitted, uses the subclass's `collection_name` attribute.
        """
        effective_collection = collection_name or self.collection_name
        if not effective_collection:
            raise ValueError(f"{self.__class__.__name__} must define collection_name")
        self._db = db
        self._collection: Collection = db[effective_collection]

    @property
    def collection(self) -> Collection:
        """Direct access to PyMongo collection for advanced operations."""
        return self._collection

    # --- Basic CRUD Operations ---

    def find_one(self, query: Dict, projection: Dict = None) -> Optional[T]:
        """Find a single document matching the query."""
        return self._collection.find_one(query, projection)

    def find(
        self,
        query: Dict,
        projection: Dict = None,
        sort: List = None,
        limit: int = 0,
        skip: int = 0
    ) -> List[T]:
        """Find documents matching the query."""
        cursor = self._collection.find(query, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def find_by_id(self, doc_id: Any) -> Optional[T]:
        """Find document by _id."""
        return self.find_one({'_id': doc_id})

    def insert_one(self, document: Dict) -> InsertOneResult:
        """Insert a single document."""
        return self._collection.insert_one(document)

    def insert_many(self, documents: List[Dict]) -> List:
        """Insert multiple documents."""
        if not documents:
            return []
        result = self._collection.insert_many(documents)
        return result.inserted_ids

    def update_one(
        self,
        query: Dict,
        update: Dict,
        upsert: bool = False
    ) -> UpdateResult:
        """Update a single document."""
        return self._collection.update_one(query, update, upsert=upsert)

    def update_many(
        self,
        query: Dict,
        update: Dict,
        upsert: bool = False
    ) -> UpdateResult:
        """Update multiple documents."""
        return self._collection.update_many(query, update, upsert=upsert)

    def delete_one(self, query: Dict) -> DeleteResult:
        """Delete a single document."""
        return self._collection.delete_one(query)

    def delete_many(self, query: Dict) -> DeleteResult:
        """Delete multiple documents."""
        return self._collection.delete_many(query)

    def count(self, query: Dict = None) -> int:
        """Count documents matching query."""
        return self._collection.count_documents(query or {})

    def exists(self, query: Dict) -> bool:
        """Check if any document matches the query."""
        return self.find_one(query, {'_id': 1}) is not None

    # --- Aggregation ---

    def aggregate(self, pipeline: List[Dict]) -> List[Dict]:
        """Run an aggregation pipeline."""
        return list(self._collection.aggregate(pipeline))

    # --- Utility Methods ---

    def distinct(self, field: str, query: Dict = None) -> List:
        """Get distinct values for a field."""
        return self._collection.distinct(field, query or {})

    def create_index(self, keys: List, **kwargs) -> str:
        """Create an index on the collection."""
        return self._collection.create_index(keys, **kwargs)
