"""
MongoDB Connection - Core Infrastructure

Provides the MongoDB connection wrapper used by all sport apps.
"""

from urllib.parse import urlparse
from pymongo import MongoClient
from sportscore.config import config


class Mongo:
    """
    MongoDB connection wrapper.

    Usage:
        mongo = Mongo()
        collection = mongo.db.games
        results = collection.find({...})
    """

    def __init__(self):
        conn_str = config["mongo_conn_str"]
        self.client = MongoClient(conn_str)

        # Parse database name from connection string (required)
        parsed = urlparse(conn_str)
        db_name = parsed.path.lstrip('/') if parsed.path and parsed.path != '/' else None
        if not db_name:
            raise ValueError("Database name must be specified in mongo_conn_str (e.g., mongodb://localhost:27017/sports_analytics)")
        self.db = self.client[db_name]
