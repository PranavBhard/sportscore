"""
Kalshi prediction market integration.

Provides unauthenticated access to market data for display purposes.
Uses the public API at api.elections.kalshi.com (no auth required for reads).

Sport-specific ticker building/parsing logic belongs in the sport app,
not here. This module provides only the generic API client and cache.
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock

import requests

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    data: Any
    expires_at: float


class SimpleCache:
    """Thread-safe in-memory cache with TTL."""

    def __init__(self, default_ttl: int = 60):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._cache[key]
                return None
            return entry.data

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.default_ttl
        with self._lock:
            self._cache[key] = CacheEntry(
                data=value,
                expires_at=time.time() + ttl
            )

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class KalshiPublicClient:
    """
    Unauthenticated client for Kalshi public market data.

    Uses the public API which doesn't require authentication for
    reading market data, events, and series information.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Kalshi API error: {e}")
            raise

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        return self._get(f"/events/{event_ticker}")

    def get_events(self, series_ticker: Optional[str] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None) -> Dict[str, Any]:
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._get(f"/markets/{ticker}")

    def get_markets(self, event_ticker: Optional[str] = None,
                    series_ticker: Optional[str] = None,
                    limit: int = 100,
                    cursor: Optional[str] = None) -> Dict[str, Any]:
        params = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params)
