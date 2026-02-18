"""
Generic market/trading connector for prediction market APIs.
"""
import os
import time
import base64
from typing import Optional, Dict, List, Any

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class MarketConnector:
    """Connector for prediction market trading API."""

    HOST = "https://api.elections.kalshi.com"
    BASE_PATH = "/trade-api/v2"
    BASE_URL = f"{HOST}{BASE_PATH}"

    def __init__(self, config: Dict[str, str]):
        self.api_key = config.get("KALSHI_API_KEY")
        private_key_dir = config.get("KALSHI_PRIVATE_KEY_DIR")

        if not self.api_key:
            raise ValueError("KALSHI_API_KEY not found in config")
        if not private_key_dir:
            raise ValueError("KALSHI_PRIVATE_KEY_DIR not found in config")

        self.private_key_path = self._find_private_key(private_key_dir)
        self._private_key = None

    def _find_private_key(self, directory: str) -> str:
        if not os.path.isdir(directory):
            if os.path.isfile(directory):
                return directory
            raise ValueError(f"Private key directory not found: {directory}")

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if filename.endswith(('.pem', '.txt', '.key')):
                    return filepath
                try:
                    with open(filepath, 'r') as f:
                        content = f.read(50)
                        if 'PRIVATE KEY' in content:
                            return filepath
                except Exception:
                    continue

        raise ValueError(f"No private key file found in: {directory}")

    def _load_private_key(self):
        if self._private_key is None:
            with open(self.private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        return self._private_key

    def _sign_request(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        full_path = f"{self.BASE_PATH}{path}"
        message = f"{timestamp}{method}{full_path}"

        private_key = self._load_private_key()
        signature = private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode()
        }

    def _get(self, path: str, params: Optional[Dict] = None, debug: bool = False) -> Dict[str, Any]:
        headers = self._sign_request("GET", path)
        url = f"{self.BASE_URL}{path}"

        if debug:
            print(f"=== DEBUG REQUEST ===")
            print(f"URL: {url}")
            print(f"Signed path: {self.BASE_PATH}{path}")

        response = requests.get(url, headers=headers, params=params)

        if debug:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text[:500]}")

        response.raise_for_status()
        return response.json()

    def _post(self, path: str, body: Dict) -> Dict[str, Any]:
        headers = self._sign_request("POST", path)
        headers["Content-Type"] = "application/json"
        response = requests.post(
            f"{self.BASE_URL}{path}",
            headers=headers,
            json=body
        )
        response.raise_for_status()
        return response.json()

    def _delete(self, path: str) -> Dict[str, Any]:
        headers = self._sign_request("DELETE", path)
        response = requests.delete(
            f"{self.BASE_URL}{path}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    # ==================== Debug/Test ====================

    def test_auth(self):
        """Test authentication and print debug info."""
        import time

        print("=== Auth Test ===")
        print(f"API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"Private key path: {self.private_key_path}")

        with open(self.private_key_path, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            print(f"Key first line: {lines[0]}")
            print(f"Key last line: {lines[-1]}")

        pk = self._load_private_key()
        print(f"Key type: {type(pk).__name__}")
        print(f"Key size: {pk.key_size} bits")

        timestamp = str(int(time.time() * 1000))
        path = "/portfolio/balance"
        full_path = f"{self.BASE_PATH}{path}"
        message = f"{timestamp}GET{full_path}"
        print(f"\nMessage to sign: {message}")

        print("\n=== Making Request ===")
        try:
            result = self._get("/portfolio/balance", debug=True)
            print(f"Success! Balance: {result}")
        except Exception as e:
            print(f"Error: {e}")

    # ==================== Account/Portfolio ====================

    def get_balance(self) -> Dict[str, Any]:
        """Get current account balance (in cents)."""
        return self._get("/portfolio/balance")

    def get_positions(self, ticker: Optional[str] = None,
                      event_ticker: Optional[str] = None,
                      limit: int = 100) -> Dict[str, Any]:
        """Get current positions."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        return self._get("/portfolio/positions", params)

    # ==================== Transaction History ====================

    def get_fills(self,
                  ticker: Optional[str] = None,
                  order_id: Optional[str] = None,
                  min_ts: Optional[int] = None,
                  max_ts: Optional[int] = None,
                  limit: int = 100,
                  cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Get fill history (matched trades).

        Args:
            ticker: Filter by market ticker
            order_id: Filter by order ID
            min_ts: Minimum timestamp (ms)
            max_ts: Maximum timestamp (ms)
            limit: Max results to return
            cursor: Pagination cursor
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self._get("/portfolio/fills", params)

    def get_settlements(self,
                        ticker: Optional[str] = None,
                        event_ticker: Optional[str] = None,
                        min_ts: Optional[int] = None,
                        max_ts: Optional[int] = None,
                        limit: int = 100,
                        cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Get settlement history (market resolutions).

        Args:
            ticker: Filter by market ticker
            event_ticker: Filter by event ticker
            min_ts: Minimum timestamp (ms)
            max_ts: Maximum timestamp (ms)
            limit: Max results to return
            cursor: Pagination cursor
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self._get("/portfolio/settlements", params)

    # ==================== Orders ====================

    def get_orders(self, ticker: Optional[str] = None,
                   event_ticker: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None) -> Dict[str, Any]:
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._get("/portfolio/orders", params)

    def place_order(self, ticker: str, action: str, side: str, count: int,
                    order_type: str = "limit", yes_price: Optional[int] = None,
                    no_price: Optional[int] = None,
                    expiration_ts: Optional[int] = None,
                    client_order_id: Optional[str] = None) -> Dict[str, Any]:
        body = {
            "ticker": ticker, "action": action, "side": side,
            "count": count, "type": order_type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order by ID."""
        return self._delete(f"/portfolio/orders/{order_id}")

    def batch_create_orders(self, orders: List[Dict]) -> Dict[str, Any]:
        """
        Create multiple orders in a batch.

        Args:
            orders: List of order dictionaries (same format as place_order)
        """
        return self._post("/portfolio/orders/batched", {"orders": orders})

    def batch_cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        """Cancel multiple orders by ID."""
        return self._delete("/portfolio/orders/batched")

    # ==================== Market Data ====================

    def get_markets(self, event_ticker: Optional[str] = None,
                    series_ticker: Optional[str] = None,
                    status: Optional[str] = None,
                    limit: int = 100,
                    cursor: Optional[str] = None) -> Dict[str, Any]:
        params = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._get(f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        """Get orderbook for a market."""
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_trades(self,
                   ticker: Optional[str] = None,
                   min_ts: Optional[int] = None,
                   max_ts: Optional[int] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Get public trade history.

        Args:
            ticker: Filter by market ticker
            min_ts: Minimum timestamp (ms)
            max_ts: Maximum timestamp (ms)
            limit: Max results
            cursor: Pagination cursor
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets/trades", params)

    def get_events(self, series_ticker: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None) -> Dict[str, Any]:
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params)

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        return self._get(f"/events/{event_ticker}")
