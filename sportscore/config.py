"""
Configuration management.

Loads settings from environment variables. Sport-specific apps can extend
this config dict with their own keys.
"""

import os

config = {
    "mongo_conn_str": os.environ.get("MONGO_CONN_STR") or os.environ.get("MONGODB_URI", ""),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "KALSHI_API_KEY": os.environ.get("KALSHI_API_KEY", ""),
    "KALSHI_PRIVATE_KEY_DIR": os.environ.get("KALSHI_PRIVATE_KEY_DIR", ""),
    "KALSHI_READ_KEY": os.environ.get("KALSHI_READ_KEY", ""),
    "KALSHI_READ_PRIVATE_KEY_DIR": os.environ.get("KALSHI_READ_PRIVATE_KEY_DIR", ""),
    "STAKE_TOKEN": os.environ.get("STAKE_TOKEN", ""),
    "SERP_API_KEY": os.environ.get("SERP_API_KEY", ""),
}


def register_config_keys(keys: dict):
    """
    Register additional config keys from a sport-specific app.

    Args:
        keys: Dict of key -> value pairs to add to the global config.
              Existing keys are NOT overwritten.
    """
    for k, v in keys.items():
        if k not in config:
            config[k] = v
