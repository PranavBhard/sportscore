# Sportscore - Shared Core Infrastructure

## What This Package Is
Sportscore is the sport-agnostic shared library for multi-sport analytics platforms.
It provides base classes and infrastructure that sport-specific apps (basketball, hockey, etc.) extend.

## Architecture Principles

### This package contains ONLY sport-agnostic code
- Database connectivity and repository patterns (`db/`)
- ML model training, evaluation, and ensemble infrastructure (`training/`)
- Feature registry framework - the schema, NOT sport-specific definitions (`features/`)
- League config loader - the mechanism, NOT sport-specific YAML (`league_config.py`)
- Prediction market integrations (`market/`)
- Pipeline orchestration patterns (`pipeline/`)
- Base service classes (`services/`)

### What does NOT belong here
- Sport-specific stat definitions (PER, Corsi, xG, etc.)
- Sport-specific feature sets or feature blocks
- Sport-specific data source parsers
- Sport-specific business rules
- League YAML config files (those live in the sport app)

### How sport apps use sportscore
```python
# Install as editable dependency
pip install -e ~/Documents/sportscore

# In basketball app:
from sportscore.db import Mongo, BaseRepository
from sportscore.features import BaseFeatureRegistry, StatDefinition, StatCategory
from sportscore.training import create_model_with_c, evaluate_model_combo
from sportscore.league_config import BaseLeagueConfig, load_league_config

class BasketballFeatureRegistry(BaseFeatureRegistry):
    STAT_DEFINITIONS = { ... }  # PER, ELO, pace, etc.

class HockeyFeatureRegistry(BaseFeatureRegistry):
    STAT_DEFINITIONS = { ... }  # Corsi, Fenwick, xG, etc.
```

## Python Environment
- Requires Python 3.12+
- Install for development: `pip install -e .`
- Install with ML extras: `pip install -e ".[ml]"`
- Install with market extras: `pip install -e ".[market]"`
- Install everything: `pip install -e ".[all]"`
