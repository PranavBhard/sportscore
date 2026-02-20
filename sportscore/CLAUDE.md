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

## Unified CLI

The `sportscore` command is the unified CLI entry point. Sport apps register as plugins via entry points.

```bash
sportscore --list-leagues                            # Show available leagues & commands
sportscore <command> <league> --help                 # Command-specific help
```

### Generic Commands (available for all sports)

| Command | Description | Example |
|---------|-------------|---------|
| `cache_league_stats` | Compute & cache league stats (pace, PER constants) | `sportscore cache_league_stats wcbb --season 2025-2026` |
| `compute_market_calibration` | Compute market calibration (Brier/log-loss) | `sportscore compute_market_calibration nba --rolling-seasons 3` |
| `compute_bin_trust` | Compute bin trust weights from portfolio P&L | `sportscore compute_bin_trust nba --bin-width 5` |

### Sport-Specific Commands (registered by sport app plugins)

Sport apps add their own commands in `<app>/cli/plugin.py`. Example (basketball):
- `full_data_pipeline` — Full ESPN data pipeline
- `generate_training_data` — Generate master training CSV

## Python Environment
- Requires Python 3.12+
- Install for development: `pip install -e .`
- Install with ML extras: `pip install -e ".[ml]"`
- Install with market extras: `pip install -e ".[market]"`
- Install everything: `pip install -e ".[all]"`
