"""Shared CLI framework for sport apps."""

from sportscore.cli.base import (
    BaseCommand,
    SportsCLI,
    parse_list,
    parse_float_list,
    parse_seasons,
    format_table,
    print_metrics,
    print_feature_importances,
)

__all__ = [
    "BaseCommand",
    "SportsCLI",
    "parse_list",
    "parse_float_list",
    "parse_seasons",
    "format_table",
    "print_metrics",
    "print_feature_importances",
]
