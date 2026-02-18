"""
Shared CLI framework for sport apps.

Provides BaseCommand, SportsCLI, and helper functions so that
football/basketball CLIs only define args and handle() logic.
League and DB are loaded lazily after parse (so --help is fast).
"""

import argparse
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


def parse_list(value: str) -> List[str]:
    """Comma-separated string → list of stripped non-empty strings."""
    if not value or not value.strip():
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_float_list(value: str) -> List[float]:
    """Comma-separated string → list of floats."""
    if not value or not value.strip():
        return []
    out = []
    for s in value.split(","):
        s = s.strip()
        if s:
            try:
                out.append(float(s))
            except ValueError:
                pass
    return out


def parse_seasons(value: str) -> List[int]:
    """Comma-separated string → list of ints (season years)."""
    if not value or not value.strip():
        return []
    out = []
    for s in value.split(","):
        s = s.strip()
        if s:
            try:
                out.append(int(s))
            except ValueError:
                pass
    return out


def format_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
) -> str:
    """Format as aligned text table with header underline. Auto-computes widths if not given."""
    if not headers:
        return ""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for row in rows:
                if i < len(row):
                    w = max(w, len(str(row[i])))
            col_widths.append(max(w, 1))
    parts = []
    header_line = "".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers) if i < len(col_widths))
    parts.append(header_line.rstrip())
    parts.append("-" * len(header_line))
    for row in rows:
        line = "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)) if i < len(row) and i < len(col_widths))
        parts.append(line.rstrip())
    return "\n".join(parts)


def print_metrics(metrics: Dict, label: str = "METRICS") -> None:
    """Print accuracy, log_loss, brier, AUC from a metrics dict."""
    if not metrics:
        return
    print(f"\n{label}")
    print("-" * 40)
    for k in ("accuracy_mean", "accuracy_std", "log_loss_mean", "brier_mean", "auc"):
        v = metrics.get(k)
        if v is not None:
            if isinstance(v, float) and 0 < v < 1 and "accuracy" in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v}")
    print()


def print_feature_importances(
    importances: Dict[str, float],
    label: str = "FEATURE IMPORTANCES",
    top_n: int = 20,
) -> None:
    """Print sorted feature importance scores, top N."""
    if not importances:
        return
    print(f"\n{label} (top {top_n})")
    print("-" * 50)
    sorted_items = sorted(importances.items(), key=lambda x: (x[1] or 0), reverse=True)
    for name, score in sorted_items[:top_n]:
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {name}: {score_str}")
    print()


class BaseCommand(ABC):
    """Base for CLI subcommands. League and db are injected by the framework."""

    name: str = ""
    help: str = ""
    description: str = ""
    epilog: str = ""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments. League positional is added by framework."""
        pass

    @abstractmethod
    def handle(self, args: argparse.Namespace, league: Any, db: Any) -> None:
        """Execute the command. league and db are pre-loaded."""
        ...

    def error(self, message: str) -> None:
        """Print error to stderr and exit(1)."""
        print(message, file=sys.stderr)
        sys.exit(1)


class SportsCLI:
    """
    Registers commands, sets up argparse subparsers, dispatches.
    league_loader(league_id) and db_factory() are called only after parse.
    """

    def __init__(
        self,
        prog: str,
        description: str,
        league_loader: Callable[[str], Any],
        db_factory: Callable[[], Any],
        available_leagues: Optional[Callable[[], List[str]]] = None,
    ):
        self.prog = prog
        self.description = description
        self.league_loader = league_loader
        self.db_factory = db_factory
        self.available_leagues = available_leagues or (lambda: [])
        self._commands: Dict[str, BaseCommand] = {}

    def register(self, command: BaseCommand) -> None:
        """Register a subcommand."""
        if not command.name:
            raise ValueError("Command must have a non-empty name")
        self._commands[command.name] = command

    def run(self, argv: Optional[List[str]] = None) -> None:
        """Parse argv (default sys.argv[1:]), load league/db only after parse, dispatch."""
        parser = argparse.ArgumentParser(
            prog=self.prog,
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        for name, cmd in self._commands.items():
            subp = subparsers.add_parser(
                name,
                help=cmd.help or cmd.description,
                description=cmd.description or cmd.help,
                epilog=cmd.epilog,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            subp.add_argument("league", help="League ID (e.g., nfl, nba)")
            cmd.add_arguments(subp)

        args = parser.parse_args(argv)

        if not args.command:
            parser.print_help()
            sys.exit(0)

        cmd = self._commands.get(args.command)
        if not cmd:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

        league_id = getattr(args, "league", None)
        if not league_id:
            print("Error: league is required", file=sys.stderr)
            subparsers.choices[args.command].print_help()
            sys.exit(1)

        try:
            league = self.league_loader(league_id)
        except Exception as e:
            print(f"Error loading league {league_id}: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            db = self.db_factory()
        except Exception as e:
            print(f"Error connecting to database: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            cmd.handle(args, league, db)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
