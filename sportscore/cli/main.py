"""Unified sportscore CLI entry point.

Usage:
    sportscore --list-leagues
    sportscore full_data_pipeline nba --skip-espn
    sportscore generate_training_data wcbb --add --features "vegas_*"
"""

import argparse
import sys
from typing import Dict, List

from sportscore.cli.commands import GENERIC_COMMANDS
from sportscore.cli.discovery import (
    SportCommand,
    SportPlugin,
    build_league_map,
    discover_plugins,
)


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Discover plugins
    plugins = discover_plugins()

    if not plugins:
        print(
            "No sport plugins installed.\n"
            "Install a sport app with entry points, e.g.:\n"
            "  pip install -e ~/Documents/basketball",
            file=sys.stderr,
        )
        return 1

    league_map = build_league_map(plugins)

    # Collect all commands across plugins (sport-specific)
    all_commands: Dict[str, Dict[str, SportCommand]] = {}  # cmd_name -> {sport -> cmd}
    for sport_name, plugin in plugins.items():
        for cmd_name, cmd in plugin.get_commands().items():
            all_commands.setdefault(cmd_name, {})[sport_name] = cmd

    # Register generic commands (available for all sports)
    generic_cmd_names = set()
    for cmd in GENERIC_COMMANDS:
        generic_cmd_names.add(cmd.name)
        for sport_name in plugins:
            all_commands.setdefault(cmd.name, {})[sport_name] = cmd

    # --- Handle --list-leagues early (no subcommand needed) ---
    if "--list-leagues" in argv:
        print("Available leagues:\n")
        for league_id in sorted(league_map):
            sport = league_map[league_id]
            print(f"  {league_id:<10} ({sport})")
        print(f"\nAvailable commands: {', '.join(sorted(all_commands))}")
        return 0

    # --- Two-pass argparse ---
    # First pass: extract command name and league
    first_parser = argparse.ArgumentParser(add_help=False)
    first_parser.add_argument("command", nargs="?", default=None)
    first_parser.add_argument("league", nargs="?", default=None)
    first_args, remaining = first_parser.parse_known_args(argv)

    cmd_name = first_args.command
    league_id = first_args.league

    # Show help if no command
    if not cmd_name or cmd_name in ("-h", "--help"):
        _print_help(all_commands, league_map)
        return 0

    # Validate command
    if cmd_name not in all_commands:
        print(f"Unknown command: {cmd_name}", file=sys.stderr)
        print(f"Available commands: {', '.join(sorted(all_commands))}",
              file=sys.stderr)
        return 1

    # If no league but --help in remaining, show command help with league placeholder
    if (not league_id or league_id.startswith("-")) and (
        "-h" in remaining or "--help" in remaining
    ):
        # Show the command's help by building its parser
        any_cmd = next(iter(all_commands[cmd_name].values()))
        help_parser = argparse.ArgumentParser(
            prog=f"sportscore {cmd_name}",
            description=any_cmd.help,
        )
        help_parser.add_argument("league", help=f"League ID ({', '.join(sorted(league_map))})")
        any_cmd.add_arguments(help_parser)
        help_parser.parse_args(["--help"])
        return 0  # unreachable, parse_args --help exits

    # Validate league
    if not league_id or league_id.startswith("-"):
        print(f"Usage: sportscore {cmd_name} <league> [options]", file=sys.stderr)
        print(f"Available leagues: {', '.join(sorted(league_map))}",
              file=sys.stderr)
        return 1

    if league_id not in league_map:
        print(f"Unknown league: {league_id}", file=sys.stderr)
        print(f"Available leagues: {', '.join(sorted(league_map))}",
              file=sys.stderr)
        return 1

    # Resolve sport -> plugin -> command
    sport_name = league_map[league_id]
    if sport_name not in all_commands[cmd_name]:
        print(
            f"Command '{cmd_name}' is not available for sport '{sport_name}'",
            file=sys.stderr,
        )
        return 1

    command = all_commands[cmd_name][sport_name]

    # Second pass: full parser with command-specific arguments
    full_parser = argparse.ArgumentParser(
        prog=f"sportscore {cmd_name}",
        description=command.help,
    )
    full_parser.add_argument("league", help="League ID")
    command.add_arguments(full_parser)
    full_args = full_parser.parse_args([league_id] + remaining)

    # Run — generic commands get league_loader/db_factory from the plugin
    try:
        if cmd_name in generic_cmd_names:
            plugin = plugins[sport_name]
            return command.run(
                full_args,
                league_loader=plugin.get_league_loader(),
                db_factory=plugin.get_db_factory(),
            )
        return command.run(full_args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def _print_help(
    all_commands: Dict[str, Dict[str, SportCommand]],
    league_map: Dict[str, str],
) -> None:
    print("usage: sportscore <command> <league> [options]")
    print("       sportscore --list-leagues\n")
    print("commands:")
    for cmd_name in sorted(all_commands):
        # Pick any sport's command for the help text
        cmd = next(iter(all_commands[cmd_name].values()))
        print(f"  {cmd_name:<30} {cmd.help}")
    print(f"\nleagues: {', '.join(sorted(league_map))}")
    print("\nRun 'sportscore <command> <league> --help' for command-specific options.")


if __name__ == "__main__":
    sys.exit(main())
