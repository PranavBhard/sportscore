"""Plugin discovery for the unified sportscore CLI."""

import sys
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Dict, List


class SportCommand(ABC):
    """Base for commands registered via sport plugins.

    Unlike BaseCommand (which receives league + db from the framework),
    SportCommand manages its own lifecycle — suitable for pipelines and
    other heavyweight operations.
    """

    name: str = ""
    help: str = ""

    def add_arguments(self, parser) -> None:
        """Add command-specific arguments to the parser."""

    @abstractmethod
    def run(self, args) -> int:
        """Execute the command. Returns exit code (0 = success)."""
        ...


class SportPlugin(ABC):
    """Protocol that each sport app implements to register with sportscore CLI."""

    @abstractmethod
    def get_leagues(self) -> List[str]:
        """Return league IDs this sport provides (e.g. ['nba', 'cbb', 'wcbb'])."""
        ...

    @abstractmethod
    def get_commands(self) -> Dict[str, SportCommand]:
        """Return command_name -> SportCommand mapping (sport-specific commands)."""
        ...

    @abstractmethod
    def get_league_loader(self):
        """Return a callable(league_id) -> LeagueConfig."""
        ...

    @abstractmethod
    def get_db_factory(self):
        """Return a callable() -> db connection."""
        ...

    def get_ingestion_pipeline(self, league_id, **kwargs):
        """Return a BasePipeline that does ESPN pull + enrichment (no training).

        Override in sport plugins to enable the ``db_ingestion`` generic command.
        Returns *None* by default (command will report "not supported").

        Expected kwargs: seasons, max_workers, skip_espn, skip_post,
        dry_run, verbose.
        """
        return None

    # -- Web integration (optional) ------------------------------------------

    def get_web_blueprint(self):
        """Return a Flask Blueprint with sport-specific routes, or None."""
        return None

    def get_web_services(self, db, league) -> "SportServices":
        """Return a ``SportServices`` container for shared web routes.

        Sport plugins override this to populate the service groups they
        support (model, market, features, elo, data, jobs).  Unused groups
        default to empty sub-dataclasses (all fields None), and shared_routes
        returns 501 for those endpoints.
        """
        from sportscore.web.services import SportServices
        return SportServices()

    def get_index_view(self):
        """Return the index (games page) view function, or None.

        The returned callable should accept ``league_id=None`` and return
        a Flask response.  The app-level ``/<league_id>/`` route dispatches
        to this so that each sport renders its own games page.
        """
        return None

    def get_nav_items(self, league_id: str) -> list:
        """Return sport-specific nav items for the web UI.

        Each item: {'label': str, 'url': str, 'group': str}
        group is 'data_caches', 'modeling', 'markets', etc.
        """
        return []


def discover_plugins() -> Dict[str, SportPlugin]:
    """Discover sport plugins via entry points.

    Each sport app registers under the 'sportscore.sports' group:
        [project.entry-points."sportscore.sports"]
        basketball = "bball.cli.plugin:BasketballPlugin"

    Returns:
        sport_name -> SportPlugin instance
    """
    plugins: Dict[str, SportPlugin] = {}
    eps = entry_points()

    # Python 3.12+: entry_points() returns a SelectableGroups or dict-like
    if hasattr(eps, "select"):
        sport_eps = eps.select(group="sportscore.sports")
    else:
        sport_eps = eps.get("sportscore.sports", [])

    for ep in sport_eps:
        try:
            plugin_cls = ep.load()
            plugins[ep.name] = plugin_cls()
        except Exception as exc:
            print(f"Warning: Failed to load plugin '{ep.name}': {exc}",
                  file=sys.stderr)

    return plugins


def build_league_map(plugins: Dict[str, SportPlugin]) -> Dict[str, str]:
    """Build league_id -> sport_name lookup from all plugins.

    Raises ValueError on league ID conflicts across sports.
    """
    league_map: Dict[str, str] = {}
    for sport_name, plugin in plugins.items():
        for league_id in plugin.get_leagues():
            if league_id in league_map:
                raise ValueError(
                    f"League '{league_id}' claimed by both "
                    f"'{league_map[league_id]}' and '{sport_name}'"
                )
            league_map[league_id] = sport_name
    return league_map
