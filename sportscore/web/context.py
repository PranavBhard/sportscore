"""Request context: before_request and context_processor for the web app."""

from typing import Dict

from flask import g, request, current_app

from sportscore.web.services import SportServices

# Process-lifetime caches.
_db_cache: Dict[str, object] = {}                # sport_name -> db connection
_services_cache: Dict[str, SportServices] = {}   # league_id -> SportServices


def _get_db_for_sport(sport_name, plugin):
    """Return a cached DB connection for the given sport."""
    if sport_name not in _db_cache:
        _db_cache[sport_name] = plugin.get_db_factory()()
    return _db_cache[sport_name]


def register_context(app):
    """Attach before_request and context_processor to *app*."""

    @app.before_request
    def set_league_context():
        """Parse league from URL path and populate ``g``."""
        league_map = current_app.config["LEAGUE_MAP"]
        plugins = current_app.config["PLUGINS"]

        # Extract first path segment as potential league_id
        path_parts = request.path.strip("/").split("/")
        league_id = path_parts[0] if path_parts and path_parts[0] else None

        if league_id not in league_map:
            # Not a league-scoped route (e.g. "/" or static files).
            # Provide a SportServices with market defaults so global
            # market endpoints (/api/market/dashboard, etc.) still work.
            from sportscore.web.service_helpers import make_market_helpers
            g.league_id = None
            g.league = None
            g.sport_name = None
            g.plugin = None
            g.db = None
            g.services = SportServices(market=make_market_helpers())
            return

        sport_name = league_map[league_id]
        plugin = plugins[sport_name]
        league = plugin.get_league_loader()(league_id)
        db = _get_db_for_sport(sport_name, plugin)

        # Cache the full services dict per league (process-lifetime).
        # build_services() is expensive: imports, closures, caching setup.
        if league_id not in _services_cache:
            _services_cache[league_id] = plugin.get_web_services(db, league)
        services = _services_cache[league_id]

        g.league_id = league_id
        g.league = league
        g.sport_name = sport_name
        g.plugin = plugin
        g.db = db
        g.services = services

        # Fix cross-sport route conflicts: multiple sport blueprints register
        # the same URL patterns (e.g. /<league_id>/api/predict). Flask matches
        # whichever blueprint was registered first. If the matched endpoint
        # belongs to the wrong sport, reroute to the correct one.
        endpoint = request.endpoint
        if endpoint and "." in endpoint:
            bp_name, view_name = endpoint.split(".", 1)
            if bp_name != sport_name and bp_name in plugins:
                correct_endpoint = f"{sport_name}.{view_name}"
                if correct_endpoint in current_app.view_functions:
                    return current_app.view_functions[correct_endpoint](
                        **request.view_args
                    )

    @app.context_processor
    def inject_template_globals():
        """Inject league context + nav data into all templates."""
        sport_leagues = current_app.config.get("SPORT_LEAGUES", {})

        sport_nav_items = []
        if getattr(g, "plugin", None) and getattr(g, "league_id", None):
            try:
                sport_nav_items = g.plugin.get_nav_items(g.league_id)
            except Exception:
                pass

        return {
            "league": getattr(g, "league", None),
            "league_id": getattr(g, "league_id", None),
            "sport_name": getattr(g, "sport_name", None),
            "sport_leagues": sport_leagues,
            "sport_nav_items": sport_nav_items,
        }
