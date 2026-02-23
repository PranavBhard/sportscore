"""App factory for the unified sportscore web application."""

import os

from flask import Flask, redirect, render_template
from jinja2 import ChoiceLoader

from sportscore.cli.discovery import discover_plugins, build_league_map
from sportscore.web.context import register_context
from sportscore.web.filters import register_filters
from sportscore.web.shared_pages import register_shared_pages
from sportscore.web.shared_routes import register_shared_routes


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "dev-secret-key-change-in-production"
    )

    # ── Plugin discovery ────────────────────────────────────────────────
    plugins = discover_plugins()
    if not plugins:
        raise RuntimeError(
            "No sport plugins found. Install at least one sport app "
            "(e.g. pip install -e ~/Documents/basketball)."
        )

    league_map = build_league_map(plugins)

    # Build sport -> leagues mapping for the grouped nav dropdown
    sport_leagues = {}
    for sport_name, plugin in plugins.items():
        for league_id in plugin.get_leagues():
            loader = plugin.get_league_loader()
            try:
                cfg = loader(league_id)
            except Exception:
                continue
            sport_leagues.setdefault(sport_name, []).append(
                {
                    "id": league_id,
                    "display_name": cfg.display_name,
                    "logo_url": getattr(cfg, "logo_url", None),
                }
            )

    app.config.update(
        PLUGINS=plugins,
        LEAGUE_MAP=league_map,
        SPORT_LEAGUES=sport_leagues,
    )

    # ── Register core layers ────────────────────────────────────────────
    register_context(app)
    register_filters(app)
    register_shared_pages(app)
    register_shared_routes(app)

    # ── Root redirect ───────────────────────────────────────────────────
    # Prefer a league that has a blueprint (games page), falling back to
    # the first available league.
    _preferred = ["nba", "epl", "mlb"]

    @app.route("/")
    def root_index():
        for lid in _preferred:
            if lid in league_map:
                return redirect(f"/{lid}/")
        return redirect(f"/{next(iter(league_map))}/")

    # ── Sport-specific blueprints ───────────────────────────────────────
    loaders = [app.jinja_loader]
    for sport_name, plugin in plugins.items():
        bp = plugin.get_web_blueprint()
        if bp is not None:
            app.register_blueprint(bp)
            if hasattr(bp, "jinja_loader") and bp.jinja_loader:
                loaders.append(bp.jinja_loader)
    app.jinja_loader = ChoiceLoader(loaders)

    # ── Index dispatcher ──────────────────────────────────────────────
    # Each sport provides its own index view via get_index_view().
    # A single app-level route dispatches to the correct one.
    @app.route("/<league_id>/")
    def index(league_id=None):
        from flask import g
        if g.plugin:
            view = g.plugin.get_index_view()
            if view:
                return view(league_id=league_id)
        return render_template("fallback_index.html")

    return app
