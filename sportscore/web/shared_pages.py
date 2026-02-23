"""Page routes for the unified sportscore web app.

Each route renders a shared template, passing data from ``g.services``
(populated by the sport plugin's ``get_web_services``).
"""

from flask import g, render_template


def register_shared_pages(app):
    """Register page routes on *app*."""

    # ── Modeling pages ──────────────────────────────────────────────────

    @app.route("/<league_id>/model-config")
    def model_config(league_id=None):
        feat = g.services.features
        feature_sets = feat.feature_sets or {}
        feature_set_descriptions = feat.feature_set_descriptions or {}
        available_features = feat.available_features or {
            f for feats in feature_sets.values() for f in feats
        }
        return render_template(
            "model_config.html",
            feature_sets=feature_sets,
            feature_set_descriptions=feature_set_descriptions,
            default_config=None,
            available_features=available_features,
        )

    @app.route("/<league_id>/model-config-points")
    def model_config_points(league_id=None):
        feat = g.services.features
        feature_sets = feat.feature_sets or {}
        feature_set_descriptions = feat.feature_set_descriptions or {}
        available_features = feat.available_features or {
            f for feats in feature_sets.values() for f in feats
        }
        return render_template(
            "model_config_points.html",
            feature_sets=feature_sets,
            feature_set_descriptions=feature_set_descriptions,
            default_config=None,
            available_features=available_features,
        )

    @app.route("/<league_id>/ensemble-config")
    def ensemble_config(league_id=None):
        feat = g.services.features
        feature_sets = feat.feature_sets or {}
        feature_set_descriptions = feat.feature_set_descriptions or {}
        return render_template(
            "ensemble_config.html",
            feature_sets=feature_sets,
            feature_set_descriptions=feature_set_descriptions,
        )

    @app.route("/<league_id>/master-training")
    def master_training(league_id=None):
        return render_template("master_training.html")

    # ── Data cache pages ────────────────────────────────────────────────

    @app.route("/<league_id>/elo-manager")
    def elo_manager(league_id=None):
        return render_template("elo_manager.html", stats={})

    @app.route("/<league_id>/cached-league-stats")
    def cached_league_stats_manager(league_id=None):
        return render_template("cached_league_stats.html", summary={})

    @app.route("/<league_id>/espn-db-audit")
    def espn_db_audit_manager(league_id=None):
        return render_template("espn_db_audit.html")

    # ── Market pages (sport-agnostic — work with or without league) ─────

    @app.route("/market-dashboard", strict_slashes=False)
    @app.route("/<league_id>/market-dashboard")
    def market_dashboard(league_id=None):
        return render_template("market_dashboard.html")

    @app.route("/market-bins", strict_slashes=False)
    @app.route("/<league_id>/market-bins")
    def market_bins(league_id=None):
        return render_template("market_bins.html")
