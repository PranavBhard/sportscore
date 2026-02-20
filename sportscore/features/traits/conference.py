"""
Conference feature trait — reusable conference feature bundle.

Provides conference-specific stats for any college sport league with
conference structures: conf_wins, conf_margin, conf_gp, same_conf.

Opt-in via league YAML ``extra_features.stats`` containing conference stat names.
"""

from typing import Dict, Optional

from sportscore.features.base_registry import StatCategory, StatDefinition


# ---------------------------------------------------------------------------
# Stat definitions (replaces YAML entries for these 4 stats)
# ---------------------------------------------------------------------------

CONFERENCE_STAT_DEFINITIONS: Dict[str, StatDefinition] = {
    "conf_wins": StatDefinition(
        name="conf_wins",
        category=StatCategory.DERIVED,
        description="Conference win percentage (wins vs same-conference opponents / total conference games)",
        stat_type="custom",
        handler="compute_conference",
        valid_calc_weights={"avg"},
        valid_time_periods={"season"},
        valid_perspectives={"away", "diff", "home"},
    ),
    "same_conf": StatDefinition(
        name="same_conf",
        category=StatCategory.DERIVED,
        description="Binary: 1 if both teams are in the same conference, 0 otherwise",
        stat_type="custom",
        handler="compute_conference",
        valid_calc_weights={"binary"},
        valid_time_periods={"none"},
        valid_perspectives={"none"},
    ),
    "conf_gp": StatDefinition(
        name="conf_gp",
        category=StatCategory.DERIVED,
        description="Number of conference games played this season",
        stat_type="custom",
        handler="compute_conference",
        valid_calc_weights={"raw"},
        valid_time_periods={"season"},
        valid_perspectives={"away", "diff", "home"},
    ),
    "conf_margin": StatDefinition(
        name="conf_margin",
        category=StatCategory.DERIVED,
        description="Average margin per game against same-conference opponents this season",
        stat_type="custom",
        handler="compute_conference",
        valid_calc_weights={"avg"},
        valid_time_periods={"season"},
        valid_perspectives={"away", "diff", "home"},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_perspective(home_val, away_val, perspective):
    """Combine home/away values based on perspective."""
    if perspective == "diff":
        if home_val is not None and away_val is not None:
            return home_val - away_val
        return None
    elif perspective == "home":
        return home_val
    elif perspective == "away":
        return away_val
    elif perspective == "none":
        return home_val
    return None


def _get_team_conference(team, context):
    """Look up a team's conference. Uses cache from context."""
    conf_cache = context.get("conference_cache", {})
    if team in conf_cache:
        return conf_cache[team]

    db = context.get("db")
    league = context.get("league")
    if db is None or league is None:
        return None

    teams_coll = league.collections.get("teams", "teams")
    doc = db[teams_coll].find_one(
        {"$or": [{"abbreviation": team}, {"team_id": team}]},
        {"conference": 1},
    )
    conf = doc.get("conference") if doc else None
    conf_cache[team] = conf
    return conf


def _get_conference_teams(conference, context):
    """Get all team names/ids in a conference."""
    conf_teams_cache = context.get("conf_teams_cache", {})
    if conference in conf_teams_cache:
        return conf_teams_cache[conference]

    db = context.get("db")
    league = context.get("league")
    if db is None or league is None:
        return set()

    teams_coll = league.collections.get("teams", "teams")
    docs = db[teams_coll].find({"conference": conference}, {"abbreviation": 1, "team_id": 1})
    names = set()
    for d in docs:
        if d.get("abbreviation"):
            names.add(d["abbreviation"])
        if d.get("team_id"):
            names.add(str(d["team_id"]))
    conf_teams_cache[conference] = names
    return names


def _is_conf_game(team, game, conf_teams):
    """Check if a game is a conference game for the given team."""
    home_name = game.get("homeTeam", {}).get("name", "")
    away_name = game.get("awayTeam", {}).get("name", "")
    home_id = str(game.get("homeTeam", {}).get("team_id", ""))
    away_id = str(game.get("awayTeam", {}).get("team_id", ""))

    is_home = (home_name == team or home_id == team)
    opp_name = away_name if is_home else home_name
    opp_id = away_id if is_home else home_id

    return opp_name in conf_teams or opp_id in conf_teams


def _get_conf_win_pct(team, games, context):
    """Conference win percentage."""
    conf = _get_team_conference(team, context)
    if not conf:
        return 0.5
    conf_teams = _get_conference_teams(conf, context)

    wins, total = 0, 0
    for game in games:
        if not _is_conf_game(team, game, conf_teams):
            continue
        total += 1
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        is_home = game.get("homeTeam", {}).get("name") == team
        if (is_home and home_pts > away_pts) or (not is_home and away_pts > home_pts):
            wins += 1

    return wins / total if total > 0 else 0.5


def _get_conf_avg_margin(team, games, context):
    """Average point margin in conference games."""
    conf = _get_team_conference(team, context)
    if not conf:
        return 0.0
    conf_teams = _get_conference_teams(conf, context)

    total_margin, total = 0.0, 0
    for game in games:
        if not _is_conf_game(team, game, conf_teams):
            continue
        total += 1
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        is_home = game.get("homeTeam", {}).get("name") == team
        total_margin += (home_pts - away_pts) if is_home else (away_pts - home_pts)

    return total_margin / total if total > 0 else 0.0


def _get_conf_games_count(team, games, context):
    """Count conference games played."""
    conf = _get_team_conference(team, context)
    if not conf:
        return 0
    conf_teams = _get_conference_teams(conf, context)
    return sum(1 for g in games if _is_conf_game(team, g, conf_teams))


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def compute_conference(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute conference-specific features.

    Handles: conf_wins, same_conf, conf_margin, conf_gp
    """
    if stat_name == "same_conf":
        home_conf = _get_team_conference(home_team, context)
        away_conf = _get_team_conference(away_team, context)
        return 1.0 if (home_conf and away_conf and home_conf == away_conf) else 0.0

    engine = context.get("engine")
    reference_date = context.get("reference_date")

    # All conf stats use season games
    h_games = engine._window_games(home_games, "season", reference_date) if engine else home_games
    a_games = engine._window_games(away_games, "season", reference_date) if engine else away_games

    if stat_name == "conf_wins":
        home_val = _get_conf_win_pct(home_team, h_games, context)
        away_val = _get_conf_win_pct(away_team, a_games, context)
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name == "conf_margin":
        home_val = _get_conf_avg_margin(home_team, h_games, context)
        away_val = _get_conf_avg_margin(away_team, a_games, context)
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name == "conf_gp":
        home_val = float(_get_conf_games_count(home_team, h_games, context))
        away_val = float(_get_conf_games_count(away_team, a_games, context))
        return _apply_perspective(home_val, away_val, perspective)

    return None


CONFERENCE_HANDLERS = {"compute_conference": compute_conference}
