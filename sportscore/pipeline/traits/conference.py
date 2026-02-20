"""
Conference pipeline trait — refresh team conference data from ESPN groups endpoint.

Reusable across any college sport whose ESPN API provides a ``/groups`` endpoint
with conference membership.
"""

import re
import time

import requests


def refresh_team_conferences(
    db,
    league_config,
    dry_run: bool = False,
) -> dict:
    """
    Refresh team conference data from ESPN groups endpoint.

    The /teams endpoint doesn't include conference info for college sports.
    The /groups endpoint returns conferences with their member teams, which
    is the only reliable source for this data.

    Args:
        db: MongoDB database instance
        league_config: League configuration (any BaseLeagueConfig subclass)
        dry_run: Preview without database changes

    Returns:
        Dict with statistics: success, updated, conferences
    """
    espn = league_config.espn
    base_url_site = espn.get('base_url_site', 'https://site.api.espn.com')
    sport_path = espn.get('sport_path', league_config.league_id)
    sport = league_config.sport

    url = f"{base_url_site}/apis/site/v2/sports/{sport}/{sport_path}/groups"

    teams_coll = league_config.collections.get("teams", "teams")

    print(f"Fetching conference data from ESPN groups endpoint...")

    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  Error fetching groups: {e}")
        return {'success': False, 'error': str(e), 'updated': 0, 'conferences': 0}

    # Parse conferences and their teams
    conf_count = 0
    updated = 0

    top_groups = data.get('groups', [])
    if not top_groups:
        top_groups = data.get('children', [])

    conferences = []
    for group in top_groups:
        children = group.get('children', [])
        if children:
            conferences.extend(children)
        elif group.get('teams'):
            conferences.append(group)

    if not conferences:
        print(f"  No conferences found in groups response (keys: {list(data.keys())})")
        return {'success': False, 'error': 'No conferences in response', 'updated': 0, 'conferences': 0}

    for conf in conferences:
        conf_name = conf.get('name') or conf.get('shortName') or conf.get('abbreviation')
        if not conf_name:
            continue

        conf_count += 1
        teams = conf.get('teams', [])

        for team_wrapper in teams:
            team = team_wrapper.get('team', team_wrapper)
            team_id = str(team.get('id', ''))
            if not team_id:
                continue

            if dry_run:
                abbrev = team.get('abbreviation', '?')
                print(f"  [DRY RUN] {abbrev} (id={team_id}) -> {conf_name}")
                updated += 1
                continue

            result = db[teams_coll].update_one(
                {'team_id': team_id},
                {'$set': {'conference': conf_name}},
            )
            if result.matched_count > 0:
                updated += 1

    print(f"  {'[DRY RUN] ' if dry_run else ''}Found {conf_count} conferences, updated {updated} teams")

    # --- Fallback: fetch conference from individual team endpoints ---
    missing_teams = list(db[teams_coll].find(
        {'conference': {'$exists': False}},
        {'team_id': 1, 'abbreviation': 1},
    ))

    if missing_teams:
        standing_re = re.compile(r'(?:in|of)\s+(.+)$', re.IGNORECASE)
        fallback_updated = 0
        print(f"  {len(missing_teams)} teams still missing conference — fetching individually...")

        for team_doc in missing_teams:
            team_id = team_doc.get('team_id')
            if not team_id:
                continue

            team_url = (
                f"{base_url_site}/apis/site/v2/sports/{sport}/"
                f"{sport_path}/teams/{team_id}"
            )
            try:
                tresp = requests.get(
                    team_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10
                )
                tresp.raise_for_status()
                tdata = tresp.json().get('team', tresp.json())
                standing = tdata.get('standingSummary', '')
                m = standing_re.search(standing)
                if m:
                    conf_name_fb = m.group(1).strip()
                    if dry_run:
                        print(f"  [DRY RUN] {team_doc.get('abbreviation')} -> {conf_name_fb}")
                    else:
                        db[teams_coll].update_one(
                            {'team_id': team_id},
                            {'$set': {'conference': conf_name_fb}},
                        )
                    fallback_updated += 1
            except Exception:
                pass  # skip silently — team may not exist on ESPN
            time.sleep(0.1)  # rate-limit

        print(f"  {'[DRY RUN] ' if dry_run else ''}Fallback: updated {fallback_updated}/{len(missing_teams)} teams")
        updated += fallback_updated

    return {'success': True, 'updated': updated, 'conferences': conf_count}
