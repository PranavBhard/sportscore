"""
Geocoding pipeline trait — geocode venues missing location coordinates.

Reusable across any sport whose venues have ESPN-standard address fields
(fullName, address.city, address.state, address.street, address.postalCode).
"""

import time


def geocode_missing_venues(db, league_config, dry_run=False):
    """
    Geocode venues that are missing location coordinates.

    Uses Nominatim geocoding with ESPN address fields (fullName, city, state).
    Rate-limited to 1 request/second per Nominatim ToS.

    Args:
        db: MongoDB database instance
        league_config: League configuration (any BaseLeagueConfig subclass)
        dry_run: Preview without database changes

    Returns:
        Dict with statistics: geocoded, failed, skipped, already_have_location
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    venues_coll = league_config.collections.get("venues", "venues")
    venues_col = db[venues_coll]

    already_have = venues_col.count_documents({'location.lat': {'$exists': True}})
    missing = list(venues_col.find(
        {'$or': [
            {'location': {'$exists': False}},
            {'location.lat': {'$exists': False}},
        ]},
        {'venue_guid': 1, 'fullName': 1, 'address': 1}
    ))

    if not missing:
        return {'geocoded': 0, 'failed': 0, 'skipped': 0, 'already_have_location': already_have}

    geolocator = Nominatim(user_agent="sportscore/1.0", timeout=10)

    geocoded = 0
    failed = 0
    skipped = 0

    for i, venue in enumerate(missing):
        venue_guid = venue.get('venue_guid')
        full_name = venue.get('fullName', '')
        address = venue.get('address') or {}

        if not full_name:
            skipped += 1
            continue

        city = address.get('city', '') if isinstance(address, dict) else ''
        state = address.get('state', '') if isinstance(address, dict) else ''

        # Primary query: arena name + city + state
        query_parts = [full_name]
        if city:
            query_parts.append(city)
        if state:
            query_parts.append(state)
        query = ", ".join(query_parts)

        # Rate limit: 1 req/sec
        if i > 0:
            time.sleep(1)

        try:
            location = geolocator.geocode(query)

            # Fallback: try street address if arena name didn't work
            if not location and address.get('street'):
                fallback_parts = [address['street']]
                if city:
                    fallback_parts.append(city)
                if state:
                    postal = address.get('postalCode', '')
                    fallback_parts.append(f"{state} {postal}".strip())
                fallback_query = ", ".join(fallback_parts)
                time.sleep(1)
                location = geolocator.geocode(fallback_query)

            if location:
                if not dry_run:
                    venues_col.update_one(
                        {'venue_guid': venue_guid},
                        {'$set': {'location': {'lat': location.latitude, 'lon': location.longitude}}}
                    )
                geocoded += 1
            else:
                failed += 1
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    Warning: Geocoding failed for '{full_name}': {e}")
            failed += 1
        except Exception as e:
            print(f"    Warning: Unexpected geocoding error for '{full_name}': {e}")
            failed += 1

    print(f"  Geocoded {geocoded}/{len(missing)} venues ({failed} failed, {skipped} skipped)")

    return {
        'geocoded': geocoded,
        'failed': failed,
        'skipped': skipped,
        'already_have_location': already_have,
    }
