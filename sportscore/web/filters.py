"""Shared Jinja2 template filters."""

from datetime import datetime


def register_filters(app):
    """Register custom Jinja2 filters on *app*."""

    @app.template_filter("gametime_et")
    def gametime_et_filter(gametime):
        """Convert UTC datetime to Eastern time and format as '7:00 PM'."""
        if not gametime:
            return None
        try:
            from pytz import timezone, utc

            if isinstance(gametime, str):
                if "T" in gametime:
                    gametime = datetime.fromisoformat(
                        gametime.replace("Z", "+00:00")
                    )
                else:
                    return None
            if gametime.tzinfo is None:
                gametime = utc.localize(gametime)
            eastern = timezone("US/Eastern")
            et_time = gametime.astimezone(eastern)
            return et_time.strftime("%-I:%M %p").lstrip("0")
        except Exception:
            return None
